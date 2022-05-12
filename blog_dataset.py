import random
import re
from typing import List, Dict
import os.path

import nltk
import numpy as np
import pandas as pd
from sympy import false
import torch
from torch.utils.data import Dataset
from transformers import DataCollator
from sklearn.model_selection import train_test_split

from utils import seed_for_reproducability

def get_demo_embeddings_path(no_authors):

    file_path =f"data/blog/{no_authors}_authors/demo_test_{no_authors}_authors_embeddings.pkl" 
    if os.path.exists(file_path):
        return file_path
    else:
        print(f"\nNo embeddings file in {file_path}\n")
        exit()


# create train, val and test datasets for the "n" authors with the highest number of texts
def get_datasets_for_n_authors(n, val_size, test_size, seed=42, path="./data/blog/"):
    df = pd.read_csv("./data/blog/blogtext.csv")
    df.columns = ["From", "Gender", "Age", "Topic", "Sign", "Date", "content"]
    df = df[df['content'].apply(lambda x: len(x.split())) > 0]

    authors = list(pd.DataFrame(df['From'].value_counts()[:n]).reset_index()['index'])
    new_df = df[df['From'].isin(authors)]
    new_df = new_df.dropna()

    name_to_index = {}
    name_to_no_samples = {}

    index = 0
    for name in new_df["From"]:
        if name in name_to_index:
            name_to_no_samples[name] += 1
        else:
            name_to_index[name] = index
            index += 1
            name_to_no_samples[name] = 1

    new_df["Target"] = new_df['From'].apply(lambda x: name_to_index[x])

    # following BERTAA experiments we use stratification for splitting train and test
    # that is, no. occorrences of a class in the test set wil be proportional to that in the training set
    train_val_and_test_indicies = train_test_split(new_df[["content", "Target"]], random_state=seed,
                                                   test_size=test_size, stratify=new_df["Target"])

    train_val_inds = list(train_val_and_test_indicies[0].index)
    train_val_df = new_df.loc[train_val_inds]
    # now split train into train and val
    train_and_val_indicies = train_test_split(train_val_df[["content", "Target"]], random_state=seed,
                                              test_size=(1 / (1 - test_size)) * val_size,
                                              stratify=train_val_df["Target"])
    train_inds = list(train_and_val_indicies[0].index)
    train_df = train_val_df.loc[train_inds]

    val_inds = list(train_and_val_indicies[1].index)
    val_df = train_val_df.loc[val_inds]

    test_inds = list(train_val_and_test_indicies[1].index)
    test_df = new_df.loc[test_inds]

    if path is not None:
        train_df.to_csv(f"{path}train_{n}.csv")
        val_df.to_csv(f"{path}val_{n}.csv")
        test_df.to_csv(f"{path}test_{n}.csv")

    return train_df, val_df, test_df


# Used to train a Siamese Model based on BERT we created without using sbert (from sentence-transformers)
# this model and this data set were later abandoned because the model was super slow to train compared to the sbert one
class AuthorsDatasetAV(Dataset):
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, positive_samples, negative_samples, pos_label, neg_label, tokenizer, max_source_len,
                 pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        self.pad_to_max_length = pad_to_max_length

        self.pos_label = pos_label
        self.neg_label = neg_label

        self.samples = [(p[0].strip(), p[1].strip(), pos_label) for p in positive_samples]
        self.samples.extend([(n[0].strip(), n[1].strip(), neg_label) for n in negative_samples])
        random.shuffle(self.samples)

    def __len__(self):
        """returns the length of the dataframe"""
        return len(self.samples)

    def __getitem__(self, index) -> Dict[str, List]:
        """return input ids, attention marks and target ids"""
        source_text1 = self.samples[index][0]

        # tokenizing source
        source1 = self.tokenizer(
            source_text1,
            max_length=self.max_source_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True
        )

        source_text2 = self.samples[index][1]
        source2 = self.tokenizer(
            source_text2,
            max_length=self.max_source_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True
        )

        labels = self.samples[index][2]

        return {
            "input_ids": (source1["input_ids"], source2["input_ids"]),
            "attention_mask": (source1["attention_mask"], source2["attention_mask"]),
            "labels": labels,
        }


# This is used to train/test the classification-based model It encodes texts and passes it along with a label (for
# the author of the text) and an attention mask We only consider the first 128 bert tokens, following Fabien et al. (
# 2020) in BertAA (https://aclanthology.org/2020.icon-main.16.pdf)
class AuthorsDatasetAA(Dataset):
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, df, source_tag, target_tag, tokenizer, max_source_len, pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.source_tag = source_tag
        self.source_text = df[source_tag].tolist()
        self.source_text = [sent.strip() for sent in self.source_text]

        longest_source_sequence = len(max(self.source_text, key=lambda x: len(x.split())).split())
        print("longest_source_sequence = ", longest_source_sequence)

        self.target_tag = target_tag
        self.target_classes = df[target_tag].tolist()
        unique_target_classes = set(self.target_classes)
        self.no_authors = len(unique_target_classes)
        self.author_index_to_no_samples = dict(zip(list(unique_target_classes), [0] * len(unique_target_classes)))
        for author_ind in self.target_classes:
            self.author_index_to_no_samples[author_ind] += 1

        self.one_hot_target_classes = self.get_one_hot_target_classes()
        self.pad_to_max_length = pad_to_max_length
        assert len(self.source_text) == len(self.one_hot_target_classes)

    def get_one_hot_target_classes(self):
        one_hot_targets = []
        for target_class in self.target_classes:
            one_hot_target = [1 if i == target_class else 0 for i in range(self.no_authors)]
            one_hot_targets.append(one_hot_target)
        return one_hot_targets

    def __len__(self):
        """returns the length of the dataframe"""
        return len(self.target_classes)

    def __getitem__(self, index) -> Dict[str, List]:
        """return input ids, attention marks and target ids"""
        source_text = self.source_text[index]

        # tokenizing source
        source = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True
        )

        labels = self.one_hot_target_classes[index]

        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": labels,
        }


# this allows us to do dynamic padding for batches.
# It significantly speeds up training time
class CollatorAA:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def collate_batch(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
            return seq + (max_batch_len - len(seq)) * [pad_value]

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        max_size = max([len(sample["input_ids"]) for sample in batch])
        for sample_dict in batch:
            batch_input_ids += [pad_seq(sample_dict["input_ids"], max_size, self.pad_token_id)]
            batch_attention_mask += [pad_seq(sample_dict["attention_mask"], max_size, self.pad_token_id)]
            batch_labels.append(sample_dict["labels"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.float)
        }


# this allows us to do dynamic padding for batches.
# It significantly speeds up training time
class CollatorAV:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def collate_batch(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
            return seq + (max_batch_len - len(seq)) * [pad_value]

        batch_input_ids1 = []
        batch_attention_mask1 = []

        batch_input_ids2 = []
        batch_attention_mask2 = []

        batch_labels = []

        max_size1 = max([len(sample["input_ids"][0]) for sample in batch])
        max_size2 = max([len(sample["input_ids"][1]) for sample in batch])

        for sample_dict in batch:
            batch_input_ids1 += [pad_seq(sample_dict["input_ids"][0], max_size1, self.pad_token_id)]
            batch_attention_mask1 += [pad_seq(sample_dict["attention_mask"][0], max_size1, self.pad_token_id)]

            batch_input_ids2 += [pad_seq(sample_dict["input_ids"][1], max_size2, self.pad_token_id)]
            batch_attention_mask2 += [pad_seq(sample_dict["attention_mask"][1], max_size2, self.pad_token_id)]

            batch_labels.append(sample_dict["labels"])

        return {
            "input_ids": (torch.tensor(batch_input_ids1, dtype=torch.long),
                          torch.tensor(batch_input_ids2, dtype=torch.long)),
            "attention_mask": ((torch.tensor(batch_attention_mask1, dtype=torch.long)),
                               (torch.tensor(batch_attention_mask2, dtype=torch.long))),
            "labels": torch.tensor(batch_labels, dtype=torch.float)
        }


def balance_fn(texts, labels):
    texts = list(texts)
    labels = list(labels)

    authors_indicies = sorted(set(labels))
    no_authors = len(authors_indicies)

    authors_texts = [[] for _ in range(no_authors)]
    for text, label in zip(texts, labels):
        authors_texts[label].append(text.strip())

    # find the average number of articles by each author
    no_articles_by_each_author = [labels.count(i) for i in range(no_authors)]
    mean = np.mean(no_articles_by_each_author)
    std = np.std(no_articles_by_each_author)
    length_to_balance_for = int(mean + std)

    for author_index in authors_indicies:
        current_author_texts = authors_texts[author_index]

        if len(current_author_texts) > length_to_balance_for:
            current_author_texts = current_author_texts[:length_to_balance_for]
        else:
            no_samples_to_add = length_to_balance_for - len(current_author_texts)
            # randomly oversample
            random_indicies = np.random.random_integers(low=0, high=len(current_author_texts) - 1,
                                                        size=no_samples_to_add)
            for random_index in random_indicies:
                current_author_texts.append(current_author_texts[random_index])

        authors_texts[author_index] = current_author_texts

    # convert back to texts and labels
    new_texts = []
    new_labels = []
    for author_index in authors_indicies:
        current_author_texts = authors_texts[author_index]
        current_author_labels = [author_index] * len(current_author_texts)

        new_texts.extend(current_author_texts)
        new_labels.extend(current_author_labels)

    return new_texts, new_labels


# split dataset into positive pairs (coming from same author) and negative pairs (coming from different authors)
# the approach is similar to that described in page 9 of this paper https://arxiv.org/pdf/1912.10616.pdf
# We do the following (for each author):
# - split the texts produced by each author into 4 equal sized chunks
# - merge the first two chunks to create positive/same-author pairs for the given author
# - split the third chunk into N - 1 pieces, where N is the total number of authors in the dataset
# - merge the pieces from the third chunk with random texts from the other (N-1) authors' forth chunks
# This results in an equal number of positive and negative pairs in these datasets.
def create_pos_neg_pairs(df, balance=False):
    # split as described in page ten of this paper https://arxiv.org/pdf/1912.10616.pdf
    texts = df["content"].tolist()
    labels = df["Target"].tolist()

    if balance:
        texts, labels = balance_fn(texts, labels)

    pos_pairs = []
    neg_pairs = []

    authors_indicies = sorted(set(labels))
    no_authors = len(authors_indicies)

    # list of contents for each author, for author i their content is at the ith index
    authors_chunks = []
    for i in authors_indicies:
        current_author_texts = [texts[j].strip() for j in range(len(labels)) if labels[j] == i]

        # divide authors texts into 4 chuncks
        current_author_chunks = [arr.tolist() for arr in np.array_split(current_author_texts, 4)]
        authors_chunks.append(current_author_chunks)

    for i in authors_indicies:
        chunks_for_current_author = authors_chunks[i]
        first_chunk = chunks_for_current_author[0]
        second_chunk = chunks_for_current_author[1]

        pos_pairs_for_current_author = map_chunks(first_chunk, second_chunk)
        pos_pairs.extend(pos_pairs_for_current_author)

        third_chunk = chunks_for_current_author[2]
        third_chunk_splitted = [arr.tolist() for arr in
                                np.array_split(third_chunk, min(no_authors - 1, len(third_chunk)))]

        other_author_indicies = [k for k in authors_indicies if k != i]
        for j in range(min(len(third_chunk_splitted), no_authors - 1)):
            other_author_forth_chunk = authors_chunks[other_author_indicies[j]][3]
            neg_pairs_for_current_chunk = map_chunks(third_chunk_splitted[j], other_author_forth_chunk)
            neg_pairs.extend(neg_pairs_for_current_chunk)

    return pos_pairs, neg_pairs


# this merges chunks of texts together
def map_chunks(texts1, texts2):
    # map texts between two authors to create negative pairs
    # the number of texts might not be similar
    random.shuffle(texts1)
    random.shuffle(texts2)

    return list(zip(texts1[:len(texts2)], texts2)) if len(texts2) < len(texts1) else list(
        zip(texts1, texts2[:len(texts1)]))


def get_pairs_and_labels(no_authors, split, balanced=False):
    pairs_df = pd.read_csv(
        f"./data/blog/{no_authors}_authors/{split}_pairs_{no_authors}_authors{'_balanced' if balanced else ''}.csv")
    s1s = pairs_df["s1"].tolist()
    s2s = pairs_df["s2"].tolist()
    pairs = [list(pair) for pair in zip(s1s, s2s)]
    labels = pairs_df["label"].tolist()
    return pairs, labels


def get_samples_and_labels(no_authors, split, balanced=False, demo=False):
    df = pd.read_csv(
        f"./data/blog/{no_authors}_authors/{'demo_' if demo else ''}{split}_{no_authors}_authors{'_balanced' if balanced else ''}.csv")
    samples = df["content"].tolist()
    labels = df["Target"].tolist()
    return samples, labels


# this creates all the datasets needed - more details in READ.me
def create_all_datasets(no_authors, balance=False):
    seed_for_reproducability()

    # normal AA dataset
    train_df, val_df, test_df = get_datasets_for_n_authors(n=no_authors,
                                                           val_size=0.1,
                                                           test_size=0.2)

    if balance:
        train_samples = train_df["content"]
        train_labels = train_df["Target"]
        train_samples, train_labels = balance_fn(train_samples, train_labels)
        train_df = pd.DataFrame({
            "content": train_samples,
            "Target": train_labels
        })

    train_df.to_csv(f"train_{no_authors}_authors{'_balanced' if balance else ''}.csv")
    val_df.to_csv(f"val_{no_authors}_authors.csv")
    test_df.to_csv(f"test_{no_authors}_authors.csv")

    #########
    # contrastive train dataset
    train_pos, train_neg = create_pos_neg_pairs(train_df, balance=balance)

    train_samples = [[train_pos[i][0], train_pos[i][1]] for i in range(len(train_pos))]
    train_labels = [1 for _ in range(len(train_pos))]
    train_samples.extend([[train_neg[i][0], train_neg[i][1]] for i in range(len(train_neg))])
    train_labels.extend([0 for _ in range(len(train_neg))])

    p1_samples_train = [train_samples[i][0] for i in range(len(train_samples))]
    p2_samples_train = [train_samples[i][1] for i in range(len(train_samples))]

    train_pairs_df = pd.DataFrame({
        "s1": p1_samples_train,
        "s2": p2_samples_train,
        "label": train_labels
    })
    train_pairs_df.to_csv(f"train_pairs_{no_authors}_authors{'_balanced' if balance else ''}.csv")

    #############
    # contrastive validation dataset
    val_pos, val_neg = create_pos_neg_pairs(val_df)

    val_samples = [[val_pos[i][0], val_pos[i][1]] for i in range(len(val_pos))]
    val_labels = [1 for _ in range(len(val_pos))]
    val_samples.extend([[val_neg[i][0], val_neg[i][1]] for i in range(len(val_neg))])
    val_labels.extend([0 for _ in range(len(val_neg))])

    p1_samples_val = [val_samples[i][0] for i in range(len(val_samples))]
    p2_samples_val = [val_samples[i][1] for i in range(len(val_samples))]

    val_pairs_df = pd.DataFrame({
        "s1": p1_samples_val,
        "s2": p2_samples_val,
        "label": val_labels
    })
    val_pairs_df.to_csv(f"val_pairs_{no_authors}_authors.csv")

    ###########
    # contrastive test dataset

    test_pos, test_neg = create_pos_neg_pairs(test_df)

    test_samples = [[test_pos[i][0], test_pos[i][1]] for i in range(len(test_pos))]
    test_labels = [1 for _ in range(len(test_pos))]
    test_samples.extend([[test_neg[i][0], test_neg[i][1]] for i in range(len(test_neg))])
    test_labels.extend([0 for _ in range(len(test_neg))])

    p1_samples_test = [test_samples[i][0] for i in range(len(test_samples))]
    p2_samples_test = [test_samples[i][1] for i in range(len(test_samples))]

    test_pairs_df = pd.DataFrame({
        "s1": p1_samples_test,
        "s2": p2_samples_test,
        "label": test_labels
    })
    test_pairs_df.to_csv(f"test_pairs_{no_authors}_authors.csv")


if __name__ == "__main__":
    create_all_datasets(no_authors=75, balance=True)
