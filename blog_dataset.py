import random
import re
from typing import List, Dict

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollator
from constants import *
from sklearn.model_selection import train_test_split


def get_AV_dataset_from_AA_dataset(df, m=1, path=None):
    """

    :param df:
    :param m: each positive sample will have no_authors / 2 * m many negative samples
    :param path:
    :return:
    """
    authors_indicies = set(df["Target"].tolist())

    # list of contents for each author, for author i their content is at the ith index
    authors_contents_list = []
    for i in authors_indicies:
        authors_contents_list.append(df.loc[df["Target"] == i]["content"].tolist())

    # positive matches for each author - samples coming from the same author
    # overall we have N * L positive samples, N is number of authors, L is no texts

    # todo do something here if u want to increase the number of positive samples
    positive_samples = []
    for author_content in authors_contents_list:
        for i in range(len(author_content)):
            if i + 1 < len(author_content):
                positive_samples.append((author_content[i], author_content[i + 1]))

    # negative matches, e.g. content for author 1 and content for author 2
    # overall we have L * N * (N + 1) * m / 2  negative samples
    negative_samples = []
    for i in range(len(authors_contents_list)):
        for j in range(len(authors_contents_list[i])):
            for k in range(i + 1, len(authors_contents_list)):
                if j + m - 1 < len(authors_contents_list[k]):
                    for l in range(j, j + m):
                        negative_samples.append((authors_contents_list[i][j], authors_contents_list[k][l]))
                else:
                    # choose m samples at random
                    random_indicies = np.random.random_integers(low=0, high=len(authors_contents_list[k]) - 1, size=m)
                    for ri in random_indicies:
                        negative_samples.append((authors_contents_list[i][j], authors_contents_list[k][ri]))

    if path:
        f = open(path, "wb")
        f.dump({
            "positive": positive_samples,
            "negative": negative_samples
        })

    return positive_samples, negative_samples



def get_datasets_for_n_authors_AV(n, val_size, test_size, m, seed=42):
    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n, val_size, test_size, seed)

    train_pos, train_neg = get_AV_dataset_from_AA_dataset(train_df, m)
    val_pos, val_neg = get_AV_dataset_from_AA_dataset(val_df, m)
    test_pos, test_neg = get_AV_dataset_from_AA_dataset(test_df, m)

    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg


def get_datasets_for_n_authors_AA(n, val_size, test_size, seed=42, path="./data/blog/"):
    # "n" authors with the highest number of samples
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


class AuthorsDatasetAV(Dataset):
    # [1, 0] or 1 means positive sample
    # [0, 1] or -1 means negative sample
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, positive_samples, negative_samples, tokenizer, max_source_len,
                 pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len

        self.pad_to_max_length = pad_to_max_length

        self.samples = [(p[0].strip(), p[1].strip(), 1) for p in positive_samples]
        self.samples.extend([(n[0].strip(), n[1].strip(), -1) for n in negative_samples])
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


# this allows us to do dynamic padding for batches. It significantly speeds up training time
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


if __name__ == "__main__":
    # train_df, test_df, val_df = get_datasets_for_n_authors_AA(n=5, val_size=0.1, test_size=0.2)
    df = pd.read_csv("test.csv")
    pos, neg = get_AV_dataset_from_AA_dataset(df, m=2)

    print("pos")
    print(len(pos))
    print("neg")
    print(len(neg))
