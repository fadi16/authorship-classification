import re
from typing import List, Dict

import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollator
from constants import *
from sklearn.model_selection import train_test_split


def get_datasets_for_n_authors(n, val_size, test_size, seed=42):
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
                                              test_size=(1 / (1 - test_size)) * val_size, stratify=train_val_df["Target"])
    train_inds = list(train_and_val_indicies[0].index)
    train_df = train_val_df.loc[train_inds]

    val_inds = list(train_and_val_indicies[1].index)
    val_df = train_val_df.loc[val_inds]

    test_inds = list(train_val_and_test_indicies[1].index)
    test_df = new_df.loc[test_inds]

    return train_df, val_df, test_df


class AuthorsDataset(Dataset):
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, df, source_tag, target_tag, tokenizer, max_source_len, pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.source_tag = source_tag
        self.source_text = df[source_tag].tolist()
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

    # todo: brackets? they can be used as features by the model, but they can also confuse it
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
class Collator:
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


if __name__ == "__main__":
    train_df, test_df, val_df = get_datasets_for_n_authors(n=5, val_size=0.1, test_size=0.2)
