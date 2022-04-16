import re
from typing import List, Dict

import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollator
from constants import *


def preprocess_sents(sents):
    preprocessed_sents = []
    for sent in sents:
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        sent = re.sub(r"([?.!,¿;])", r" \1 ", sent)
        sent = re.sub(r'[" "]+', " ", sent)

        # true casing: first word lower case and keep the rest as they are
        sent_arr = sent.split()
        sent_arr[0] = sent_arr[0].lower()
        sent = " ".join(sent_arr)
        preprocessed_sents.append(sent)
    return preprocessed_sents


class AuthorsDataset(Dataset):
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, df, tokenizer, max_source_len, pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.source_text = preprocess_sents(df[TEXT_TAG_CSV])
        longest_source_sequence = len(max(self.source_text, key=lambda x: len(x.split())).split())
        print("longest_source_sequence = ", longest_source_sequence)
        self.target_classes = df[AUTHOR_TAG_CSV]
        self.author_index_to_no_samples = {AUTHOR_TO_INDEX[AUTHOR0]: 0, AUTHOR_TO_INDEX[AUTHOR1]: 0,
                                           AUTHOR_TO_INDEX[AUTHOR2]: 0}
        self.one_hot_target_classes = self.get_one_hot_target_classes()
        self.pad_to_max_length = pad_to_max_length
        assert len(self.source_text) == len(self.one_hot_target_classes)

    # todo: brackets? they can be used as features by the model, but they can also confuse it
    def get_one_hot_target_classes(self):
        one_hot_targets = []
        for target_class in self.target_classes:
            if target_class not in AUTHORS_TAGS:
                raise Exception("Unknown label")
            self.author_index_to_no_samples[AUTHOR_TO_INDEX[target_class]] += 1
            one_hot_target = [1 if i == AUTHOR_TO_INDEX[target_class] else 0 for i in range(len(AUTHORS_TAGS))]
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
    pass
