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

from seed import seed_for_reproducability

##############################################################################
# This module contains utilities for constructing the datasets from the original blogtexts.csv file based on the
# number of authors required, the training objective etc.
# We follow all Authorship Identification previous works in not applying any preprocessing/cleaning for the data
# as that would eliminate stylistic features needed for solving the problem

def get_demo_embeddings_path(model, no_authors):
    file_path =f"data/blog/{no_authors}_authors/demo_{model}_{no_authors}_authors_embeddings.pkl" 
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


def balance_fn(texts, labels):
    """
    Randomly over/under-sample texts from different authors to make sure that the number of texts for each author is equal to
    the mean + std number of texts for each author
    """
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


def create_pos_neg_pairs(df, balance=False):
    """
    split dataset into positive pairs (coming from same author) and negative pairs (coming from different authors)
    the approach is similar to that described in page 9 of this paper https://arxiv.org/pdf/1912.10616.pdf
    We do the following (for each author):
    - split the texts produced by each author into 4 equal sized chunks
    - merge the first two chunks to create positive/same-author pairs for the given author
    - split the third chunk into N - 1 pieces, where N is the total number of authors in the dataset
    - merge the pieces from the third chunk with random texts from the other (N-1) authors' forth chunks
    This results in an equal number of positive and negative pairs in these datasets.
    :param df:
    :param balance: whether to balance the dataset using balance_fn before constructing the pairs
    """

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


# helper to get pairs of texts written by same/different authors
def get_pairs_and_labels(no_authors, split, balanced=False):
    pairs_df = pd.read_csv(
        f"./data/blog/{no_authors}_authors/{split}_pairs_{no_authors}_authors{'_balanced' if balanced else ''}.csv")
    s1s = pairs_df["s1"].tolist()
    s2s = pairs_df["s2"].tolist()
    pairs = [list(pair) for pair in zip(s1s, s2s)]
    labels = pairs_df["label"].tolist()
    return pairs, labels


# helper
def get_samples_and_labels(no_authors, split, balanced=False, demo=False):
    df = pd.read_csv(
        f"./data/blog/{no_authors}_authors/{'demo_' if demo else ''}{split}_{no_authors}_authors{'_balanced' if balanced else ''}.csv")
    samples = df["content"].tolist()
    labels = df["Target"].tolist()
    return samples, labels


def create_all_datasets(no_authors, balance=False):
    """
    Creates all the datasets needed:
   - train, test, val datasets of texts to labels indicating the class/author.
   - train, test, val datasets of pairs of text with label 1 if pair comes from same author, 0 if they come from different authors

   The split is 70% for train, 10% for validation and 20% for test

    :param no_authors: the number of authors to include in the dataset, select all texts from the no_authors with the highest number of texts
    :param balance: if true, it over/under-samples texts from different authors to make sure that the number of texts for each author is equal to
                    the mean + std number of texts for each author. Note that balacing is only applied to the training set and not
                    to the testing/validation sets
    """
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
