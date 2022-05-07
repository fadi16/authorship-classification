import os.path
import pickle
import random
import sys

import numpy as np

from blog_dataset import *
from model_params import *
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, InputExample, models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn import metrics
import tqdm.std
from utils import seed_for_reproducability
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def create_pos_neg_pairs(df):
    # split as described in page ten of this paper https://arxiv.org/pdf/1912.10616.pdf
    pos_pairs = []
    neg_pairs = []

    authors_indicies = sorted(set(df["Target"].tolist()))
    no_authors = len(authors_indicies)

    # list of contents for each author, for author i their content is at the ith index
    authors_chunks = []
    for i in authors_indicies:
        current_author_texts = [text.strip() for text in df.loc[df["Target"] == i]["content"].tolist()]

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


def map_chunks(texts1, texts2):
    # map texts between two authors to create negative pairs
    # the number of texts might not be similar
    random.shuffle(texts1)
    random.shuffle(texts2)

    return list(zip(texts1[:len(texts2)], texts2)) if len(texts2) < len(texts1) else list(
        zip(texts1, texts2[:len(texts1)]))


def train(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # todo
    params[NO_AUTHORS] = 2

    ######## defining the model
    model = CrossEncoder(params[CHECKPOINT], device=device, num_labels=1)
    #########################
    # train, val and test splits
    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n=params[NO_AUTHORS],
                                                              val_size=0.1,
                                                              test_size=0.2,
                                                              seed=params[SEED])
    train_df = train_df[:20]
    val_df = val_df[:10]

    # positive label is 1, negative label is 0
    train_pos, train_neg = create_pos_neg_pairs(train_df)
    train_examples = [InputExample(texts=[train_pos[i][0], train_pos[i][1]], label=1) for i in range(len(train_pos))]
    train_examples.extend(
        [InputExample(texts=[train_neg[i][0], train_neg[i][1]], label=0) for i in range(len(train_neg))])

    train_dataset = SentencesDataset(train_examples, model)
    print(f"Training with {len(train_pos + train_neg)} samples")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params[TRAIN_BATCH_SIZE])

    val_pos, val_neg = create_pos_neg_pairs(val_df)
    val_samples = [[val_pos[i][0], val_pos[i][1]] for i in range(len(val_pos))]
    val_labels = [1 for _ in range(len(val_pos))]
    val_samples.extend([[val_neg[i][0], val_neg[i][1]] for i in range(len(val_neg))])
    val_labels.extend([0 for _ in range(len(val_neg))])

    no_training_steps = params[TRAIN_EPOCHS] * (len(train_df) // params[TRAIN_BATCH_SIZE])
    no_warmup_steps = params[WARMUP_RATIO] * no_training_steps

    def pprint(score, epoch, steps):
        print(f"\nEpoch {epoch} - Score = {score}\n")

    evaluator = CEBinaryClassificationEvaluator(sentence_pairs=val_samples, labels=val_labels)
    ######## Train the model
    model.fit(train_dataloader=train_loader,
              evaluator=evaluator,
              epochs=params[TRAIN_EPOCHS],
              evaluation_steps=-1,  # evaluate after every epoch
              save_best_model=True,
              warmup_steps=no_warmup_steps,
              output_path="output/checkpoints",
              use_amp=True,
              show_progress_bar=True,
              callback=pprint,
              scheduler="warmupconstant")

    return model


if __name__ == "__main__":
    seed_for_reproducability()
    params = model_paramsAV1

    model = train(params)



