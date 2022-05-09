import os.path
import pickle
import random
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

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


def test_AV(model, threshold, test_pairs, test_labels):
    print("Testing Cross Encoder Accuracy on Authorship Validation")
    pred_scores = model.predict(test_pairs, convert_to_numpy=True, show_progress_bar=True)
    predicted_labels = [1 if s > threshold else 0 for s in pred_scores]
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels,
        "actual_labels": test_labels
    })
    predictions_df.to_csv(f"test_cross_encoder_AV_eval_authors{len(set(test_labels))}.csv")
    print(f"Test Accuracy = {accuracy}")


def test_classify_with_bi_encoder(cross_encoder_model, bi_encoder_model, train_samples, train_labels,
                                  test_samples, test_labels, top_k, batch_size, threshold=0.5):
    train_embeddings = bi_encoder_model.encode(train_samples,
                                               convert_to_numpy=True,
                                               batch_size=batch_size,
                                               show_progress_bar=True)

    test_embeddings = bi_encoder_model.encode(test_samples,
                                              convert_to_numpy=True,
                                              batch_size=batch_size,
                                              show_progress_bar=True)

    cos_dists = []
    for i in range(len(test_embeddings)):
        val_embedding = [test_embeddings[i]]
        cos_dists.append(cosine_distances(val_embedding, train_embeddings))

    predicted_labels = []

    for i in range(len(test_embeddings)):
        candidate_labels = []
        cos_dist = cos_dists[i]
        top_k_indicies = np.argsort(cos_dist)[0][:top_k]

        predictions = cross_encoder_model.predict(
            [[test_samples[i], train_samples[topk_index]] for topk_index in top_k_indicies],
            convert_to_numpy=True,
            show_progress_bar=True)
        for topk_index, prediction in zip(top_k_indicies, predictions):
            if prediction > threshold:
                candidate_label = train_labels[topk_index]
                candidate_labels.append(candidate_label)

        # now we choose the label with the highest count
        voted_label = max(set(candidate_labels), key=candidate_labels.count) if len(candidate_labels) != 0 else -1
        predicted_labels.append(voted_label)

    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels,
        "actual_labels": test_labels
    })
    predictions_df.to_csv(f"test_e2e_classification_authors{len(set(test_labels))}.csv")
    print(f"E2E bi-encoder + cross-encoder (k={top_k}) Test Accuracy = {accuracy}")


def train(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ######## defining the model
    model = CrossEncoder(params[CHECKPOINT], device=device, num_labels=1, max_length=params[MAX_SOURCE_TEXT_LENGTH])
    #########################
    # train, val and test splits
    train_df, val_df, test_df = get_datasets_for_n_authors(n=params[NO_AUTHORS],
                                                           val_size=0.1,
                                                           test_size=0.2,
                                                           seed=params[SEED])

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


def e2e_AV_test(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_for_reproducability()

    params[CHECKPOINT] = "./output/checkpoints/"

    model = CrossEncoder(params[CHECKPOINT], device=device, max_length=params[MAX_SOURCE_TEXT_LENGTH])

    test_pairs_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/test_pairs_{params[NO_AUTHORS]}_authors.csv")
    s1s = test_pairs_df["s1"].tolist()
    s2s = test_pairs_df["s2"].tolist()
    test_pairs = [list(pair) for pair in zip(s1s, s2s)]
    test_labels = test_pairs_df["label"].tolist()

    test_AV(model, 0.5, test_pairs, test_labels)


def e2e_classification_test(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_for_reproducability()

    bi_encoder = SentenceTransformer(f"./output/checkpoints/bi-encoder-{params[NO_AUTHORS]}", device=device)
    cross_encoder = CrossEncoder(f"./output/checkpoints/cross-encoder-{params[NO_AUTHORS]}", device=device, max_length=params[MAX_SOURCE_TEXT_LENGTH])

    train_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/train_{params[NO_AUTHORS]}_authors.csv")
    train_samples = train_df["content"].tolist()
    train_labels = train_df["Target"].tolist()

    test_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/test_{params[NO_AUTHORS]}_authors.csv")
    test_samples = test_df["content"].tolist()
    test_labels = test_df["Target"].tolist()

    test_classify_with_bi_encoder(cross_encoder_model=cross_encoder, bi_encoder_model=bi_encoder,
                                  train_samples=train_samples,
                                  train_labels=train_labels,
                                  test_samples=test_samples, test_labels=test_labels, top_k=10, batch_size=32,
                                  threshold=0.5)


if __name__ == "__main__":
    params = model_paramsAV1
    params[NO_AUTHORS] = 2
    e2e_classification_test(params)
