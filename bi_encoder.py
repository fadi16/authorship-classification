import os.path
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from blog_dataset import *
from model_params import *
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, InputExample, models
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.evaluation import SentenceEvaluator, BinaryClassificationEvaluator

from sklearn.metrics.pairwise import cosine_distances
from sklearn import metrics
import tqdm.std
from seed import seed_for_reproducability
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


######################### ABBREVIATIOS ######################################################################
# AV: means authorship verification - given 2 texts do they come from the same or from different authors
# AA/Classification: means authorship attribution - given a piece of text, predict its author
#############################################################################################################

# ####################### WHAT DOES THIS MODEL DO ? ########################################################
# This model is a BERT based bi-encoder/siamese network, using the sentence-transformers library.
# It predicts the author of a given piece of text It's an illustration of a similarity based approach for AA, in which, we have a database
# of known authors (i.e. training set) which we use to learn embeddings that capture the stylistic features the model
# needs/"thinks it needs" to differentiate between authors.
# At test time, we get the embedding of the given test sample, find its K most similar samples in the database
# (based on some similarity measure - we use cosine similarity here), and attribute it to the author with the highest
# number of samples among those K.
# NOTE: we do not leak training data at test time, the training data is our database, all test samples were never seen by the model.
# Why not just use the embeddings of a normal Bert (not a siamese one)?
# It has been shown by e.g., https://arxiv.org/pdf/1908.10084.pdf that raw embeddings from a raw Bert are not meaningful
# in the sense that they don't make similar setances close, and different ones far from each other. Hence, they can't be used
# for any similarity measure.
#
# This file contains the training and testing of the model.
# For testing here we only report a scores for a subset of the metrics used.
# Each test writes the results to a csv file, which will be then used to report performance against more metrics in evaluation.py


def tsne_plot(embeddings, labels, name, path="./output", show=False):
    """Creates TSNE plot of embeddings"""
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, n_jobs=10)
    x = embeddings
    y = np.array(labels)
    tsne_embeddings = tsne.fit_transform(x)
    tsne_embeddings = np.array(tsne_embeddings)

    plt.figure(figsize=(10, 10))
    for i in range(max(y) + 1):
        ys = np.where(y == i)
        plt.scatter(tsne_embeddings[ys, 0], tsne_embeddings[ys, 1], label=i)

    plt.legend()
    plt.title(f'Blogs-{name} t-SNE')
    plt.savefig(os.path.join(path, f'tSNE_{name}.png'))
    if show:
        plt.show()


def save_embeddings(model, train_samples, val_samples, test_samples, path):
    """Saves embeddings to avoid recalculation
    """
    train_embeddings = model.encode(train_samples,
                                    convert_to_numpy=True,
                                    batch_size=32,
                                    show_progress_bar=True)
    val_embeddings = model.encode(val_samples,
                                  convert_to_numpy=True,
                                  batch_size=32,
                                  show_progress_bar=True)
    test_embeddings = model.encode(test_samples,
                                   convert_to_numpy=True,
                                   batch_size=32,
                                   show_progress_bar=True)

    d = {
        "train": train_embeddings,
        "val": val_embeddings,
        "test": test_embeddings
    }

    f = open(os.path.join(path, "embeddings.pkl"), "wb")
    pickle.dump(d, f)
    f.close()


def test_classification(params, training_samples, training_labels, val_samples, val_labels, batch_size, top_k,
                        model=None, demo=False, saved_embeddings_path=""):
    """Tests bi-encoder on classification"""
    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(params[CHECKPOINT], device=device)
    evaluator = ClassificationEvaluator(training_samples, training_labels, val_samples, val_labels, batch_size, [top_k],
                                        top_k)
    accuracy, f1_micro, f1_macro, mcc= evaluator(model, output_path=os.path.join(params[OUTPUT_DIR],
                                                         f"cls_authors{len(set(val_labels))}_topk{top_k}.csv"),
                         save=True,
                         demo=demo,
                         saved_embeddings_path=saved_embeddings_path)
    print(f"Test Classification Accuracy = {accuracy} with k = {top_k}")
    return accuracy, f1_micro, f1_macro, mcc


def test_AV(params, threshold, val_pairs, val_labels, batch_size, model=None):
    """Tests bi-encoder on Author verification
    How well can it differentiate between texts written by same author vs different authors?
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = SentenceTransformer(params[CHECKPOINT], device=device)

    val_s1s = [pair[0] for pair in val_pairs]
    val_s2s = [pair[1] for pair in val_pairs]

    predicted_labels = []

    embeddings_val_s1s = model.encode(val_s1s,
                                      convert_to_numpy=True,
                                      batch_size=batch_size,
                                      show_progress_bar=True)
    embeddings_val_s2s = model.encode(val_s2s,
                                      convert_to_numpy=True,
                                      batch_size=batch_size,
                                      show_progress_bar=True)
    for i in range(len(val_s1s)):
        emb1 = embeddings_val_s1s[i]
        emb2 = embeddings_val_s2s[i]

        cos_sim = 1 - cosine_distances([emb1], [emb2])
        if cos_sim > threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels,
        "actual_labels": val_labels
    })
    predictions_df.to_csv(os.path.join(params[OUTPUT_DIR], f"AV_authors{len(set(val_labels))}.csv"))

    accuracy = metrics.accuracy_score(val_labels, predicted_labels)
    print(f"Accuracy - AV = {accuracy}")
    return accuracy


def tune_k(params, training_samples, training_labels, val_samples, val_labels, batch_size, show=False, model=None):
    """Finds the best value for K for KNN classification
    i.e. the one yielding the highest classification accuracy
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = SentenceTransformer(params[CHECKPOINT], device=device)
    candidate_ks = list(range(1, 200))
    evaluator = ClassificationEvaluator(training_samples, training_labels, val_samples, val_labels, batch_size,
                                        candidate_ks,
                                        10)
    accuracies = evaluator(model, output_path="output/", output_list=True)

    # plot how the accuracy changes as the chosen k increases
    plt.figure(figsize=(10, 10))
    plt.scatter(candidate_ks, accuracies)
    plt.title(f'Blogs-{params[NO_AUTHORS]} accuracy for different values of K')
    plt.savefig(f'k_vs_accuracy.png')
    if show:
        plt.show()

    best_accuracy = max(accuracies)
    best_k = accuracies.index(best_accuracy) + 1

    print(f"Best Accuracy = {best_accuracy}, with best_k = {best_k}")
    return best_k


class ClassificationEvaluator(SentenceEvaluator):
    """
    Tests how the model does on authorship attribution/classification.
    This is used to decide on which checkpoint to save while training the model
    Classification involves the following steps:
    1. obtain embeddings for all samples in the database (training samples). In real life that would be a FAISS index
    2. obtain embeddings for test/validation samples
    3. for a given test sample, measure similarity to every training sample (using cosine similarity here)
    4. rerank database based on similarity measure
    5. select top K most similar
    6. attribute test sample to the author who authors the highest number of the given test samples
    """

    def __init__(self, training_samples, training_labels, val_samples, val_labels, batch_size, top_ks, top_k):
        self.train_samples = training_samples
        self.train_labels = training_labels
        self.val_samples = val_samples
        self.val_labels = val_labels
        self.batch_size = batch_size
        self.top_ks = top_ks
        self.top_k = top_k
        self.no_authors = len(set(val_labels))

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, output_list=False,
                 save=False, demo=False, saved_embeddings_path="") -> float:
        print("** Validation **")
        if demo:
            # use saved embeddings for demo
            if (saved_embeddings_path == ""):
                print("If this is a demo, saved embeddings need to be supplied")
                return
            with open(saved_embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
                train_embeddings = embeddings["train"]
        else:
            # use model to obtain embeddings during training
            print("obtaining training samples embeddings")
            train_embeddings = model.encode(self.train_samples,
                                            convert_to_numpy=True,
                                            batch_size=self.batch_size,
                                            show_progress_bar=True)

            d = {
                "train": train_embeddings,
            }
            f = open("./data/train_embeddings.pkl", "wb")
            pickle.dump(d, f)
            f.close()
        print("obtaining validation samples embeddings")
        val_embeddings = model.encode(self.val_samples,
                                    convert_to_numpy=True,
                                    batch_size=self.batch_size,
                                    show_progress_bar=True)

        tsne_plot(val_embeddings, self.val_labels, f"authors{self.no_authors}_epoch{epoch}")

        # measure similarity, note that cosine distance = 1 - cosine similarity
        cos_dists = cosine_distances(val_embeddings, train_embeddings)
        # rerank
        sorted_indicies = np.argsort(cos_dists, axis=1)

        top_k_accuracies = []
        top_k_f1_micros =  []
        top_k_f1_macros = []
        top_k_mccs = []
        print("obtaining accuracies for all topks")
        for top_k in self.top_ks:
            predicted_val_labels = []
            sorted_indicies_topk = sorted_indicies[:, :top_k]
            for i in range(len(val_embeddings)):
                candidate_labels = [self.train_labels[topk_index] for topk_index in sorted_indicies_topk[i]]
                # now we choose the label with the highest count
                voted_label = np.bincount(candidate_labels).argmax()
                predicted_val_labels.append(voted_label)

            if save:
                predictions_df = pd.DataFrame({
                    "predicted_labels": predicted_val_labels,
                    "actual_labels": self.val_labels
                })
                predictions_df.to_csv(output_path)

            accuracy = metrics.accuracy_score(self.val_labels, predicted_val_labels)
            f1_micro = metrics.f1_score(self.val_labels, predicted_val_labels, average='micro')
            f1_macro = metrics.f1_score(self.val_labels, predicted_val_labels, average='macro')
            mcc = metrics.matthews_corrcoef(self.val_labels, predicted_val_labels)            
            top_k_accuracies.append(accuracy)
            top_k_f1_micros.append(f1_micro)
            top_k_f1_macros.append(f1_macro)
            top_k_mccs.append(mcc)

        print(f"Accuracies for {self.top_ks} are {top_k_accuracies}")
        accuracy = top_k_accuracies[self.top_ks.index(self.top_k)]
        f1_micro = top_k_f1_micros[self.top_ks.index(self.top_k)]
        f1_macro = top_k_f1_macros[self.top_ks.index(self.top_k)]
        mcc = top_k_mccs[self.top_ks.index(self.top_k)]

        if output_list:
            return top_k_accuracies
        return accuracy, f1_micro, f1_macro, mcc


def train_AV_with_sbert(params, train_samples, train_labels, val_samples, val_labels, train_pair_samples,
                        train_pair_labels, val_pair_samples, val_pair_labels):
    """
    Construct and train a Bert-based bi-encoder.
    The model tries to learn embeddings that are "close" for similar authors, and distant for different authors
    We experimented with multiple training objectives / loss functions, including:
    - contrastive loss: tries to reduce distance between similar samples, and increase distance between different samples.
    - online contrastive loss:  it selects hard positive (positives that are far apart) and hard negative pairs
        (negatives that are close) and computes the loss only for these pairs. The library reports that this often
        yields better results than contrastive losss, but that wasn't the case for us
    - batch hard triplet loss: takes a batch with (label, sentence) pairs and computes the loss for all possible,
        valid triplets, i.e., anchor and positive must have the same label, anchor and negative a different label.
        It then looks for the hardest positive and the hardest negatives.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # defining the model
    word_embedding_model = models.Transformer(params[CHECKPOINT], max_seq_length=params[MAX_SOURCE_TEXT_LENGTH])
    # use mean pooling, was reported to work better than cls pooling in https://arxiv.org/pdf/1908.10084.pdf
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)

    if params[LOSS] != BATCH_HARD_TRIPLET:  # contrastive losses need pairs
        # positive label is 1, negative label is 0
        train_examples = [InputExample(texts=train_pair_samples[i], label=train_pair_labels[i]) for i in
                          range(len(train_pair_samples))]

    else:
        # triplet loss needs just the samples, all possible triplets will be constructed from the batches automatically
        train_examples = [InputExample(texts=[train_samples[i]], label=train_labels[i]) for i in
                          range(len(train_samples))]

    train_dataset = SentencesDataset(train_examples, model)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params[TRAIN_BATCH_SIZE])
    print(f"Training with {len(train_dataset)} samples")

    evaluator = ClassificationEvaluator(train_samples, train_labels, val_samples, val_labels, params[VALID_BATCH_SIZE],
                                        [10], 10)

    if params[LOSS] == CONTRASTIVE:
        # Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the two
        # embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
        train_loss = losses.ContrastiveLoss(model=model, margin=0.5,
                                            distance_metric=SiameseDistanceMetric.COSINE_DISTANCE)
    elif params[LOSS] == ONLINE_CONTRASTIVE:
        train_loss = losses.OnlineContrastiveLoss(model=model, margin=0.5,
                                                  distance_metric=SiameseDistanceMetric.COSINE_DISTANCE)
    elif params[LOSS] == BATCH_HARD_TRIPLET:
        train_loss = losses.BatchHardTripletLoss(model=model,
                                                 distance_metric=losses.BatchHardTripletLossDistanceFunction.cosine_distance)

    no_training_steps = params[TRAIN_EPOCHS] * (len(train_dataset) // params[TRAIN_BATCH_SIZE])
    no_warmup_steps = params[WARMUP_RATIO] * no_training_steps

    def pprint(score, epoch, steps):
        print(f"\nEpoch {epoch} - Accuracy with topK = {score}\n")

    ######## Train the model
    model.fit(train_objectives=[(train_loader, train_loss)],
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


def tune_AV_threshold(params, val_pairs, val_labels, model=None):
    """
    :param params: the model parameters, e.g. NO_AUTHORS, CHECKPOINT, etc
    :param val_pairs: pairs of positive and negative samples from the validation set
    :param val_labels: positive 1 or negative 0
    :param model: optional, provide the mode
    :return:
    """
    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = SentenceTransformer(params[CHECKPOINT], device=device)

    # first elements in pairs
    s1s_val = [s1 for s1, _ in val_pairs]
    # second elements in pairs
    s2s_val = [s2 for _, s2 in val_pairs]

    evaluator = BinaryClassificationEvaluator(sentences1=s1s_val, sentences2=s2s_val, labels=val_labels,
                                              batch_size=32, show_progress_bar=True)
    # the evaluator will produce a csv file with the optimal accuracy, and its corresponding threshold
    evaluator(model, os.path.join(params[OUTPUT_DIR]))


def e2e_experiment(params, train, test, tune):
    # obtain training/testing texts and labels
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    # obtain testing pairs
    test_pairs, test_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "test")

    try:
        val_samples, val_labels = get_samples_and_labels(params[NO_AUTHORS], "val")
        train_pairs, train_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
        val_pairs, val_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "val")
    except:
        print("some files were not found, this is not expected unless you're testing for unknown authors"
              "15 authors for the 10 authors checkpoint, or 75 for the 50 authors checkpoint)")

    model = None
    if train:
        # train the model
        model = train_AV_with_sbert(params, train_samples, train_labels, val_samples, val_labels, train_pairs,
                                    train_pairs_labels, val_pairs, val_pairs_labels)

    if tune:
        # tune the model - using validation set
        # the checkpoint will be here after training
        params[CHECKPOINT] = "./output/checkpoints/"
        # find best K for K-NN
        tune_k(params, train_samples, train_labels, val_samples, val_labels, batch_size=32, show=True,
               model=model)
        # find best threshold for AV
        tune_AV_threshold(params, val_pairs, val_labels, model=None)

    if test:
        # test the model for AV
        acc_av = test_AV(params, params[THRESHOLD], test_pairs, test_pairs_labels, batch_size=32, model=model)
        # test authorship classification using 10-NN
        acc_classification_k10, f1_micro_k10, f1_macro_k10, mcc_k10 = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                     batch_size=32, top_k=10, model=None)
        # test authorship classification using BEST_K-NN
        acc_classification_best_k, f1_micro_best_k, f1_macro_best_k, mcc_best_k = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                      batch_size=32, top_k=params[BEST_K], model=None)
        # reuse embedding next time
        save_embeddings(model, train_samples, val_samples, test_samples, path=params[OUTPUT_DIR])

        save_embeddings(model, train_samples, val_samples, test_samples, path=params[OUTPUT_DIR])
        stats = {
            "Classification Accuracy k = 10": acc_classification_k10,
            f"Classification Accuracy k = {params[BEST_K]}": acc_classification_best_k,
            "AV Accuracy": acc_av,
        }
        print(stats)


# demo a model trained on 10 authors using a (reduced) test set containing the same 10 authors the model was exposed to
def demo_tr_10_tst_10():
    seed_for_reproducability()
    params = bi_encoder_params_batch_hard_triplet_10
    params[NO_AUTHORS]=10
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    saved_embeddings_path= get_demo_embeddings_path("bi_encoder", params[NO_AUTHORS])
    acc_classification_k10, f1_micro, f1_macro, mcc = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                 batch_size=32, top_k=10, model=None, demo=True, saved_embeddings_path=saved_embeddings_path)
    stats = {
        "Classification Accuracy for 10 authors k = 10": acc_classification_k10,
        "f1 micro": f1_micro,
        "f1 macro": f1_macro,
        "mcc": mcc
    }
    print(stats)

# demo a model trained on 10 authors using a (reduced) test set containing the same 10 authors the model was exposed to
# IN ADDITION TO 5 authors that the model never saw before
def demo_tr_10_tst_15():
    seed_for_reproducability()
    params = bi_encoder_params_batch_hard_triplet_10
    params[NO_AUTHORS] = 15
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    saved_embeddings_path = get_demo_embeddings_path(params[NO_AUTHORS])
    acc_classification_k10, f1_micro, f1_macro, mcc = test_classification(params, train_samples, train_labels,
                                                                          test_samples, test_labels,
                                                                          batch_size=32, top_k=10, model=None,
                                                                          demo=True,
                                                                          saved_embeddings_path=saved_embeddings_path)

    stats = {
        "Classification Accuracy for 15 authors k = 10": acc_classification_k10,
        "f1 micro": f1_micro,
        "f1 macro": f1_macro,
        "mcc": mcc
    }
    print(stats)


if __name__ == "__main__":
    seed_for_reproducability()
    params = bi_encoder_params_batch_hard_triplet_10
    # e2e_experiment(params, train=False, test=True, tune=False)

    # Uncomment below to run demos
    demo_tr_10_tst_10()
    demo_tr_10_tst_15()
