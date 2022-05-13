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
from utils import seed_for_reproducability
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
    """Tests bi-encoder on Author verification"""
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
    """Tunes k to find best value
    the top k documents and perform classification"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if model is None:
        model = SentenceTransformer(params[CHECKPOINT], device=device)
    candidate_ks = list(range(1, 200))
    evaluator = ClassificationEvaluator(training_samples, training_labels, val_samples, val_labels, batch_size,
                                        candidate_ks,
                                        10)
    accuracies = evaluator(model, output_path="output/", output_list=True)

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
                 save=False, demo=False,saved_embeddings_path="") -> float:
        print("** Validation **")
        if (demo):
            if (saved_embeddings_path==""):
                print("If this is a demo, saved embeddings need to be supplied")
                return
            with open(saved_embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
                train_embeddings = embeddings["train"]
        else:
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

        cos_dists = cosine_distances(val_embeddings, train_embeddings)
        sorted_indicies = np.argsort(cos_dists, axis=1)

        top_k_accuracies = []
        top_k_f1_micros =  []
        top_k_f1_macros = []
        top_k_mccs = []
        print("obtaining accuracies for all topks")
        for top_k in self.top_ks:
            predicted_val_labels = []
            sorted_indicies_topk = sorted_indicies[:,:top_k]
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ######## defining the model
    word_embedding_model = models.Transformer(params[CHECKPOINT], max_seq_length=params[MAX_SOURCE_TEXT_LENGTH])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    #########################

    if params[LOSS] != BATCH_HARD_TRIPLET:  # contrastive losses need pairs
        # positive label is 1, negative label is 0
        train_examples = [InputExample(texts=train_pair_samples[i], label=train_pair_labels[i]) for i in
                          range(len(train_pair_samples))]

    else:  # triplet loss needs just the samples
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model is None:
        model = SentenceTransformer(params[CHECKPOINT], device=device)

    s1s_val = [s1 for s1, _ in val_pairs]
    s2s_val = [s2 for _, s2 in val_pairs]

    evaluator = BinaryClassificationEvaluator(sentences1=s1s_val, sentences2=s2s_val, labels=val_labels,
                                              batch_size=32, show_progress_bar=True)

    ap = evaluator(model, os.path.join(params[OUTPUT_DIR]))



def e2e_experiment(params, train, test, tune):
    seed_for_reproducability()

    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    test_pairs, test_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "test")

    try:
        val_samples, val_labels = get_samples_and_labels(params[NO_AUTHORS], "val")
        train_pairs, train_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
        val_pairs, val_pairs_labels = get_pairs_and_labels(params[NO_AUTHORS], "val")
    except:
      print("some files were not found, this is not expected unless you're testing for unknown authors")

    model = None
    if train:
        model = train_AV_with_sbert(params, train_samples, train_labels, val_samples, val_labels, train_pairs,
                                    train_pairs_labels, val_pairs, val_pairs_labels)

    if tune:
        # the checkpoint will be here after training
        params[CHECKPOINT] = "./output/checkpoints/bi-encoder-10/"
        best_k = tune_k(params, train_samples, train_labels, val_samples, val_labels, batch_size=32, show=True,
                        model=model)

        # tune_AV_threshold(params, val_pairs, val_labels, model=None)

    if test:
        # acc50_av = test_AV(params, params[THRESHOLD], test_pairs, test_pairs_labels, batch_size=32, model=model)
        # acc75_av = test_AV(params, params[THRESHOLD], test_pairs, test_pairs_labels, batch_size=32, model=model)

        acc_classification_k10, f1_micro, f1_macro, mcc = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                     batch_size=32, top_k=10, model=None)
        # acc_classification_topk = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                    #   batch_size=32, top_k=params[BEST_K], model=None)
        # save_embeddings(model, train_samples, val_samples, test_samples, path=params[OUTPUT_DIR])
        stats = {
            "Classification Accuracy k = 10": acc_classification_k10
            # "AV Accuracy": acc50_av,
            # "AV Accuracy": acc75_av
            # f"Classification Accuracy k = {params[BEST_K]}": acc_classification_topk,
        }
        print(stats)

    # look at
    # saved embeddings
    # saved classification results
    # saved AV results
    # saved embeddings plot

def demo_tr_10_tst_10():
    seed_for_reproducability()
    params = bi_encoder_params_batch_hard_triplet_10
    params[NO_AUTHORS]=10

    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)



    model = None
    saved_embeddings_path= get_demo_embeddings_path("bi_encoder", params[NO_AUTHORS])
    acc_classification_k10, f1_micro, f1_macro, mcc = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                 batch_size=32, top_k=10, model=None, demo=True, saved_embeddings_path=saved_embeddings_path)
    # acc_classification_topk = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                #   batch_size=32, top_k=params[BEST_K], model=None)
    # save_embeddings(model, train_samples, val_samples, test_samples, path=params[OUTPUT_DIR])
    stats = {
        "Classification Accuracy for 15 authors k = 10": acc_classification_k10,
        "f1 micro": f1_micro,
        "f1 macro": f1_macro,
        "mcc": mcc
    }
    print(stats)

def demo_tr_10_tst_15():
    seed_for_reproducability()
    params = bi_encoder_params_batch_hard_triplet_10
    params[NO_AUTHORS]=15

    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", balanced=params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)


    model = None
    saved_embeddings_path= get_demo_embeddings_path("bi_encoder", params[NO_AUTHORS])
    acc_classification_k10, f1_micro, f1_macro, mcc = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                 batch_size=32, top_k=10, model=None, demo=True, saved_embeddings_path=saved_embeddings_path)
    # acc_classification_topk = test_classification(params, train_samples, train_labels, test_samples, test_labels,
                                                #   batch_size=32, top_k=params[BEST_K], model=None)
    # save_embeddings(model, train_samples, val_samples, test_samples, path=params[OUTPUT_DIR])
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