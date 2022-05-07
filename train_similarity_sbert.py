import os.path
import pickle
import random
import sys

import numpy as np

from blog_dataset import *
from model_params import *
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, InputExample, models
from sentence_transformers.losses.ContrastiveLoss import SiameseDistanceMetric
from sentence_transformers.evaluation import SentenceEvaluator
from sklearn.metrics.pairwise import cosine_distances
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


def tsne_plot(embeddings, labels, name, show=False):
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
    plt.savefig(f'tSNE_{name}.png')
    if show:
        plt.show()


def save_embeddings(model, train_samples, val_samples, test_samples, path):
    train_embeddings = model.encode(train_samples,
                                    convert_to_numpy=True,
                                    batch_size=32,
                                    show_progress_bar=True)
    val_embeddings = model.encode(train_samples,
                                  convert_to_numpy=True,
                                  batch_size=32,
                                  show_progress_bar=True)
    test_embeddings = model.encode(train_samples,
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


def test(params, training_samples, training_labels, val_samples, val_labels, batch_size, top_k):
    model = SentenceTransformer(params[CHECKPOINT])
    evaluator = ClassificationEvaluator(training_samples, training_labels, val_samples, val_labels, batch_size, [top_k],
                                        top_k)
    accuracy = evaluator(model, output_path="output/", save_results=True)
    print(f"Test Accuracy = {accuracy} with k = {top_k}")


def tune_k(params, training_samples, training_labels, val_samples, val_labels, batch_size, show=False):
    model = SentenceTransformer(params[CHECKPOINT])
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
                 save_results=False) -> float:
        print("** Validation **")
        print("obtaining training samples embeddings")
        train_embeddings = model.encode(self.train_samples,
                                        convert_to_numpy=True,
                                        batch_size=self.batch_size,
                                        show_progress_bar=True)
        print("obtaining validation samples embeddings")
        val_embeddings = model.encode(self.val_samples,
                                      convert_to_numpy=True,
                                      batch_size=self.batch_size,
                                      show_progress_bar=True)

        tsne_plot(val_embeddings, self.val_labels, f"authors{self.no_authors}_epoch{epoch}")

        cos_dists = []
        for i in range(len(val_embeddings)):
            val_embedding = [val_embeddings[i]]
            cos_dists.append(cosine_distances(val_embedding, train_embeddings))

        top_k_accuracies = []
        print("obtaining accuracies for all topks")
        for top_k in self.top_ks:
            predicted_val_labels = []

            for i in range(len(val_embeddings)):
                candidate_labels = []
                cos_dist = cos_dists[i]
                sorted_indicies = np.argsort(cos_dist)[0][:top_k]
                for topk_index in sorted_indicies:
                    candidate_label = self.train_labels[topk_index]
                    candidate_labels.append(candidate_label)

                # now we choose the label with the highest count
                voted_label = max(set(candidate_labels), key=candidate_labels.count)
                predicted_val_labels.append(voted_label)

            if save_results:
                predictions_df = pd.DataFrame({
                    "predicted_labels": predicted_val_labels,
                    "actual_labels": self.val_labels
                })
                predictions_df.to_csv(f"eval_authors{len(set(self.val_labels))}_topk{top_k}.csv")

            accuracy = metrics.accuracy_score(self.val_labels, predicted_val_labels)
            top_k_accuracies.append(accuracy)

        print(f"Accuracies for {self.top_ks} are {top_k_accuracies}")
        accuracy = top_k_accuracies[self.top_ks.index(self.top_k)]

        if output_list:
            return top_k_accuracies
        return accuracy


def train_AV_with_sbert_contrastive(params):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ######## defining the model
    word_embedding_model = models.Transformer(params[CHECKPOINT], max_seq_length=params[MAX_SOURCE_TEXT_LENGTH])
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    #########################
    # train, val and test splits
    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n=params[NO_AUTHORS],
                                                              val_size=0.1,
                                                              test_size=0.2,
                                                              seed=params[SEED])
    # train_df = train_df[:100]
    # val_df = val_df[:10]

    # positive label is 1, negative label is 0
    train_pos, train_neg = create_pos_neg_pairs(train_df)
    train_examples = [InputExample(texts=[train_pos[i][0], train_pos[i][1]], label=1) for i in range(len(train_pos))]
    train_examples.extend(
        [InputExample(texts=[train_neg[i][0], train_neg[i][1]], label=0) for i in range(len(train_neg))])

    train_dataset = SentencesDataset(train_examples, model)
    print(f"Training with {len(train_pos + train_neg)} samples")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params[TRAIN_BATCH_SIZE])

    # samples for validation
    train_samples = train_df["content"].tolist()
    train_labels = train_df["Target"].tolist()

    val_samples = val_df["content"].tolist()
    val_labels = val_df["Target"].tolist()

    evaluator = ClassificationEvaluator(train_samples, train_labels, val_samples, val_labels, params[VALID_BATCH_SIZE],
                                        [10], 10)

    # Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the two
    # embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    train_loss = losses.ContrastiveLoss(model=model, margin=0.5, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE)

    no_training_steps = params[TRAIN_EPOCHS] * (len(train_df) // params[TRAIN_BATCH_SIZE])
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


if __name__ == "__main__":
    seed_for_reproducability()
    params = model_paramsAV1

    model = train_AV_with_sbert_contrastive(params)

    # train, val and test splits
    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n=params[NO_AUTHORS],
                                                              val_size=0.1,
                                                              test_size=0.2,
                                                              seed=params[SEED])

    train_samples = train_df["content"].tolist()
    train_labels = train_df["Target"].tolist()

    val_samples = val_df["content"].tolist()
    val_labels = val_df["Target"].tolist()

    test_samples = test_df["content"].tolist()
    test_labels = test_df["Target"].tolist()

    params[CHECKPOINT] = "./output/checkpoints"
    best_k = tune_k(params, train_samples, train_labels, val_samples, val_labels, 32)

    test(params, train_samples, train_labels, test_samples, test_labels, 32, 10)
    test(params, train_samples, train_labels, test_samples, test_labels, 32, best_k)

    print("save embeddings")
    save_embeddings(model, train_samples, val_samples, test_samples, "./output")