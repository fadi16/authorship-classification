import os.path
import pickle
import random
import sys

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sympy import true
from bi_encoder import save_embeddings

from blog_dataset import *
from model_params import *
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, LoggingHandler, losses, InputExample, models
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sklearn import metrics
from utils import seed_for_reproducability



def test_AV(model, threshold, test_pairs, test_labels):
    """ Tests model on author verification
    """
    print("Testing Cross Encoder Accuracy on Authorship Validation")
    pred_scores = model.predict(test_pairs, convert_to_numpy=True, show_progress_bar=True)
    predicted_labels = [1 if s > threshold else 0 for s in pred_scores]
    accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels,
        "actual_labels": test_labels
    })
    predictions_df.to_csv(f"test_AV_authors.csv")
    print(f"Test Accuracy = {accuracy}")


def test_classify_with_bi_encoder(cross_encoder_model, bi_encoder_model, train_samples, train_labels,
                                  test_samples, test_labels, top_k, batch_size, threshold=0.5, demo = False, save = False, saved_embeddings_path=""):
    """Tests using combination of cross and bi-encoder model on author classification
    """
    if (demo):
        if (saved_embeddings_path==""):
            print("If this is a demo, saved embeddings need to be supplied")
            return
        with open(saved_embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
            train_embeddings = embeddings["train"]
            test_embeddings = embeddings["test"]
    else:
        train_embeddings = bi_encoder_model.encode(train_samples,
                                                convert_to_numpy=True,
                                                batch_size=batch_size,
                                                show_progress_bar=True)
        
        test_embeddings = bi_encoder_model.encode(test_samples,
                                                convert_to_numpy=True,
                                                batch_size=batch_size,
                                                show_progress_bar=True)
        if (save):
            d = {
                "train": train_embeddings,
                "test": test_embeddings
            }
            f = open("./data/test_embeddings.pkl", "wb")
            pickle.dump(d, f)
            f.close()
    

    cos_dists = []
    for i in range(len(test_embeddings)):
        val_embedding = [test_embeddings[i]]
        cos_dists.append(cosine_distances(val_embedding, train_embeddings))

    predicted_labels = []

    for i in range(len(test_embeddings)):
        candidate_labels = []
        cos_dist = cos_dists[i]
        top_k_indicies = np.argsort(cos_dist)[0][:top_k]

        pairs =  [[test_samples[i], train_samples[topk_index]] for topk_index in top_k_indicies]

        predictions = cross_encoder_model.predict(
           pairs,
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
    predictions_df.to_csv(f"test_e2e_classification_authors{len(set(test_labels))}_samples{len(test_samples)}.csv")
    print(f"E2E bi-encoder + cross-encoder (k={top_k}) Test Accuracy = {accuracy}")


def train(params):
    """Trains model on given params
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # defining the model
    model = CrossEncoder(params[CHECKPOINT], device=device, num_labels=1, max_length=params[MAX_SOURCE_TEXT_LENGTH])

    # positive label is 1, negative label is 0
    train_pairs, train_labels = get_pairs_and_labels(params[NO_AUTHORS], "train", params[BALANCE])
    train_examples = [InputExample(texts=train_pairs[i], label=train_labels[i]) for i in range(len(train_labels))]

    train_dataset = SentencesDataset(train_examples, model)
    print(f"Training with {len(train_dataset)} samples")

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=params[TRAIN_BATCH_SIZE])

    val_pairs, val_labels = get_pairs_and_labels(params[NO_AUTHORS], "val")

    no_training_steps = params[TRAIN_EPOCHS] * (len(train_dataset) // params[TRAIN_BATCH_SIZE])
    no_warmup_steps = params[WARMUP_RATIO] * no_training_steps

    def pprint(score, epoch, steps):
        print(f"\nEpoch {epoch} - Score = {score}\n")

    evaluator = CEBinaryClassificationEvaluator(sentence_pairs=val_pairs, labels=val_labels)

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
    """End to End author verification test
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_for_reproducability()
    params[CHECKPOINT] = "./output/checkpoints/"

    model = CrossEncoder(params[CHECKPOINT], device=device, max_length=params[MAX_SOURCE_TEXT_LENGTH])
    test_pairs, test_labels = get_pairs_and_labels(params[NO_AUTHORS], "test")
    test_AV(model, params[THRESHOLD], test_pairs, test_labels)


def e2e_classification_test(params):
    """End to End classification test
    uses both cross and bi encoder to test on author classification"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_for_reproducability()

    bi_encoder = SentenceTransformer(f"./output/checkpoints/bi-encoder-{params[NO_AUTHORS]}", device=device)
    cross_encoder = CrossEncoder(f"./output/checkpoints/cross-encoder-{params[NO_AUTHORS]}", device=device,
                                 max_length=params[MAX_SOURCE_TEXT_LENGTH])
    # params[NO_AUTHORS] = 15
    
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test")
    
    test_classify_with_bi_encoder(cross_encoder_model=cross_encoder, bi_encoder_model=bi_encoder,
                                  train_samples=train_samples,
                                  train_labels=train_labels,
                                  test_samples=test_samples, test_labels=test_labels, top_k=161, batch_size=32,
                                  threshold=params[THRESHOLD],
                                  save=False)

def demo_tr_10_tst_10_with_bi_encoder():
    """Small test to demonstrate performance during live demo
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_for_reproducability()

    params = cross_encoder_params_10
    params[NO_AUTHORS] = 10

    bi_encoder = SentenceTransformer(f"./output/checkpoints/bi-encoder-{params[NO_AUTHORS]}", device=device)
    cross_encoder = CrossEncoder(f"./output/checkpoints/cross-encoder-{params[NO_AUTHORS]}", device=device,
                                 max_length=params[MAX_SOURCE_TEXT_LENGTH])
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    saved_embeddings_path= get_demo_embeddings_path(params[NO_AUTHORS])
    test_classify_with_bi_encoder(cross_encoder_model=cross_encoder, bi_encoder_model=bi_encoder,
                                  train_samples=train_samples,
                                  train_labels=train_labels,
                                  test_samples=test_samples, test_labels=test_labels, top_k=10, batch_size=32,
                                  threshold=params[THRESHOLD], 
                                  demo=True,
                                  saved_embeddings_path=saved_embeddings_path)

def demo_tr_10_tst_15_with_bi_encoder():
    """Small test to demonstrate performance during live demo
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_for_reproducability()

    params = cross_encoder_params_10

    bi_encoder = SentenceTransformer(f"./output/checkpoints/bi-encoder-{params[NO_AUTHORS]}", device=device)
    cross_encoder = CrossEncoder(f"./output/checkpoints/cross-encoder-{params[NO_AUTHORS]}", device=device,
                                 max_length=params[MAX_SOURCE_TEXT_LENGTH])
    params[NO_AUTHORS] = 15
    train_samples, train_labels = get_samples_and_labels(params[NO_AUTHORS], "train", params[BALANCE])
    test_samples, test_labels = get_samples_and_labels(params[NO_AUTHORS], "test", demo=True)

    saved_embeddings_path= get_demo_embeddings_path(params[NO_AUTHORS])

    test_classify_with_bi_encoder(cross_encoder_model=cross_encoder, bi_encoder_model=bi_encoder,
                                  train_samples=train_samples,
                                  train_labels=train_labels,
                                  test_samples=test_samples, test_labels=test_labels, top_k=10, batch_size=32,
                                  threshold=params[THRESHOLD], 
                                  demo=True,
                                  saved_embeddings_path=saved_embeddings_path)


if __name__ == "__main__":
    params = cross_encoder_params_10
    #train(params)
    #e2e_AV_test(params)
    # e2e_classification_test(params)
    # Uncomment below to run demos
    demo_tr_10_tst_10_with_bi_encoder()
    demo_tr_10_tst_15_with_bi_encoder()