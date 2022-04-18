import random
from typing import List

import numpy as np
import torch
from sklearn import metrics


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_eval_scores(outputs, labels):
    pred_labels = [get_one_hot_class_from_probs(output) for output in outputs]
    # no. correctly classified / total number of samples
    accuracy = metrics.accuracy_score(labels, pred_labels)
    f1_score_micro = metrics.f1_score(labels, pred_labels, average='micro')
    f1_score_macro = metrics.f1_score(labels, pred_labels, average='macro')

    # this clips probabilities - like they do in the experiment (they even use the same parameter)
    log_loss = metrics.log_loss(labels, outputs)

    return log_loss, accuracy, f1_score_micro, f1_score_macro


def get_one_hot_class_from_probs(probs: List[float]):
    max_prob = max(probs)
    return [0 if p != max_prob else 1 for p in probs]


def get_class_index_from_probs(probs: List[float]):
    max_prob = max(probs)
    return probs.index(max_prob)


def rescale_probabilities(probs: List[float]):
    total = sum(probs)
    return [prob / total for prob in probs]


def seed_for_reproducability(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True