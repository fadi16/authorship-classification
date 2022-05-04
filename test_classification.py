import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from blog_dataset import AuthorsDatasetAA, CollatorAA, get_datasets_for_n_authors_AA
from model_params import *
from utils import *


def test(params):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #test_df = pd.read_csv(f"./data/blog/test_{params_and_scores[NO_AUTHORS]}.csv")
    _, _, test_df = get_datasets_for_n_authors_AA(params[NO_AUTHORS], 0.1, 0.2, params[SEED])
    print(test_df[0])

    if params[MODEL] == AA:
        tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])
        config = AutoConfig.from_pretrained(params[CHECKPOINT], num_labels=params[NO_AUTHORS])
        model = AutoModelForSequenceClassification.from_pretrained(params[CHECKPOINT], config=config).to(device)

    test_dataset = AuthorsDatasetAA(test_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                    pad_to_max_length=False)

    collator = CollatorAA(pad_token_id=tokenizer.pad_token_id)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate_batch
    )

    model.eval()
    all_labels = []
    all_outputs_with_softmax = []
    val_losses = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for _, data in enumerate(test_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            output_logits = outputs.logits
            loss = loss_fn(output_logits, labels)
            val_losses.append(loss.item())

            all_labels.extend(labels.cpu().detach().numpy().tolist())
            all_outputs_with_softmax.extend(torch.softmax(output_logits, dim=1).cpu().detach().numpy().tolist())

    predicted_labels_indidices = [get_class_index_from_probs(output) for output in all_outputs_with_softmax]
    actual_labels_indicies = [get_class_index_from_probs(actual_label) for actual_label in all_labels]

    log_loss_softmax, accuracy_softmax, f1_score_micro_softmax, f1_score_macro_softmax = get_eval_scores(
        all_outputs_with_softmax, all_labels)

    print(f"Accuracy Score = {accuracy_softmax}")
    print(f"F1 Score (Micro) = {f1_score_micro_softmax}")
    print(f"F1 Score (Macro) = {f1_score_macro_softmax}")
    print(f"Multi Class Log Loss (softmax) = {log_loss_softmax}")

    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels_indidices,
        "actual_labels": actual_labels_indicies
    })
    predictions_df.to_csv(f"eval_{params[MODEL]}_{params[NO_AUTHORS]}.csv")


if __name__ == "__main__":
    test(model_params1)
