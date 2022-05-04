import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from blog_dataset import AuthorsDatasetAA, CollatorAA, get_datasets_for_n_authors_AA, get_datasets_for_n_authors_AV, \
    AuthorsDatasetAV, CollatorAV
from model_params import *
from utils import *
from model import *


def train_loop_AA(params):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for producing graphs with tensorboard
    tb = SummaryWriter()

    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n=params[NO_AUTHORS], val_size=0.1, test_size=0.2, )

    tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])
    config = AutoConfig.from_pretrained(params[CHECKPOINT], num_labels=params[NO_AUTHORS])
    model = AutoModelForSequenceClassification.from_pretrained(params[CHECKPOINT], config=config).to(device)
    optimizer = transformers.AdamW(params=model.parameters(), lr=params[LEARNING_RATE])
    scheduler = None
    if params[USE_SCHEDULER]:
        no_training_steps = params[TRAIN_EPOCHS] * (len(train_df) // params[TRAIN_BATCH_SIZE])
        no_warmup_steps = params[WARMUP_RATIO] * no_training_steps
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=no_warmup_steps,
                                                                 num_training_steps=no_training_steps)

    train_dataset = AuthorsDatasetAA(train_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                     pad_to_max_length=False)
    val_dataset = AuthorsDatasetAA(val_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                   pad_to_max_length=False)
    test_dataset = AuthorsDatasetAA(test_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                    pad_to_max_length=False)

    #####################################
    print("check if split is stratified")
    n = params[NO_AUTHORS]
    print(train_dataset.author_index_to_no_samples)
    sum_train_samples = sum(train_dataset.author_index_to_no_samples.values())
    train_samples = [train_dataset.author_index_to_no_samples[i] / sum_train_samples for i in range(n)]
    print(train_samples)

    print(val_dataset.author_index_to_no_samples)
    sum_val_samples = sum(val_dataset.author_index_to_no_samples.values())
    val_samples = [val_dataset.author_index_to_no_samples[i] / sum_val_samples for i in range(n)]
    print(val_samples)

    print(test_dataset.author_index_to_no_samples)
    sum_test_samples = sum(test_dataset.author_index_to_no_samples.values())
    test_samples = [test_dataset.author_index_to_no_samples[i] / sum_test_samples for i in range(n)]
    print(test_samples)
    ######################################

    if params[USE_CLASS_WEIGHTED_LOSS]:
        class_weights = get_ens_class_weights(params[BETA_FOR_WEIGHTED_CLASS_LOSS],
                                              train_dataset.author_index_to_no_samples)
        class_weights.to(device, dtype=torch.float)

    else:
        class_weights = None

    collator = CollatorAA(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate_batch
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate_batch
    )

    best_log_loss = 1000
    print("begin training")
    for epoch in range(params[TRAIN_EPOCHS]):
        print(f"Begin epoch {epoch}")
        train_step_AA(epoch, model, optimizer, scheduler, train_loader, class_weights, device, tb)
        log_loss = val_step_AA(epoch, model, val_loader, device, tb)

        if log_loss < best_log_loss:
            best_log_loss = log_loss

            # save the best model so far
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoints")
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print(f"SAVED MODEL AT from epoch {epoch} at " + model_checkpoint_path + "\n")

        print(f"Finished Epoch {epoch} log_loss = {log_loss}, best log_loss = {best_log_loss}")
        print("**" * 30)


def train_step_AA(epoch, model, optimizer, scheduler, training_loader, class_weights, device, tb):
    model.train()
    # use cross entropy loss (not binary cross entropy loss because that's for multi class multi label - out problem
    # is not multi label) we shouldn't apply a softmax on the output of the model because the CrossEntropyLoss
    # function internally does that
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights.to(device, dtype=float) if class_weights is not None else None)

    train_losses = []
    for _, data in enumerate(training_loader, 0):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['labels'].to(device, dtype=torch.float)

        outputs = model(ids, mask)
        output_logits = outputs.logits
        loss = loss_fn(output_logits, labels)

        train_losses.append(loss.item())
        # order from https://huggingface.co/docs/transformers/training
        loss.backward()
        optimizer.step()
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            tb.add_scalar("lr", current_lr, epoch * len(training_loader) + _)
            scheduler.step()
        optimizer.zero_grad()

    average_train_loss = np.mean(train_losses)
    tb.add_scalar("train_loss", average_train_loss, epoch)
    print(f"Average Train Loss = {average_train_loss}")


def val_step_AA(epoch, model, val_loader, device, tb):
    model.eval()
    all_labels = []
    all_outputs_with_softmax = []
    val_losses = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)
            outputs = model(ids, mask)
            output_logits = outputs.logits
            loss = loss_fn(output_logits, labels)
            val_losses.append(loss.item())

            all_labels.extend(labels.cpu().detach().numpy().tolist())
            all_outputs_with_softmax.extend(torch.softmax(output_logits, dim=1).cpu().detach().numpy().tolist())

    log_loss_softmax, accuracy_softmax, f1_score_micro_softmax, f1_score_macro_softmax = get_eval_scores(
        all_outputs_with_softmax, all_labels)

    average_val_loss = np.mean(val_losses)

    # results with sigmoid and softmax
    print(f"Average Validation Loss = {average_val_loss}")
    print(f"** Finished validating epoch {epoch} **")
    print(f"Accuracy Score = {accuracy_softmax}")
    print(f"F1 Score (Micro) = {f1_score_micro_softmax}")
    print(f"F1 Score (Macro) = {f1_score_macro_softmax}")
    print(f"Multi Class Log Loss (softmax) = {log_loss_softmax}")

    tb.add_scalar("val_loss", average_val_loss, epoch)
    tb.add_scalar("val_accuracy", accuracy_softmax, epoch)
    tb.add_scalar("val_f1_score_macro", f1_score_macro_softmax, epoch)
    tb.add_scalar("val_f1_score_micro", f1_score_micro_softmax, epoch)
    tb.add_scalar("val_log_loss_softmax", log_loss_softmax, epoch)

    return average_val_loss


def get_eval_scores(outputs, labels):
    pred_labels = [get_one_hot_class_from_probs(output) for output in outputs]
    # no. correctly classified / total number of samples
    accuracy = metrics.accuracy_score(labels, pred_labels)
    f1_score_micro = metrics.f1_score(labels, pred_labels, average='micro')
    f1_score_macro = metrics.f1_score(labels, pred_labels, average='macro')

    probs = [rescale_probabilities(output) for output in outputs]
    # this clips probabilities - like they do in the experiment (they even use the same parameter)
    log_loss = metrics.log_loss(labels, probs)

    return log_loss, accuracy, f1_score_micro, f1_score_macro


# weighting classes based on the effective number of samples - from paper Class-Balanced Loss Based on Effective
# Number of Samples https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class
# -Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
def get_ens_class_weights(beta, class_to_no_samples):
    weights = []
    for cls, no_samples in class_to_no_samples.items():
        # weight for class = 1 / effective no. samples = 1 / (1 - beta ** no_samples) / (1 - beta)
        w_cls = (1 - beta) / (1 - beta ** no_samples)
        weights.append(w_cls)

    # normalize to make the total loss roughly in the same scale when applying the weights
    sum_weights = sum(weights)
    weights = [len(class_to_no_samples) * w / sum_weights for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    train_loop_AV(model_paramsAV1)
