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
from blog_dataset import *
from model_params import *
from utils import *
from model import *


def train_loop_AV(params):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for producing graphs with tensorboard
    tb = SummaryWriter()

    # train, val and test splits
    train_df, val_df, test_df = get_datasets_for_n_authors_AA(n=params[NO_AUTHORS],
                                                              val_size=0.1,
                                                              test_size=0.2,
                                                              seed=params[SEED])

    train_pos, train_neg = get_AV_dataset_from_AA_dataset(train_df, params[MULTIPLIER])
    val_pos, val_neg = get_AV_dataset_from_AA_dataset(val_df, params[MULTIPLIER])
    test_pos, test_neg = get_AV_dataset_from_AA_dataset(test_df, params[MULTIPLIER])

    print(f"train pos / neg = {len(train_pos) / len(train_neg)}")
    print(f"val pos / neg = {len(val_pos) / len(val_neg)}")
    print(f"test pos / neg = {len(test_pos) / len(test_neg)}")

    ########################
    tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])

    # train val and test dataset construction
    train_dataset = AuthorsDatasetAV(train_pos, train_neg, tokenizer, params[MAX_SOURCE_TEXT_LENGTH], False)
    train_dataset_original = AuthorsDatasetAA(train_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                              pad_to_max_length=False)

    val_dataset = AuthorsDatasetAV(val_pos, val_neg, tokenizer, params[MAX_SOURCE_TEXT_LENGTH], False)
    val_dataset_original = AuthorsDatasetAA(val_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                            pad_to_max_length=False)
    test_dataset = AuthorsDatasetAV(test_pos, test_neg, tokenizer, params[MAX_SOURCE_TEXT_LENGTH], False)

    ########################

    model = BertSiam(
        dropout=params[DROUPOUT],
        checkpoint=params[CHECKPOINT],
        pooling_method=params[POOLING],
    ).to(device)

    optimizer = transformers.AdamW(params=model.parameters(), lr=params[LEARNING_RATE])
    scheduler = None
    if params[USE_SCHEDULER]:
        no_training_steps = params[TRAIN_EPOCHS] * ((len(train_pos) + len(train_neg)) // params[TRAIN_BATCH_SIZE])
        no_warmup_steps = params[WARMUP_RATIO] * no_training_steps
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=no_warmup_steps,
                                                                 num_training_steps=no_training_steps)

    collator = CollatorAV(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate_batch
    )

    train_loader = DataLoader(
        dataset=train_dataset_original,
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

    val_loader_original = DataLoader(
        dataset=val_dataset_original,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator.collate_batch
    )

    best_loss = 1000
    print("begin training")

    bert_frozen = False
    if params[FREEZE_NO_EPOCHS] != 0:
        model.freeze_subnetworks()
        print("Froze BERT weights")

    for epoch in range(params[TRAIN_EPOCHS]):

        print(f"Begin epoch {epoch}")

        if bert_frozen and epoch > params[FREEZE_NO_EPOCHS]:
            model.unfreeze_subnetworks()
            bert_frozen = False
            print("Unfroze BERT weights")

        train_step_AV(epoch, model, optimizer, scheduler, train_loader, params, None, device, tb)
        val_loss = val_step_AV(epoch, model, val_loader, params, device, tb)

        if val_loss < best_loss:
            best_loss = val_loss

            # save the best model so far
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoints")
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print(f"SAVED MODEL AT from epoch {epoch} at " + model_checkpoint_path + "\n")

        print(f"Finished Epoch {epoch} log_loss = {val_loss}, best log_loss = {best_loss}")
        print("**" * 30)


def train_step_AV(epoch, model, optimizer, scheduler, training_loader, params, class_weights, device, tb):
    model.train()

    loss_fn = torch.nn.MSELoss()

    train_losses = []
    for _, data in enumerate(training_loader, 0):

        # embedding for first sentence
        ids1 = data['input_ids'][0].to(device, dtype=torch.long)
        mask1 = data['attention_mask'][0].to(device, dtype=torch.long)
        ids2 = data['input_ids'][1].to(device, dtype=torch.long)
        mask2 = data['attention_mask'][1].to(device, dtype=torch.long)

        logits = model.forward(ids1, mask1, ids2, mask2)

        # 1 if they come from the same author, -1 otherrwise
        labels = data['labels'].to(device, dtype=torch.float)

        loss = loss_fn(logits, labels)

        train_losses.append(loss.item())

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


def val_step_AV(epoch, model, val_loader, params, device, tb):
    model.eval()
    val_losses = []

    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):
            ids1 = data['input_ids'][0].to(device, dtype=torch.long)
            mask1 = data['attention_mask'][0].to(device, dtype=torch.long)
            ids2 = data['input_ids'][1].to(device, dtype=torch.long)
            mask2 = data['attention_mask'][1].to(device, dtype=torch.long)

            logits = model.forward(ids1, mask1, ids2, mask2)

            # [1, 0] if the 2 from the same author, [0, 1] otherwise
            labels = data['labels'].to(device, dtype=torch.float)

            loss = loss_fn(logits, labels)
            val_losses.append(loss.item())

    average_val_loss = np.mean(val_losses)

    tb.add_scalar("val_loss", average_val_loss, epoch)
    print(f"** Finished validating epoch {epoch} **")

    return average_val_loss


def val_step_AV_classify(epoch, model, original_val_loader, original_train_loader, params, device, tb):
    model.eval()
    val_losses = []

    loss_fn = torch.nn.MSELoss()

    val_embeddings = []
    train_embeddings = []

    with torch.no_grad():
        for _, data in enumerate(original_val_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)

            embeddings = model.get_embedding(ids, mask).tolist()
            val_embeddings.extend(embeddings)

        for _, data in enumerate(original_train_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)

            embeddings = model.get_embedding(ids, mask).tolist()
            val_embeddings.extend(embeddings)


    average_val_loss = np.mean(val_losses)

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


if __name__ == "__main__":
    train_loop_AV(model_paramsAV1)