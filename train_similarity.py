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
from sklearn.metrics.pairwise import cosine_distances


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
    print(f"Training with {len(train_dataset)} samples")

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

    collator_av = CollatorAV(pad_token_id=tokenizer.pad_token_id)
    collator_aa = CollatorAA(pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator_av.collate_batch
    )

    train_loader_original = DataLoader(
        dataset=train_dataset_original,
        batch_size=params[TRAIN_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator_aa.collate_batch
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator_av.collate_batch
    )

    val_loader_original = DataLoader(
        dataset=val_dataset_original,
        batch_size=params[VALID_BATCH_SIZE],
        shuffle=True,
        num_workers=0,
        collate_fn=collator_aa.collate_batch
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

        train_loss = train_step_AV(epoch, model, optimizer, scheduler, train_loader, params, None, device, tb)
        print(f"train_loss = {train_loss}")

        val_loss = val_step_AV(epoch, model, val_loader, params, device, tb)
        print(f"val_loss = {val_loss}")

        val_classification_accuracy = val_step_AV_classify(epoch, model, val_loader_original, train_loader_original, params, device, tb)
        print(f"val_classification_accuracy = {val_classification_accuracy}")

        if val_loss < best_loss:
            best_loss = val_loss

            # save the best model so far
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoints")
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print(f"SAVED MODEL AT from epoch {epoch} at " + model_checkpoint_path + "\n")

        print(f"Finished Epoch {epoch} val_loss = {val_loss}, best val_loss = {best_loss}")
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
    return average_train_loss


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
    return average_val_loss


def val_step_AV_classify(epoch, model, original_val_loader, original_train_loader, params, device, tb):
    model.eval()

    # classify based on the authors of the top_k highest ranked samples in the training set
    top_k = 10

    val_embeddings = []
    train_embeddings = []

    train_labels = []
    actual_val_labels = []
    predicted_val_labels = []

    with torch.no_grad():
        for _, data in enumerate(original_val_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)
            actual_val_labels.extend(labels.tolist())

            embeddings = model.get_embedding(ids, mask).tolist()
            val_embeddings.extend(embeddings)

        for _, data in enumerate(original_train_loader, 0):
            ids = data['input_ids'].to(device, dtype=torch.long)
            mask = data['attention_mask'].to(device, dtype=torch.long)
            labels = data['labels'].to(device, dtype=torch.float)
            train_labels.extend(labels.tolist())

            embeddings = model.get_embedding(ids, mask).tolist()
            train_embeddings.extend(embeddings)

        val_embeddings = np.array(val_embeddings)
        train_embeddings = np.array(train_embeddings)

        for i in range(len(val_embeddings)):
            candidate_labels = []

            val_embedding = [val_embeddings[i]]
            cos_sims = cosine_distances(val_embedding, train_embeddings)
            sorted_indicies = np.argsort(cos_sims)[0][:top_k]
            for topk_index in sorted_indicies:
                candidate_label = train_labels[topk_index]
                candidate_labels.append(candidate_label.index(max(candidate_label)))

            # now we choose the label with the highest count
            voted_label = max(set(candidate_labels), key = candidate_labels.count)
            predicted_val_labels.append([0 if j != voted_label else 1 for j in range(params[NO_AUTHORS])])

    accuracy = metrics.accuracy_score(actual_val_labels, predicted_val_labels)
    tb.add_scalar("classification_accuracy", accuracy, epoch)

    return accuracy


if __name__ == "__main__":
    train_loop_AV(model_paramsAV1)
