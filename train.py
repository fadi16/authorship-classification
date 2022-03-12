import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from dataset import AuthorsDataset, Collator
from model_params import *

TRAIN_DATA_CSV_PATH = "./data/train.csv"
VAL_DATA_CSV_PATH = "./data/val.csv"
TEST_DATA_CSV_PATH = "./data/test.csv"


def train_loop(params):

    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for producing graphs with tensorboard
    tb = SummaryWriter()

    train_df = pd.read_csv(TRAIN_DATA_CSV_PATH)
    val_df = pd.read_csv(VAL_DATA_CSV_PATH)

    if params[MODEL] == BERT:
        tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])
        # model = BertForAuthorshipIdentification(droupout=params[DROUPOUT], checkpoint=[CHECKPOINT])
        config = AutoConfig.from_pretrained(params[CHECKPOINT], num_labels=3)
        model = AutoModelForSequenceClassification.from_pretrained(params[CHECKPOINT], config=config).to(device)
        print(model)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=params[LEARNING_RATE])
    else:
        raise Exception("Unknown Model")

    train_dataset = AuthorsDataset(train_df, tokenizer, params[MAX_SOURCE_TEXT_LENGTH], pad_to_max_length=False)
    val_dataset = AuthorsDataset(val_df, tokenizer, params[MAX_SOURCE_TEXT_LENGTH], pad_to_max_length=False)

    class_weights = get_ens_class_weights(params[BETA_FOR_WEIGHTED_CLASS_LOSS], train_dataset.author_index_to_no_samples)
    class_weights.to(device, dtype=torch.float)

    collator = Collator(pad_token_id=tokenizer.pad_token_id)

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

    best_log_loss = 1
    print("begin training")
    for epoch in range(params[TRAIN_EPOCHS]):
        train_step(epoch, model, optimizer, train_loader, class_weights, device, tb)
        log_loss = val_step(epoch, model, val_loader, device, tb)

        if log_loss < best_log_loss:
            best_log_loss = log_loss

            # save the best model so far
            model_checkpoint_path = os.path.join(params[OUTPUT_DIR], "checkpoints")
            model.save_pretrained(model_checkpoint_path)
            tokenizer.save_pretrained(model_checkpoint_path)
            print("SAVED MODEL AT " + model_checkpoint_path + "\n")

        print(f"Epoch {epoch} log_loss = {log_loss}, best log_loss = {best_log_loss}")
        print("**" * 30)


def train_step(epoch, model, optimizer, training_loader, class_weights, device, tb):
    model.train()
    # use cross entropy loss (not binary cross entropy loss because that's for multi class multi label - out problem is not multi label)
    # we shouldn't apply a softmax on the output of the model because the CrossEntropyLoss function internally does that
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device, dtype=float))
    train_losses = []
    for _, data in enumerate(training_loader, 0):
        ids = data['input_ids'].to(device, dtype=torch.long)
        mask = data['attention_mask'].to(device, dtype=torch.long)
        labels = data['labels'].to(device, dtype=torch.float)

        outputs = model(ids, mask)
        output_logits = outputs.logits

        optimizer.zero_grad()
        loss = loss_fn(output_logits, labels)
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = np.mean(train_losses)
    tb.add_scalar("train_loss", average_train_loss, epoch)
    print(f"Average Train Loss = {average_train_loss}")


def val_step(epoch, model, val_loader, device, tb):
    model.eval()
    all_labels = []
    all_outputs_with_sigmoid = []
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
            all_outputs_with_sigmoid.extend(torch.sigmoid(output_logits).cpu().detach().numpy().tolist())
            all_outputs_with_softmax.extend(torch.softmax(output_logits, dim=1).cpu().detach().numpy().tolist())

    log_loss_sigmoid, accuracy_sigmoid, f1_score_micro_sigmoid, f1_score_macro_sigmoid = get_eval_scores(all_outputs_with_sigmoid, all_labels)
    log_loss_softmax, accuracy_softmax, f1_score_micro_softmax, f1_score_macro_softmax = get_eval_scores(all_outputs_with_softmax, all_labels)

    average_val_loss = np.mean(val_losses)

    # results with sigmoid and softmax
    print(f"Average Validation Loss = {average_val_loss}")
    print(f"** Finished validating epoch {epoch} **")
    print(f"Accuracy Score = {accuracy_sigmoid}")
    print(f"F1 Score (Micro) = {f1_score_micro_sigmoid}")
    print(f"F1 Score (Macro) = {f1_score_macro_sigmoid}")
    print(f"Multi Class Log Loss (sigmoid) = {log_loss_sigmoid}")
    print(f"Multi Class Log Loss (softmax) = {log_loss_softmax}")

    tb.add_scalar("val_loss", average_val_loss, epoch)
    tb.add_scalar("val_accuracy", accuracy_sigmoid, epoch)
    tb.add_scalar("val_f1_score_macro", f1_score_macro_sigmoid, epoch)
    tb.add_scalar("val_f1_score_micro", f1_score_micro_sigmoid, epoch)
    tb.add_scalar("val_log_loss_sigmoid", log_loss_sigmoid, epoch)
    tb.add_scalar("val_log_loss_softmax", log_loss_softmax, epoch)

    return log_loss_sigmoid


def get_eval_scores(outputs, labels):
    pred_labels = [get_class_from_probs(output) for output in outputs]
    # no. correctly classified / total number of samples
    accuracy = metrics.accuracy_score(labels, pred_labels)
    f1_score_micro = metrics.f1_score(labels, pred_labels, average='micro')
    f1_score_macro = metrics.f1_score(labels, pred_labels, average='macro')

    probs = [rescale_probabilities(output) for output in outputs]
    # this clips probabilities - like they do in the experiment (they even use the same parameter)
    log_loss = metrics.log_loss(labels, probs)

    return log_loss, accuracy, f1_score_micro, f1_score_macro


# weighting classes based on the effective number of samples - from paper Class-Balanced Loss Based on Effective Number of Samples
# https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.pdf
def get_ens_class_weights(beta, class_to_no_samples):
    weights = []
    for cls, no_samples in class_to_no_samples.items():
        # weight for class = 1 / effective no. samples = 1 / (1 - beta ** no_samples) / (1 - beta)
        w_cls = (1 - beta) / (1 - beta ** no_samples)
        weights.append(w_cls)

    # normalize to make the total loss roughly in the same scale when applying the weights
    sum_weights = sum(weights)
    weights = [3 * w / sum_weights for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


# to be used when we use sigmoid to get the probabilities, cuz sigmoid doesn't not give probabilities that sum to 1
def rescale_probabilities(probs: List[float]):
    total = sum(probs)
    return [prob / total for prob in probs]


def get_class_from_probs(probs: List[float]):
    max_prob = max(probs)
    return [0 if p != max_prob else 1 for p in probs]


def seed_for_reproducability(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    train_loop(model_params_1)
