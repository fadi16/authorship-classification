import os
import random
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from model_params import *
from seed import *
from evaluation import evaluation_stats
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import DataCollator


######################### ABBREVIATIOS ######################################################################
# AA/Classification: means authorship attribution - given a piece of text, predict its author
#############################################################################################################

# ####################### WHAT DOES THIS MODEL DO ? ########################################################
# This model is a Bert with a classification head.
# This is the current SoTA for AA on the blogs dataset (https://aclanthology.org/2020.icon-main.16/)
# This file contains the training loop and the testing for the model
#
# We only report scores for a subset of the metrics used. Each test writes the results to a csv file,
# which will be then used to report performance against more metrics in evaluation.py


# It encodes texts and passes it along with a label (for the author of the text) and an attention mask
# We only consider the first 128 bert tokens, following Fabien et al. (2020) in BertAA (https://aclanthology.org/2020.icon-main.16.pdf)
class AuthorsDatasetAA(Dataset):
    # set pad_to_max_length to false when we want to do dynamic padding
    def __init__(self, df, source_tag, target_tag, tokenizer, max_source_len, pad_to_max_length=False):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.source_tag = source_tag
        self.source_text = df[source_tag].tolist()
        self.source_text = [sent.strip() for sent in self.source_text]

        longest_source_sequence = len(max(self.source_text, key=lambda x: len(x.split())).split())
        print("longest_source_sequence = ", longest_source_sequence)

        self.target_tag = target_tag
        self.target_classes = df[target_tag].tolist()
        unique_target_classes = set(self.target_classes)
        self.no_authors = len(unique_target_classes)
        self.author_index_to_no_samples = dict(zip(list(unique_target_classes), [0] * len(unique_target_classes)))
        for author_ind in self.target_classes:
            self.author_index_to_no_samples[author_ind] += 1

        self.one_hot_target_classes = self.get_one_hot_target_classes()
        self.pad_to_max_length = pad_to_max_length
        assert len(self.source_text) == len(self.one_hot_target_classes)

    def get_one_hot_target_classes(self):
        one_hot_targets = []
        for target_class in self.target_classes:
            one_hot_target = [1 if i == target_class else 0 for i in range(self.no_authors)]
            one_hot_targets.append(one_hot_target)
        return one_hot_targets

    def __len__(self):
        """returns the length of the dataframe"""
        return len(self.target_classes)

    def __getitem__(self, index) -> Dict[str, List]:
        """return input ids, attention marks and target ids"""
        source_text = self.source_text[index]

        # tokenizing source
        source = self.tokenizer(
            source_text,
            max_length=self.max_source_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True
        )

        labels = self.one_hot_target_classes[index]

        return {
            "input_ids": source["input_ids"],
            "attention_mask": source["attention_mask"],
            "labels": labels,
        }


# this allows us to do dynamic padding for batches.
# It significantly speeds up training time
class CollatorAA:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def collate_batch(self, batch: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        def pad_seq(seq: List[int], max_batch_len: int, pad_value: int) -> List[int]:
            return seq + (max_batch_len - len(seq)) * [pad_value]

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        max_size = max([len(sample["input_ids"]) for sample in batch])
        for sample_dict in batch:
            batch_input_ids += [pad_seq(sample_dict["input_ids"], max_size, self.pad_token_id)]
            batch_attention_mask += [pad_seq(sample_dict["attention_mask"], max_size, self.pad_token_id)]
            batch_labels.append(sample_dict["labels"])

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.float)
        }


def train_loop_AA(params):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for producing graphs with tensorboard
    tb = SummaryWriter()

    train_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/train_{params[NO_AUTHORS]}_authors.csv")
    val_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/val_{params[NO_AUTHORS]}_authors.csv")

    tokenizer = AutoTokenizer.from_pretrained(params[CHECKPOINT])
    config = AutoConfig.from_pretrained(params[CHECKPOINT], num_labels=params[NO_AUTHORS])
    # Bert with a classification head supporting NO_AUTHORS many classes
    model = AutoModelForSequenceClassification.from_pretrained(params[CHECKPOINT], config=config).to(device)
    optimizer = transformers.AdamW(params=model.parameters(), lr=params[LEARNING_RATE])
    scheduler = None
    if params[USE_SCHEDULER]:
        no_training_steps = params[TRAIN_EPOCHS] * (len(train_df) // params[TRAIN_BATCH_SIZE])
        no_warmup_steps = params[WARMUP_RATIO] * no_training_steps
        # use a linear scheduler with warmup and decay
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=no_warmup_steps,
                                                                 num_training_steps=no_training_steps)

    train_dataset = AuthorsDatasetAA(train_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
                                     pad_to_max_length=False)
    val_dataset = AuthorsDatasetAA(val_df, "content", "Target", tokenizer, params[MAX_SOURCE_TEXT_LENGTH],
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

    ######################################

    if params[USE_CLASS_WEIGHTED_LOSS]:
        class_weights = get_ens_class_weights(params[BETA_FOR_WEIGHTED_CLASS_LOSS],
                                              train_dataset.author_index_to_no_samples)
        class_weights.to(device, dtype=torch.float)

    else:
        class_weights = None

    # dynamic padding of batches to speed up the training
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

        # save the checkpoint with the lowest cross entropy loss
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
    # use cross entropy loss (not binary cross entropy loss because that's for multi class multi label - our problem
    # is not multi label) we shouldn't apply a softmax on the output of the model because the CrossEntropyLoss
    # function internally does that
    loss_fn = torch.nn.CrossEntropyLoss(
        # supply loss weights
        weight=class_weights.to(device, dtype=float) if class_weights is not None else None)

    train_losses = []
    with tqdm(total=len(training_loader)) as pbar:
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

            pbar.update(1)

    average_train_loss = np.mean(train_losses)
    tb.add_scalar("train_loss", average_train_loss, epoch)
    print(f"Average Train Loss = {average_train_loss}")


def val_step_AA(epoch, model, val_loader, device, tb):
    model.eval()
    all_labels = []
    all_outputs_with_softmax = []
    val_losses = []
    loss_fn = torch.nn.CrossEntropyLoss()

    with tqdm(total=len(val_loader)) as pbar:
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

                pbar.update(1)

    accuracy, precision, recall, f1 = get_eval_scores(
        all_outputs_with_softmax, all_labels)

    average_val_loss = np.mean(val_losses)

    print(f"Average Validation Loss = {average_val_loss}")
    print(f"** Finished validating epoch {epoch} **")
    print(f"Accuracy Score = {accuracy}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")

    return average_val_loss


def get_eval_scores(outputs, labels):
    pred_labels = [get_one_hot_class_from_probs(output) for output in outputs]
    # no. correctly classified / total number of samples
    accuracy = metrics.accuracy_score(labels, pred_labels)
    overall_precision_recall_f1 = precision_recall_fscore_support(labels, pred_labels, average="macro")
    overall_precision = overall_precision_recall_f1[0]
    overall_recall = overall_precision_recall_f1[1]
    overall_f1 = overall_precision_recall_f1[2]

    return accuracy, overall_precision, overall_recall, overall_f1


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


def test(params, test_csv=None):
    # for reproducibility
    seed_for_reproducability(params[SEED])

    # use gpu if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if test_csv:
        test_df = pd.read_csv(test_csv)
    else:
        test_df = pd.read_csv(f"./data/blog/{params[NO_AUTHORS]}_authors/test_{params[NO_AUTHORS]}_authors.csv")

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

    accuracy, precision, recall, f1 = get_eval_scores(
        all_outputs_with_softmax, all_labels)

    print(f"Accuracy Score = {accuracy}")
    print(f"precision = {precision}")
    print(f"recall = {recall}")
    print(f"f1 = {f1}")

    predictions_df = pd.DataFrame({
        "predicted_labels": predicted_labels_indidices,
        "actual_labels": actual_labels_indicies
    })
    predictions_df.to_csv(f"classifier_predictions_{params[NO_AUTHORS]}_authors.csv")


def get_one_hot_class_from_probs(probs: List[float]):
    max_prob = max(probs)
    return [0 if p != max_prob else 1 for p in probs]


def get_class_index_from_probs(probs: List[float]):
    max_prob = max(probs)
    return probs.index(max_prob)


def rescale_probabilities(probs: List[float]):
    total = sum(probs)
    return [prob / total for prob in probs]


def demo():
    params = model_params_classification_10
    test(params=params, test_csv=f"./data/blog/10_authors/demo_test_10_authors.csv")
    evaluation_stats(f"classifier_predictions_{params[NO_AUTHORS]}_authors.csv")


if __name__ == "__main__":
    demo()
