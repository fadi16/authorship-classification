TRAIN_BATCH_SIZE = "TRAIN_BATCH_SIZE"
VALID_BATCH_SIZE = "VALID_BATCH_SIZE"
TRAIN_EPOCHS = "TRAIN_EPOCHS"
LEARNING_RATE = "LEARNING_RATE"
# the maximum token length to be used by bert, set to 128, to compare to the SoTA from https://aclanthology.org/2020.icon-main.16/,
# whose code is avaliable here https://colab.research.google.com/drive/1m4anWkkb8tz3fKvzJFytygBkqCTdZ8bo?usp=sharing
MAX_SOURCE_TEXT_LENGTH = "MAX_SOURCE_TEXT_LENGTH"
SEED = "SEED"
DROUPOUT = "DROUPOUT"
# checkpoint path
CHECKPOINT = "CHECKPOINT"
BERT = "BERT"
NO_AUTHORS = "NO_AUTHORS"
# the beta hyperparameter from https://openaccess.thecvf.com/content_CVPR_2019/papers/Cui_Class for weighted loss
BETA_FOR_WEIGHTED_CLASS_LOSS = "BETA_FOR_WEIGHTED_CLASS_LOSS"
OUTPUT_DIR = "OUTPUT_DIR"
ADAM_EPSILON = "ADAM_EPSILON"
# whether to use a scheduler for the learning rate
USE_SCHEDULER = "USE_SCHEDULER"
# the learning rate will warm-up, i.e. increase from zero to highest value in WARMUP_RATIO * no_training_steps many steps
WARMUP_RATIO = "WARMUP_RATIO"
# whether to use weighted loss
USE_CLASS_WEIGHTED_LOSS = "USE_CLASS_WEIGHTED_LOSS"
# for pooling strategy
POOLING = "POOLING"
# pooling by only taking the embedding of the cls token from bert
CLS = "CLS"
# pooling by taking the mean embeddings of all Bert tokens - taking the attention mask into account
MEAN = "MEAN"
# if we want to freeze Bert's weights for a given number of epochs
FREEZE_NO_EPOCHS = "FREEZE_NO_EPOCHS"
POSITIVE_LABEL = "POSITIVE_LABEL"
NEGATIVE_LABEL = "NEGATIVE_LABEL"
# which loss to use mainly for the bi-encoder
LOSS = "LOSS"
# contrastive loss
CONTRASTIVE = "CONTRASTIVE"
# online contrastive loss - based on hard positive and hard negative pairs
ONLINE_CONTRASTIVE = "ONLINE_CONTRASTIVE"
# batch hard triplet loss
BATCH_HARD_TRIPLET = "BATCH_HARD_TRIPLET"
# threshold for authorship verification, if prediction > threshold, label is 1, i.e. samples were written by same author
THRESHOLD = "THRESHOLD"
# whether to use a balanced dataset or not, more details on balancing in blog_dataset.py
BALANCE = "BALANCE"
# best K value for KNN classification - mainly for bi-encoder
BEST_K = "BEST_K"

# bert authors recommend:
# batch sizes: 8, 16, 32, 64, 128. learning rates: 3e-4, 1e-4, 5e-5, 3e-5.


model_params_classification_10 = {
    CHECKPOINT: "./output/checkpoints/classifier-10",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    ADAM_EPSILON: 1e-8,
    LEARNING_RATE: 4e-5,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    USE_CLASS_WEIGHTED_LOSS: True,
    NO_AUTHORS: 10,
    SEED: 42,
    OUTPUT_DIR: "./output/classification-10",
    BETA_FOR_WEIGHTED_CLASS_LOSS: 0.9999,
}

model_params_classification_50 = {
    CHECKPOINT: "bert-base-cased",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    ADAM_EPSILON: 1e-8,
    LEARNING_RATE: 4e-5,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    USE_CLASS_WEIGHTED_LOSS: True,
    NO_AUTHORS: 50,
    SEED: 42,
    OUTPUT_DIR: "./output/classification-50",
    BETA_FOR_WEIGHTED_CLASS_LOSS: 0.9999,
}

bi_encoder_params_contrastive= {
    CHECKPOINT: "bert-base-cased",
    TRAIN_BATCH_SIZE: 32,
    VALID_BATCH_SIZE: 32,
    TRAIN_EPOCHS: 5,
    LEARNING_RATE: 4e-5,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 10,
    SEED: 42,
    OUTPUT_DIR: "./output",
    LOSS: CONTRASTIVE,
    THRESHOLD: 0.5,
    BALANCE: False
}

bi_encoder_params_online_contrastive= {
    CHECKPOINT: "bert-base-cased",
    TRAIN_BATCH_SIZE: 32,
    VALID_BATCH_SIZE: 32,
    TRAIN_EPOCHS: 5,
    LEARNING_RATE: 4e-5,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 10,
    SEED: 42,
    OUTPUT_DIR: "./output",
    POOLING: MEAN,
    POSITIVE_LABEL: 1,
    NEGATIVE_LABEL: 0,
    LOSS: ONLINE_CONTRASTIVE,
    THRESHOLD: 0.7716,
    BALANCE: False

}

bi_encoder_params_batch_hard_triplet_10= {
    CHECKPOINT: "./output/checkpoints/bi-encoder-10",
    TRAIN_BATCH_SIZE: 16,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 10,
    SEED: 42,
    OUTPUT_DIR: "./output/bi-encoder-10",
    LOSS: BATCH_HARD_TRIPLET,
    THRESHOLD: 0.1994,
    BALANCE: True,
    BEST_K: 161
}

bi_encoder_params_batch_hard_triplet_50 = {
    CHECKPOINT: "./output/checkpoints/bi-encoder-50",
    TRAIN_BATCH_SIZE: 16,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 50,
    SEED: 42,
    OUTPUT_DIR: "./output/bi-encoder-50",
    LOSS: BATCH_HARD_TRIPLET,
    THRESHOLD: 0.5, # todo need to find the right one
    BALANCE: True,
    BEST_K: 118 # todo needs fine tuning
}

cross_encoder_params_10= {
    CHECKPOINT: "output/checkpoints/cross-encoder-10",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 10,
    SEED: 42,
    OUTPUT_DIR: "./output/cross-encoder-10",
    THRESHOLD: 0.61,
    BALANCE: True
}

cross_encoder_params_50= {
    CHECKPOINT: "output/checkpoints/cross-encoder-50",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 7,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    NO_AUTHORS: 50,
    SEED: 42,
    OUTPUT_DIR: "./output/cross-encoder-50",
    THRESHOLD: 0.324,
    BALANCE: True
}