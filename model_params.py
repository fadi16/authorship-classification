
MODEL = "MODEL"
TRAIN_BATCH_SIZE = "TRAIN_BATCH_SIZE"
VALID_BATCH_SIZE = "VALID_BATCH_SIZE"
TRAIN_EPOCHS = "TRAIN_EPOCHS"
LEARNING_RATE = "LEARNING_RATE"
MAX_SOURCE_TEXT_LENGTH = "MAX_SOURCE_TEXT_LENGTH"
SEED = "SEED"
DROUPOUT = "DROUPOUT"
CHECKPOINT = "CHECKPOINT"
BERT = "BERT"
AA = "AA"
AV = "AV"
AAV = "AAV"
NO_AUTHORS = "NO_AUTHORS"

BETA_FOR_WEIGHTED_CLASS_LOSS = "BETA_FOR_WEIGHTED_CLASS_LOSS"
OUTPUT_DIR = "OUTPUT_DIR"
ADAM_EPSILON = "ADAM_EPSILON"
USE_SCHEDULER = "USE_SCHEDULER"
WARMUP_RATIO = "WARMUP_RATIO"
USE_CLASS_WEIGHTED_LOSS = "USE_CLASS_WEIGHTED_LOSS"
POOLING = "POOLING"
CLS = "CLS"
MEAN = "MEAN"
MULTIPLIER = "MULTIPLIER"
FREEZE_NO_EPOCHS = "FREEZE_NO_EPOCHS"
POSITIVE_LABEL = "POSITIVE_LABEL"
NEGATIVE_LABEL = "NEGATIVE_LABEL"

# bert authors recommend:
# batch sizes: 8, 16, 32, 64, 128. learning rates: 3e-4, 1e-4, 5e-5, 3e-5.


model_params_classification_10 = {
    MODEL : AA,
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
    MODEL : AA,
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


LOSS = "LOSS"
CONTRASTIVE = "CONTRASTIVE"
ONLINE_CONTRASTIVE = "ONLINE_CONTRASTIVE"
THRESHOLD = "THRESHOLD"
BATCH_HARD_TRIPLET = "BATCH_HARD_TRIPLET"
BALANCE = "BALANCE"
BEST_K = "BEST_K"

# bi_encoder_params_contrastive= {
#     CHECKPOINT: "bert-base-cased",
#     TRAIN_BATCH_SIZE: 32,
#     VALID_BATCH_SIZE: 32,
#     TRAIN_EPOCHS: 5,
#     LEARNING_RATE: 4e-5,
#     MAX_SOURCE_TEXT_LENGTH: 128,
#     USE_SCHEDULER: True,
#     WARMUP_RATIO: 0.06,
#     NO_AUTHORS: 10,
#     SEED: 42,
#     OUTPUT_DIR: "./output",
#     LOSS: CONTRASTIVE,
#     THRESHOLD: 0.5,
#     BALANCE: False
# }
#
# bi_encoder_params_online_contrastive= {
#     CHECKPOINT: "bert-base-cased",
#     TRAIN_BATCH_SIZE: 32,
#     VALID_BATCH_SIZE: 32,
#     TRAIN_EPOCHS: 5,
#     LEARNING_RATE: 4e-5,
#     MAX_SOURCE_TEXT_LENGTH: 128,
#     USE_SCHEDULER: True,
#     WARMUP_RATIO: 0.06,
#     NO_AUTHORS: 10,
#     SEED: 42,
#     OUTPUT_DIR: "./output",
#     POOLING: MEAN,
#     POSITIVE_LABEL: 1,
#     NEGATIVE_LABEL: 0,
#     LOSS: ONLINE_CONTRASTIVE,
#     THRESHOLD: 0.7716,
#     BALANCE: False
#
# }

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