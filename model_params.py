
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
# todo: need to try warmup and decay
# todo: model is overfitting

# bert authors recommend:
# batch sizes: 8, 16, 32, 64, 128. learning rates: 3e-4, 1e-4, 5e-5, 3e-5.

# gives 0.36 sigmoid log loss, 87.9 accuracy
model_params_spooky_authors = {
    MODEL: "BERT",
    CHECKPOINT: "bert-base-cased",
    # a bigger batch size than 16 gives cuda out of memory errors on colab
    TRAIN_BATCH_SIZE: 16,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 10,
    LEARNING_RATE: 1e-5,
    MAX_SOURCE_TEXT_LENGTH: 512,
    SEED: 42,
    DROUPOUT: 0.3,
    BETA_FOR_WEIGHTED_CLASS_LOSS: 0.9999,
    OUTPUT_DIR: "./output"
}

model_params1 = {
    MODEL : AA,
    CHECKPOINT: "bert-base-cased",
    TRAIN_BATCH_SIZE: 8,
    VALID_BATCH_SIZE: 16,
    TRAIN_EPOCHS: 10,
    ADAM_EPSILON: 1e-8,
    LEARNING_RATE: 4e-5,
    MAX_SOURCE_TEXT_LENGTH: 128,
    USE_SCHEDULER: True,
    WARMUP_RATIO: 0.06,
    USE_CLASS_WEIGHTED_LOSS: False,
    NO_AUTHORS: 5,
    SEED: 42,
    OUTPUT_DIR: "./output"

}