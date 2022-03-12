import random

import pandas as pd

ORIGINAL_TRAIN_CSV_PATH = "data/original_train.csv"

ID = "id"
AUTHOR = "author"
TEXT = "text"

if __name__ == "__main__":
    original_train_df = pd.read_csv(ORIGINAL_TRAIN_CSV_PATH)

    indicies = list(range(len(original_train_df)))
    random.shuffle(indicies)

    ids = [original_train_df["id"][i] for i in indicies]
    authors = [original_train_df["author"][i] for i in indicies]
    texts = [original_train_df["text"][i] for i in indicies]

    last_train_index = int(len(ids) * 0.8)
    new_train_ids = ids[:last_train_index]
    new_train_authors = authors[:last_train_index]
    new_train_texts = texts[:last_train_index]

    new_train_df = pd.DataFrame({
        ID: new_train_ids,
        TEXT: new_train_texts,
        AUTHOR: new_train_authors
    })

    last_dev_index = int(len(ids) * 0.9)
    new_dev_ids = ids[last_train_index: last_dev_index]
    new_dev_authors = authors[last_train_index: last_dev_index]
    new_dev_texts = texts[last_train_index: last_dev_index]

    new_dev_df = pd.DataFrame({
        ID: new_dev_ids,
        TEXT: new_dev_texts,
        AUTHOR: new_dev_authors
    })

    new_test_ids = ids[last_dev_index:]
    new_test_authors = authors[last_dev_index:]
    new_test_texts = texts[last_dev_index:]

    new_test_df = pd.DataFrame({
        ID: new_test_ids,
        TEXT: new_test_texts,
        AUTHOR: new_test_authors
    })

    new_train_df.to_csv("./data/train.csv")
    new_dev_df.to_csv("./data/val.csv")
    new_test_df.to_csv("./data/test.csv")

