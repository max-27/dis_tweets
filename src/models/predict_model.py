import pandas as pd
import transformers
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, IntervalStrategy
from src.data.dataset import TwitterDataset
import torch
import os
import numpy as np
from src.util import get_root_path

ROOT_DIR = get_root_path()
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")


def test(model_path: str):
    tweets_df_test = torch.load(os.path.join(DATA_DIR, "test.pt"))
    ids = pd.read_csv("/Users/max/Documents/Projects/dis_tweets/data/raw/test.csv")
    tweets_df_test["target"] = None
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    test_dataset = TwitterDataset(tweets_df_test, tokenizer, max_len=100)

    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    args = TrainingArguments(output_dir=" ", per_device_eval_batch_size=64)
    test_trainer = Trainer(model=model, args=args)
    raw_pred, _, _ = test_trainer.predict(test_dataset)
    pred = np.argmax(raw_pred, axis=1)
    out = pd.DataFrame()
    out["target"] = pred
    out["id"] = ids["id"]
    save_dir = os.path.join(get_root_path(), "reports", "submissions", "test_pred.csv")
    out.to_csv(save_dir, index=False)


if __name__ == "__main__":
    model_dir = os.path.join(ROOT_DIR, "models", "checkpoint-10")
    test(model_dir)
