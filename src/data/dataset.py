from typing import Optional, Any
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import transformers
import os
import torch


class TwitterDataset(Dataset):
    def __init__(self, tweets_df: pd.DataFrame, tokenizer, max_len):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tweets_df = tweets_df

    def __len__(self):
        return len(self.tweets_df)

    def __getitem__(self, idx):
        sample = self.tweets_df["clean_text"].iloc[idx]
        location = self.tweets_df["location"].iloc[idx]
        location = location if location != "nan" else ""
        keyword = self.tweets_df["keyword"].iloc[idx]
        keyword = keyword if keyword != "nan" else ""
        target = self.tweets_df["target"].iloc[idx]
        encoding = self.tokenizer(
            sample,
            location,
            keyword,
            # add_special_tokens=True,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {"sample": sample + " " + location + " " + keyword,
                "labels": target,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                }


class TwitterDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        tokenizer: Any,
        max_len: int = 100,
        batch_size: int = 32,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None, val_ratio: Optional[int] = 0.2) -> None:
        if stage == "fit" or stage is None:
            tweets_df_train = torch.load(os.path.join(self.data_dir, "train.pt"))
            full_train_dataset = TwitterDataset(tweets_df_train, self.tokenizer, self.max_len)
            val_size = int(len(tweets_df_train)*val_ratio)
            train_size = len(tweets_df_train) - val_size
            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])
        if stage == "test":
            tweets_df_test = torch.load(os.path.join(self.data_dir, "test.pt"))
            tweets_df_test["target"] = -1
            self.test_dataset = TwitterDataset(tweets_df_test, self.tokenizer, self.max_len)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    data_folder = "/Users/max/Documents/Projects/dis_tweets/data/processed"
    tokenizer_test = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    tw_datamodule = TwitterDataModule(data_folder, tokenizer_test, max_len=100)
    tw_datamodule.setup(stage="test")
    test_loader = tw_datamodule.test_dataloader()
