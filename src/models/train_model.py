import transformers
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, IntervalStrategy
from torch.utils.data import random_split
from src.data.dataset import TwitterDataset
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from src.utils import get_root_path
import logging
import wandb
import hydra
from omegaconf import DictConfig

ROOT_DIR = get_root_path()
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
logger = logging.getLogger(__name__)
wandb.login()


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(labels, pred)
    recall = recall_score(labels, pred)
    f1 = f1_score(labels, pred)
    precision = precision_score(labels, pred)
    return {"accuracy": acc, "f1": f1, "recall": recall, "precision": precision}


def train(cfg: DictConfig):
    train_cfg = cfg.train
    optimizer_cfg = cfg.optim
    data_cfg = cfg.data
    logging_cfg = cfg.logging
    model_cfg = cfg.model

    wandb.init(
        project=logging_cfg.wandb.project,
        name=logging_cfg.wandb.name,
        tags=logging_cfg.wandb.tags,
        group=logging_cfg.wandb.group,
    )

    evaluation_strategy = IntervalStrategy(value=train_cfg.evaluation.evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        save_steps=logging_cfg.save_steps,
        seed=train_cfg.random_seed,
        num_train_epochs=train_cfg.training.epochs,
        max_steps=train_cfg.training.max_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=train_cfg.evaluation.steps,
        fp16=train_cfg.fp16,
        max_grad_norm=train_cfg.max_grad_norm,
        per_device_train_batch_size=data_cfg.batch_size.train,
        per_device_eval_batch_size=data_cfg.batch_size.eval,
        learning_rate=optimizer_cfg.optimizer.lr,
        adam_epsilon=optimizer_cfg.optimizer.adam_epsilon,
        warmup_steps=optimizer_cfg.optimizer.lr_scheduler.warmup_steps,
        weight_decay=optimizer_cfg.loss.weight_decay,
        report_to=["wandb"],
        run_name=logging_cfg.wandb.name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=model_cfg.num_classes)

    tweets_df_train = torch.load(os.path.join(DATA_DIR, "train.pt"))
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    full_train_dataset = TwitterDataset(tweets_df_train, tokenizer, max_len=100)
    val_size = int(len(tweets_df_train) * 0.2)
    train_size = len(tweets_df_train) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # optimizers=
    )

    logger.info(f"Using {training_args.n_gpu} GPUs for training")
    trainer.train()


@hydra.main(config_path=os.path.join(ROOT_DIR, "config"), config_name="default")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
