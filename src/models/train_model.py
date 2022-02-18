import transformers
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, IntervalStrategy
from torch.utils.data import random_split
from src.data.dataset import TwitterDataset
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from src.util import get_root_path
import logging
import wandb
import hydra
from omegaconf import DictConfig
from git import Repo

ROOT_DIR = get_root_path()
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
logger = logging.getLogger(__name__)
wandb.login()
repo = Repo(ROOT_DIR, search_parent_directories=True)
os.environ['HYDRA_FULL_ERROR'] = "1"


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    acc = accuracy_score(labels, pred)
    recall = recall_score(labels, pred)
    f1 = f1_score(labels, pred)
    precision = precision_score(labels, pred)
    return {"eval_accuracy": acc, "eval_f1": f1, "eval_recall": recall, "eval_precision": precision}


def train(cfg: DictConfig):
    train_cfg = cfg.train
    optimizer_cfg = cfg.optim
    data_cfg = cfg.data
    logging_cfg = cfg.logging
    model_cfg = cfg.model

    if not model_cfg.metric_best_model and logging_cfg.save_steps != train_cfg.evaluation.steps:
        logger.warning("Evaluation steps and saving steps are not the same!")

    wandb.init(
        project=logging_cfg.wandb.project,
        name=logging_cfg.wandb.name,
        tags=logging_cfg.wandb.tags,
        group=logging_cfg.wandb.group,
        # config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    )

    save_subdir = logging_cfg.save_dir.split("/")
    output_dir = os.path.join(MODEL_DIR, save_subdir[0], save_subdir[1])
    evaluation_strategy = IntervalStrategy(value=train_cfg.evaluation.evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=output_dir,
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
        logging_steps=logging_cfg.logging_steps,
        load_best_model_at_end=model_cfg.load_best_model,
        metric_for_best_model=model_cfg.metric_best_model,
    )

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-cased",
            num_labels=model_cfg.num_classes,
        )

    tweets_df_train = torch.load(os.path.join(DATA_DIR, "train.pt"))
    tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-cased")
    full_train_dataset = TwitterDataset(tweets_df_train, tokenizer, max_len=120)
    val_size = int(len(tweets_df_train) * 0.1)
    train_size = len(tweets_df_train) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    os.environ['WANDB_WATCH'] = 'false'  # used in Trainer
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # data_cfg.data_version = next((tag for tag in repo.tags if tag.commit == repo.head.commit)).name
    logger.info(f"Using {training_args.n_gpu} GPUs for training")
    logger.info(f"Using data version: {data_cfg.data_version}")
    trainer.train()
    trainer.evaluate()


@hydra.main(config_path=os.path.join(ROOT_DIR, "config"), config_name="default")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
