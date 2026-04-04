import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

try:
    from src.nlp.prepare_data import LABEL_LIST, prepare_dataset
except ImportError:
    from prepare_data import LABEL_LIST, prepare_dataset


LABEL2ID = {label: index for index, label in enumerate(LABEL_LIST)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 192
OUTPUT_DIR = Path("results")
LOGGING_DIR = Path("logs")
MODEL_DIR = Path("models/ner-bert")


def get_runtime_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA GPU: {device_name}")
        return "cuda"

    print("CUDA was not detected. Falling back to CPU.")
    return "cpu"


def get_training_config(runtime_device):
    if runtime_device != "cuda":
        return {
            "bf16": False,
            "fp16": False,
            "per_device_train_batch_size": 4,
            "tf32": torch.cuda.get_device_capability(0)[0] >= 8,
        }

    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16_enabled = torch.cuda.is_bf16_supported()
    batch_size = 16 if gpu_memory_gb >= 14 else 8

    return {
        "bf16": bf16_enabled,
        "fp16": not bf16_enabled,
        "per_device_train_batch_size": batch_size,
        "tf32": torch.cuda.get_device_capability(0)[0] >= 8,
    }


def configure_torch_backend(runtime_device):
    if runtime_device != "cuda":
        return

    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_tokenized_dataset(csv_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    data = prepare_dataset(csv_path)
    dataset = Dataset.from_list(data)

    def tokenize_and_align_labels(example):
        tokenized_inputs = tokenizer(
            example["tokens"],
            truncation=True,
            max_length=MAX_LENGTH,
            is_split_into_words=True,
        )

        word_ids = tokenized_inputs.word_ids()
        labels = example["ner_tags"]
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(LABEL2ID.get(labels[word_idx], 0))
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        tokenized_inputs["labels"] = label_ids
        return tokenized_inputs

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset, tokenizer


def train_ner_model(csv_path="data/processed/cve_dataset.csv"):
    dataset_path = Path(csv_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run main.py first to generate it."
        )

    runtime_device = get_runtime_device()
    training_config = get_training_config(runtime_device)
    configure_torch_backend(runtime_device)
    dataset, tokenizer = build_tokenized_dataset(str(dataset_path))
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(LOGGING_DIR)

    steps_per_epoch = max(
        1,
        len(dataset) // training_config["per_device_train_batch_size"],
    )
    warmup_steps = max(1, int(steps_per_epoch * 0.1))

    print(
        "Training config:",
        {
            "batch_size": training_config["per_device_train_batch_size"],
            "bf16": training_config["bf16"],
            "fp16": training_config["fp16"],
            "tf32": training_config["tf32"],
            "max_length": MAX_LENGTH,
            "examples": len(dataset),
        },
    )

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=100,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        dataloader_pin_memory=runtime_device == "cuda",
        seed=42,
        bf16=training_config["bf16"],
        fp16=training_config["fp16"],
        tf32=training_config["tf32"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    print(f"Saved fine-tuned model to {MODEL_DIR}")


if __name__ == "__main__":
    train_ner_model()
