"""Data loading utilities for DBpedia title/content classification."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DBpediaExample:
    """One normalized DBpedia classification example."""

    label: int
    title: str
    content: str


def build_input_text(title: str, content: str) -> str:
    """Build the encoder input text from DBpedia title and content."""

    return f"title: {str(title).strip()}\ncontent: {str(content).strip()}"


class DBpediaDataset(Dataset):
    """Torch dataset that returns raw DBpedia examples for batched encoding."""

    def __init__(self, examples: Sequence[DBpediaExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return {
            "title": example.title,
            "content": example.content,
            "label": example.label,
        }


@dataclass
class DBpediaCollator:
    """Batch-tokenize raw DBpedia examples into classifier tensors."""

    tokenizer: object
    context_size: int

    def __call__(self, batch):
        from tokenizer import encode_batch_for_classification

        encoded = encode_batch_for_classification(
            self.tokenizer,
            titles=[item["title"] for item in batch],
            contents=[item["content"] for item in batch],
            context_size=self.context_size,
        )

        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "label": torch.tensor([item["label"] for item in batch], dtype=torch.long),
        }


def create_collate_fn(tokenizer, context_size: int) -> DBpediaCollator:
    """Create a dataloader collator that batch-tokenizes text rows."""

    return DBpediaCollator(tokenizer=tokenizer, context_size=context_size)


def load_parquet_split(path: Path) -> list[DBpediaExample]:
    """Load one DBpedia parquet split from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Parquet split not found: {path}")

    dataframe = pd.read_parquet(path, columns=["label", "title", "content"])
    expected_columns = {"label", "title", "content"}
    if set(dataframe.columns) != expected_columns:
        raise ValueError(
            f"Expected columns {sorted(expected_columns)}, got {sorted(dataframe.columns)}"
        )

    examples = [
        DBpediaExample(
            label=int(row.label),
            title=str(row.title),
            content=str(row.content),
        )
        for row in dataframe.itertuples(index=False)
    ]

    return examples


def split_train_validation(
    examples: Sequence[DBpediaExample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[DBpediaExample], list[DBpediaExample]]:
    """Split the official train split into local train and validation subsets."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")

    by_label: dict[int, list[DBpediaExample]] = {}
    for example in examples:
        by_label.setdefault(example.label, []).append(example)

    rng = random.Random(seed)
    train_examples: list[DBpediaExample] = []
    validation_examples: list[DBpediaExample] = []

    for label in sorted(by_label):
        label_examples = by_label[label]
        rng.shuffle(label_examples)
        validation_size = round(len(label_examples) * validation_fraction)

        validation_examples.extend(label_examples[:validation_size])
        train_examples.extend(label_examples[validation_size:])

    rng.shuffle(train_examples)
    rng.shuffle(validation_examples)

    return train_examples, validation_examples
