"""Tokenizer utilities for DBpedia title/content text."""

from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

from data import DBpediaExample, build_input_text


SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]


def get_tokenizer_path(config) -> Path:
    """Return the configured DBpedia tokenizer file path."""

    return config.tokenizer_dir / config.tokenizer_filename


def _iter_training_texts(training_examples: Iterable[DBpediaExample]):
    for example in training_examples:
        yield build_input_text(example.title, example.content)


def _with_classification_post_processor(tokenizer: Tokenizer) -> Tokenizer:
    cls_token_id = tokenizer.token_to_id("[CLS]")
    sep_token_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[
            ("[CLS]", cls_token_id),
            ("[SEP]", sep_token_id),
        ],
    )
    return tokenizer


def build_or_load_text_tokenizer(
    config,
    training_examples: Sequence[DBpediaExample],
    force_rebuild: bool = False,
) -> Tokenizer:
    """Build or load one tokenizer for DBpedia title/content inputs."""

    tokenizer_path = get_tokenizer_path(config)
    if tokenizer_path.exists() and not force_rebuild:
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=config.vocab_size,
        min_frequency=config.min_token_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(_iter_training_texts(training_examples), trainer=trainer)
    _with_classification_post_processor(tokenizer)
    tokenizer.save(str(tokenizer_path))

    return tokenizer


def _pad_or_truncate(
    input_ids: list[int],
    context_size: int,
    pad_token_id: int,
    sep_token_id: int,
) -> dict[str, list[int]]:
    input_ids = input_ids[:context_size]
    if len(input_ids) == context_size:
        input_ids[-1] = sep_token_id

    attention_mask = [1] * len(input_ids)
    padding_size = context_size - len(input_ids)
    if padding_size > 0:
        input_ids.extend([pad_token_id] * padding_size)
        attention_mask.extend([0] * padding_size)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def encode_text_for_classification(
    tokenizer,
    title: str,
    content: str,
    context_size: int
) -> dict[str, list[int]]:
    """Encode title/content text for the classifier encoder."""

    special_token_ids = get_special_token_ids(tokenizer)
    encoded = tokenizer.encode(build_input_text(title, content))
    return _pad_or_truncate(
        input_ids=encoded.ids,
        context_size=context_size,
        pad_token_id=special_token_ids["pad"],
        sep_token_id=special_token_ids["sep"],
    )


def encode_batch_for_classification(
    tokenizer,
    titles: Sequence[str],
    contents: Sequence[str],
    context_size: int,
) -> dict[str, list[list[int]]]:
    """Encode a batch of title/content rows for the classifier encoder."""

    if len(titles) != len(contents):
        raise ValueError("titles and contents must have the same number of items")

    special_token_ids = get_special_token_ids(tokenizer)
    texts = [build_input_text(title, content) for title, content in zip(titles, contents)]
    encodings = tokenizer.encode_batch(texts)
    padded = [
        _pad_or_truncate(
            input_ids=encoding.ids,
            context_size=context_size,
            pad_token_id=special_token_ids["pad"],
            sep_token_id=special_token_ids["sep"],
        )
        for encoding in encodings
    ]

    return {
        "input_ids": [item["input_ids"] for item in padded],
        "attention_mask": [item["attention_mask"] for item in padded],
    }


def get_special_token_ids(tokenizer) -> dict[str, int | None]:
    """Return special token IDs needed by datasets, masks, and pooling."""

    token_ids = {
        "unk": tokenizer.token_to_id("[UNK]"),
        "pad": tokenizer.token_to_id("[PAD]"),
        "cls": tokenizer.token_to_id("[CLS]"),
        "sep": tokenizer.token_to_id("[SEP]"),
    }

    missing_tokens = [name for name, token_id in token_ids.items() if token_id is None]
    if missing_tokens:
        raise ValueError(f"Tokenizer is missing special token IDs: {missing_tokens}")

    return token_ids
