"""Training entry point for DBpedia classification."""

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from data import (
    DBpediaDataset,
    DBpediaExample,
    create_collate_fn,
    load_parquet_split,
    split_train_validation,
)
from evaluate import validate
from model import build_classifier
from reporting import (
    best_checkpoint_path,
    latest_checkpoint_path,
    record_training_epoch,
    refresh_report_assets,
)
from tokenizer import build_or_load_text_tokenizer, get_special_token_ids, get_tokenizer_path
from utils import (
    compact_metrics,
    config_to_checkpoint_dict,
    configure_torch_runtime,
    resolve_device,
    set_seed,
    should_pin_memory,
)


def _save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    config,
    vocab_size: int,
    pad_token_id: int,
    batch_index: int | None = None,
    best_accuracy: float | None = None,
    best_metrics: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "batch_index": batch_index,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metrics": compact_metrics(metrics),
            "best_accuracy": best_accuracy,
            "best_metrics": compact_metrics(best_metrics),
            "config": config_to_checkpoint_dict(config),
            "vocab_size": vocab_size,
            "pad_token_id": pad_token_id,
            "tokenizer_path": str(get_tokenizer_path(config)),
        },
        path,
    )


def _load_training_checkpoint(path: Path, model, optimizer, device: torch.device):
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def _latest_checkpoint_path(config) -> Path:
    return latest_checkpoint_path(config)


def _epoch_ordered_examples(
    examples: Sequence[DBpediaExample],
    epoch: int,
    seed: int,
) -> list[DBpediaExample]:
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    indices = torch.randperm(len(examples), generator=generator).tolist()
    return [examples[index] for index in indices]


def create_train_dataloader(
    config,
    train_examples: Sequence[DBpediaExample],
    tokenizer,
    epoch: int,
    start_batch_index: int = 0,
) -> DataLoader:
    """Create a deterministic train dataloader for one epoch."""

    ordered_examples = _epoch_ordered_examples(
        train_examples,
        epoch=epoch,
        seed=config.seed,
    )
    if start_batch_index > 0:
        ordered_examples = ordered_examples[start_batch_index * config.batch_size :]

    return DataLoader(
        DBpediaDataset(ordered_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer, context_size=config.text_context_size),
        num_workers=config.num_workers,
        pin_memory=should_pin_memory(config),
    )


def create_validation_dataloader(
    config,
    validation_examples: Sequence[DBpediaExample],
    tokenizer,
) -> DataLoader:
    """Create a validation dataloader."""

    return DataLoader(
        DBpediaDataset(validation_examples),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=create_collate_fn(tokenizer, context_size=config.text_context_size),
        num_workers=config.num_workers,
        pin_memory=should_pin_memory(config),
    )


def train(
    config,
    train_examples: Sequence[DBpediaExample] | None = None,
    validation_examples: Sequence[DBpediaExample] | None = None,
    force_rebuild_tokenizer: bool = False,
    resume_from_checkpoint: Path | str | None = None,
    refresh_reports: bool = True,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Train the DBpedia Transformer classifier."""

    configure_torch_runtime(config)
    set_seed(config.seed)
    device = resolve_device(config.device)
    print(
        f"Using device: {device}; "
        f"torch_threads={torch.get_num_threads()}; "
        f"interop_threads={torch.get_num_interop_threads()}; "
        f"cpu_count={os.cpu_count()}"
    )

    if train_examples is None or validation_examples is None:
        examples = load_parquet_split(config.train_file)
        train_examples, validation_examples = split_train_validation(
            examples,
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )

    tokenizer = build_or_load_text_tokenizer(
        config,
        train_examples,
        force_rebuild=force_rebuild_tokenizer,
    )
    special_token_ids = get_special_token_ids(tokenizer)
    pad_token_id = special_token_ids["pad"]

    validation_dataloader = create_validation_dataloader(
        config,
        validation_examples=validation_examples,
        tokenizer=tokenizer,
    )

    model = build_classifier(
        config,
        vocab_size=tokenizer.get_vocab_size(),
        pad_token_id=pad_token_id,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_accuracy = -1.0
    best_metrics: dict[str, Any] | None = None
    latest_metrics: dict[str, Any] = {}
    global_step = 0
    first_epoch = 1
    first_batch_index = 0

    resume_path = resume_from_checkpoint or config.resume_from_checkpoint
    if resume_path is not None:
        checkpoint = _load_training_checkpoint(
            Path(resume_path),
            model=model,
            optimizer=optimizer,
            device=device,
        )
        checkpoint_epoch = int(checkpoint["epoch"])
        checkpoint_batch_index = checkpoint.get("batch_index")
        first_epoch = checkpoint_epoch if checkpoint_batch_index is not None else checkpoint_epoch + 1
        first_batch_index = int(checkpoint_batch_index or 0)
        global_step = int(checkpoint.get("global_step", 0))
        previous_metrics = checkpoint.get("validation_metrics") or {}
        best_accuracy = float(checkpoint.get("best_accuracy", previous_metrics.get("accuracy", best_accuracy)))
        checkpoint_best_metrics = checkpoint.get("best_metrics") or previous_metrics
        best_metrics = checkpoint_best_metrics or None
        latest_metrics = previous_metrics
        if checkpoint_batch_index is None:
            print(
                f"Resumed from {resume_path} after epoch {checkpoint_epoch} "
                f"and global step {global_step}."
            )
        else:
            print(
                f"Resumed from {resume_path} saved during epoch {checkpoint_epoch}, "
                f"batch {checkpoint_batch_index}, global step {global_step}. "
                "Training will continue with the next batch in that epoch."
            )

    current_epoch = first_epoch - 1
    current_batch_index = None
    try:
        for epoch in range(first_epoch, config.num_epochs + 1):
            current_epoch = epoch
            start_batch_index = first_batch_index if epoch == first_epoch else 0
            first_batch_index = 0
            train_dataloader = create_train_dataloader(
                config,
                train_examples=train_examples,
                tokenizer=tokenizer,
                epoch=epoch,
                start_batch_index=start_batch_index,
            )
            model.train()
            total_loss = 0.0
            total_examples = 0

            iterator = (
                tqdm(train_dataloader, desc=f"Epoch {epoch}/{config.num_epochs}")
                if show_progress
                else train_dataloader
            )
            for batch_index, batch in enumerate(iterator, start=start_batch_index + 1):
                current_batch_index = batch_index
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_function(logits, labels)
                loss.backward()
                if config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size
                global_step += 1

                if (
                    config.checkpoint_every_steps > 0
                    and global_step % config.checkpoint_every_steps == 0
                ):
                    _save_checkpoint(
                        _latest_checkpoint_path(config),
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        global_step=global_step,
                        metrics=latest_metrics,
                        config=config,
                        vocab_size=tokenizer.get_vocab_size(),
                        pad_token_id=pad_token_id,
                        batch_index=batch_index,
                        best_accuracy=best_accuracy,
                        best_metrics=best_metrics,
                    )

                if show_progress:
                    iterator.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = total_loss / max(total_examples, 1)
            metrics = validate(
                model,
                validation_dataloader,
                config,
                device=device,
                loss_function=loss_function,
                show_progress=show_progress,
            )
            metrics["train_loss"] = train_loss
            latest_metrics = metrics

            is_best_epoch = metrics["accuracy"] > best_accuracy
            if is_best_epoch:
                best_accuracy = metrics["accuracy"]
                best_metrics = metrics
                best_path = best_checkpoint_path(config)
                _save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=metrics,
                    config=config,
                    vocab_size=tokenizer.get_vocab_size(),
                    pad_token_id=pad_token_id,
                    best_accuracy=best_accuracy,
                    best_metrics=best_metrics,
                )

            record_training_epoch(
                config=config,
                epoch=epoch,
                global_step=global_step,
                train_loss=train_loss,
                metrics=metrics,
                device=str(device),
            )

            latest_path = _latest_checkpoint_path(config)
            _save_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                metrics=metrics,
                config=config,
                vocab_size=tokenizer.get_vocab_size(),
                pad_token_id=pad_token_id,
                best_accuracy=best_accuracy,
                best_metrics=best_metrics,
            )
            if refresh_reports:
                refresh_report_assets(config, quiet=show_progress)
            current_batch_index = None

            print(
                "Epoch "
                f"{epoch}: train_loss={train_loss:.4f}, "
                f"validation_loss={metrics.get('loss', 0.0):.4f}, "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['macro_f1']:.4f}"
            )
    except KeyboardInterrupt:
        if config.save_on_interrupt:
            interrupt_path = _latest_checkpoint_path(config)
            _save_checkpoint(
                interrupt_path,
                model=model,
                optimizer=optimizer,
                epoch=current_epoch,
                global_step=global_step,
                metrics=latest_metrics,
                config=config,
                vocab_size=tokenizer.get_vocab_size(),
                pad_token_id=pad_token_id,
                batch_index=current_batch_index,
                best_accuracy=best_accuracy,
                best_metrics=best_metrics,
            )
            print(f"\nInterrupted. Saved checkpoint to {interrupt_path}.")
        else:
            print("\nInterrupted before training completed.")
        return {
            "interrupted": True,
            "latest_checkpoint": _latest_checkpoint_path(config),
            "best_checkpoint": best_checkpoint_path(config),
        }

    return {
        "best_accuracy": best_accuracy,
        "best_metrics": best_metrics,
        "latest_checkpoint": _latest_checkpoint_path(config),
        "best_checkpoint": best_checkpoint_path(config),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DBpedia classifier.")
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume model and optimizer state from a checkpoint.",
    )
    parser.add_argument(
        "--force-rebuild-tokenizer",
        action="store_true",
        help="Rebuild the tokenizer before training.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Training device override: auto, cpu, cuda, or mps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes. Start with 0 or 2 on macOS.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=None,
        help="Number of CPU compute threads PyTorch should use.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=None,
        help="Number of PyTorch inter-op CPU threads.",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=None,
        help="Update the latest checkpoint every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-save-on-interrupt",
        action="store_true",
        help="Do not update the latest checkpoint when training is stopped with Ctrl+C.",
    )
    parser.add_argument(
        "--no-report-refresh",
        action="store_true",
        help="Do not regenerate report assets after each completed epoch.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = get_config()
    overrides = {}
    if args.device is not None:
        overrides["device"] = args.device
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.torch_threads is not None:
        overrides["torch_num_threads"] = args.torch_threads
    if args.torch_interop_threads is not None:
        overrides["torch_num_interop_threads"] = args.torch_interop_threads
    if args.checkpoint_every_steps is not None:
        overrides["checkpoint_every_steps"] = args.checkpoint_every_steps
    if args.no_save_on_interrupt:
        overrides["save_on_interrupt"] = False
    if overrides:
        config = replace(config, **overrides)

    train(
        config,
        force_rebuild_tokenizer=args.force_rebuild_tokenizer,
        resume_from_checkpoint=args.resume_from_checkpoint,
        refresh_reports=not args.no_report_refresh,
        show_progress=not args.no_progress,
    )
