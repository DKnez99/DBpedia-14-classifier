"""Validation and evaluation utilities for DBpedia classification."""

from typing import Any

import torch
from tqdm import tqdm

from utils import resolve_device


def predict_batch(model, input_ids, attention_mask):
    """Predict class logits for one encoded batch."""

    return model(input_ids=input_ids, attention_mask=attention_mask)


def classification_metrics(
    predictions,
    labels,
    num_classes: int,
    class_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Calculate classification metrics for predicted and expected labels."""

    predictions = torch.as_tensor(predictions, dtype=torch.long).view(-1)
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1)

    if predictions.numel() != labels.numel():
        raise ValueError("predictions and labels must have the same number of items")

    if predictions.numel() == 0:
        raise ValueError("cannot calculate metrics for an empty prediction set")

    valid_mask = (
        (labels >= 0)
        & (labels < num_classes)
        & (predictions >= 0)
        & (predictions < num_classes)
    )
    if not valid_mask.all():
        raise ValueError("predictions and labels must be valid class IDs")

    confusion_indices = labels * num_classes + predictions
    confusion_matrix = torch.bincount(
        confusion_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)

    true_positive = confusion_matrix.diag().float()
    predicted_positive = confusion_matrix.sum(dim=0).float()
    actual_positive = confusion_matrix.sum(dim=1).float()

    precision = true_positive / predicted_positive.clamp_min(1.0)
    recall = true_positive / actual_positive.clamp_min(1.0)
    f1 = (2 * precision * recall) / (precision + recall).clamp_min(1e-12)

    accuracy = (predictions == labels).float().mean().item()
    macro_precision = precision.mean().item()
    macro_recall = recall.mean().item()
    macro_f1 = f1.mean().item()

    per_class = []
    for class_index in range(num_classes):
        class_name = class_names[class_index] if class_names else str(class_index)
        per_class.append(
            {
                "label": class_index,
                "name": class_name,
                "precision": precision[class_index].item(),
                "recall": recall[class_index].item(),
                "f1": f1[class_index].item(),
                "support": int(actual_positive[class_index].item()),
            }
        )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix,
    }


def validate(
    model,
    dataloader,
    config,
    device: torch.device | str | None = None,
    loss_function=None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Run validation on a DBpedia classifier checkpoint."""

    if device is None:
        device = next(model.parameters()).device
    device = resolve_device(device)

    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    total_examples = 0

    iterator = tqdm(dataloader, desc="Validating", leave=False) if show_progress else dataloader
    with torch.no_grad():
        for batch in iterator:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = predict_batch(model, input_ids, attention_mask)

            if loss_function is not None:
                loss = loss_function(logits, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_examples += batch_size

            all_predictions.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

    if not all_predictions:
        raise ValueError("cannot validate with an empty dataloader")

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    metrics = classification_metrics(
        predictions=predictions,
        labels=labels,
        num_classes=config.num_classes,
        class_names=getattr(config, "class_names", None),
    )

    if loss_function is not None:
        metrics["loss"] = total_loss / max(total_examples, 1)

    return metrics
