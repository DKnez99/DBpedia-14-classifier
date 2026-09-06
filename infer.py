"""Inference helpers for DBpedia title/content classification."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm
from tokenizers import Tokenizer

from config import DBPEDIA_LABELS, ProjectConfig, get_config
from evaluate import classification_metrics
from model import build_classifier
from reporting import (
    best_checkpoint_path,
    infer_model_slug,
    record_test_metrics,
    refresh_report_assets,
)
from tokenizer import (
    encode_batch_for_classification,
    encode_text_for_classification,
    get_tokenizer_path,
)
from utils import config_from_checkpoint, resolve_device


PRINT_MODES = ("summary", "compact", "all", "wrong")


def _checkpoint_path(config: ProjectConfig, checkpoint_path: Path | str | None) -> Path:
    if checkpoint_path is not None:
        return Path(checkpoint_path)
    return best_checkpoint_path(config)


def load_model_for_inference(
    checkpoint_path: Path | str | None = None,
    config: ProjectConfig | None = None,
    device: torch.device | str | None = None,
):
    """Load tokenizer and model from a DBpedia classifier checkpoint."""

    fallback_config = config or get_config()
    resolved_checkpoint_path = _checkpoint_path(fallback_config, checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_checkpoint_path}")

    resolved_device = resolve_device(device or fallback_config.device)
    checkpoint = torch.load(resolved_checkpoint_path, map_location=resolved_device)
    checkpoint_config = config_from_checkpoint(checkpoint, fallback=fallback_config)

    tokenizer_path = Path(
        checkpoint.get(
            "tokenizer_path",
            get_tokenizer_path(checkpoint_config),
        )
    )
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = build_classifier(
        checkpoint_config,
        vocab_size=int(checkpoint.get("vocab_size", tokenizer.get_vocab_size())),
        pad_token_id=int(checkpoint.get("pad_token_id", tokenizer.token_to_id("[PAD]"))),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return model, tokenizer, checkpoint_config, resolved_device, checkpoint


class DBpediaClassifierPredictor:
    """Reusable DBpedia classifier inference wrapper."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        config: ProjectConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        (
            self.model,
            self.tokenizer,
            self.config,
            self.device,
            self.checkpoint,
        ) = load_model_for_inference(
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
        )

    def predict(self, title: str, content: str) -> dict[str, Any]:
        """Predict a DBpedia category and confidence for one title/content pair."""

        encoded = encode_text_for_classification(
            self.tokenizer,
            title=title,
            content=content,
            context_size=self.config.text_context_size,
        )
        input_ids = torch.tensor(
            [encoded["input_ids"]],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            [encoded["attention_mask"]],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

        label_id = int(probabilities.argmax().item())
        class_names = getattr(self.config, "class_names", DBPEDIA_LABELS)
        return {
            "label": label_id,
            "category": class_names[label_id],
            "confidence": float(probabilities[label_id].item()),
            "probabilities": probabilities.tolist(),
        }

    def categorize(self, title: str, content: str) -> str:
        """Return only the predicted DBpedia category name."""

        return self.predict(title, content)["category"]

    def predict_many(
        self,
        titles: list[str] | tuple[str, ...],
        contents: list[str] | tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Predict DBpedia categories for a batch of title/content pairs."""

        encoded = encode_batch_for_classification(
            self.tokenizer,
            titles=titles,
            contents=contents,
            context_size=self.config.text_context_size,
        )
        input_ids = torch.tensor(
            encoded["input_ids"],
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.tensor(
            encoded["attention_mask"],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(logits, dim=-1).cpu()

        class_names = getattr(self.config, "class_names", DBPEDIA_LABELS)
        predictions = []
        for row_probabilities in probabilities:
            label_id = int(row_probabilities.argmax().item())
            predictions.append(
                {
                    "label": label_id,
                    "category": class_names[label_id],
                    "confidence": float(row_probabilities[label_id].item()),
                    "probabilities": row_probabilities.tolist(),
                }
            )

        return predictions

    def predict_parquet(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        batch_size: int | None = None,
        limit: int | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Predict categories for every row in a parquet file."""

        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input parquet not found: {input_path}")

        dataframe = pd.read_parquet(input_path)
        required_columns = {"title", "content"}
        missing_columns = required_columns - set(dataframe.columns)
        if missing_columns:
            raise ValueError(f"Input parquet is missing columns: {sorted(missing_columns)}")
        if limit is not None:
            dataframe = dataframe.head(limit)

        effective_batch_size = batch_size or self.config.batch_size
        output_rows = []
        iterator = range(0, len(dataframe), effective_batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")

        for start in iterator:
            batch = dataframe.iloc[start : start + effective_batch_size]
            predictions = self.predict_many(
                titles=batch["title"].astype(str).tolist(),
                contents=batch["content"].astype(str).tolist(),
            )
            for row, prediction in zip(batch.itertuples(index=False), predictions):
                output_row = {
                    "title": row.title,
                    "content": row.content,
                    "predicted_label": prediction["label"],
                    "predicted_category": prediction["category"],
                    "confidence": prediction["confidence"],
                }
                if "label" in dataframe.columns:
                    output_row["true_label"] = int(row.label)
                    output_row["correct"] = output_row["true_label"] == output_row["predicted_label"]
                output_rows.append(output_row)

        predictions_df = pd.DataFrame(output_rows)
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions_df.to_parquet(output_path, index=False)

        return predictions_df


def _label_name(class_names: list[str] | tuple[str, ...], label: int) -> str:
    if 0 <= label < len(class_names):
        return class_names[label]
    return str(label)


def print_prediction_rows(
    predictions: pd.DataFrame,
    class_names: list[str] | tuple[str, ...],
    mode: str,
) -> None:
    """Print compact or detailed batch prediction rows."""

    if mode not in PRINT_MODES:
        raise ValueError(f"mode must be one of {PRINT_MODES}")
    if mode == "summary" or predictions.empty:
        return

    rows = predictions
    if mode == "wrong" and "correct" in predictions.columns:
        rows = predictions[predictions["correct"] == False]

    for index, row in rows.iterrows():
        predicted = f"{int(row.predicted_label)}:{row.predicted_category}"
        confidence = f"{float(row.confidence):.4f}"
        if mode == "compact":
            print(f"{index}\t{predicted}\tconfidence={confidence}")
            continue

        print(f"[{index}] predicted={predicted} confidence={confidence}")
        if "true_label" in row and not bool(row.correct):
            true_label = int(row.true_label)
            true_name = _label_name(class_names, true_label)
            print(f"true={true_label}:{true_name}")
        print(f"title={row.title}")
        print(f"content={row.content}")
        print()


def print_prediction_summary(predictions: pd.DataFrame) -> None:
    """Print aggregate prediction counts and accuracy when labels are available."""

    total_rows = len(predictions)
    if "correct" in predictions.columns:
        correct_rows = int(predictions["correct"].sum())
        incorrect_rows = total_rows - correct_rows
        accuracy = correct_rows / total_rows if total_rows else 0.0
        print(f"accuracy={accuracy:.6f}")
        print(f"correct_rows={correct_rows}")
        print(f"incorrect_rows={incorrect_rows}")
    print(f"predicted_rows={total_rows}")


def predict_category(
    title: str,
    content: str,
    checkpoint_path: Path | str | None = None,
    config: ProjectConfig | None = None,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Predict a DBpedia category and confidence for one title/content pair."""

    predictor = DBpediaClassifierPredictor(
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )
    return predictor.predict(title, content)


def categorize(
    title: str,
    content: str,
    checkpoint_path: Path | str | None = None,
    config: ProjectConfig | None = None,
    device: torch.device | str | None = None,
) -> str:
    """Categorize natural language title and content into a DBpedia class."""

    return predict_category(
        title,
        content,
        checkpoint_path=checkpoint_path,
        config=config,
        device=device,
    )["category"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DBpedia category inference.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--title", help="Article title for single-example inference.")
    input_group.add_argument("--input-parquet", help="Parquet file with title/content columns.")
    parser.add_argument("--content", help="Article content for single-example inference.")
    parser.add_argument(
        "--output-parquet",
        help=(
            "Optional path for batch prediction output. Full parquet runs default "
            "to runs/test_<model_slug>.parquet."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for parquet inference.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max parquet rows to predict.")
    parser.add_argument(
        "--print-mode",
        choices=PRINT_MODES,
        default="summary",
        help="Batch output: summary only, compact labels, all full rows, or wrong full rows only.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to checkpoints/best_<model_slug>.pt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional PyTorch device override, such as cpu, cuda, or mps.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--no-report-refresh",
        action="store_true",
        help="Do not regenerate report assets after full labeled batch inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    predictor = DBpediaClassifierPredictor(
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    if args.input_parquet:
        output_parquet = args.output_parquet
        if output_parquet is None and args.limit is None:
            output_parquet = (
                Path(predictor.config.run_dir)
                / f"test_{infer_model_slug(predictor.config)}.parquet"
            )

        predictions = predictor.predict_parquet(
            input_path=args.input_parquet,
            output_path=output_parquet,
            batch_size=args.batch_size,
            limit=args.limit,
            show_progress=not args.no_progress,
        )
        print_prediction_summary(predictions)
        print_prediction_rows(predictions, predictor.config.class_names, args.print_mode)
        if (
            args.limit is None
            and {"true_label", "predicted_label"}.issubset(predictions.columns)
        ):
            metrics = classification_metrics(
                predictions=predictions["predicted_label"],
                labels=predictions["true_label"],
                num_classes=predictor.config.num_classes,
                class_names=predictor.config.class_names,
            )
            record_test_metrics(
                config=predictor.config,
                metrics=metrics,
                rows=len(predictions),
                predictions_path=output_parquet,
                device=str(predictor.device),
            )
            if not args.no_report_refresh:
                refresh_report_assets(predictor.config, quiet=not args.no_progress)
        if output_parquet:
            print(f"output={output_parquet}")
    else:
        if args.content is None:
            raise SystemExit("--content is required with --title")
        sample = predictor.predict(title=args.title, content=args.content)
        print(
            f"{sample['category']} "
            f"(label={sample['label']}, confidence={sample['confidence']:.4f})"
        )
