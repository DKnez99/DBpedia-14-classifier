"""Report data helpers for DBpedia classifier training runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


MODEL_RUNS_FILENAME = "model_runs.json"


def best_checkpoint_path(config) -> Path:
    """Return the configured or architecture-derived best checkpoint path."""

    filename = config.best_checkpoint_filename or f"best_{infer_model_slug(config)}.pt"
    return Path(config.checkpoint_dir) / filename


def latest_checkpoint_path(config) -> Path:
    """Return the configured or architecture-derived latest checkpoint path."""

    filename = config.latest_checkpoint_filename or f"latest_{infer_model_slug(config)}.pt"
    return Path(config.checkpoint_dir) / filename


def infer_model_name(config) -> str:
    """Return the display name used in report tables and plots."""

    return _architecture_name(config)


def infer_model_slug(config) -> str:
    """Return a filesystem-safe architecture label for generated artifacts."""

    pooling = str(config.pooling).lower()
    return (
        f"{pooling}_"
        f"d{config.model_dimension}_"
        f"l{config.num_layers}_"
        f"h{config.num_heads}_"
        f"ff{config.feed_forward_dimension}"
    )


def record_training_epoch(
    config,
    epoch: int,
    global_step: int,
    train_loss: float,
    metrics: dict[str, Any],
    device: str,
) -> None:
    """Upsert one epoch into model_runs.json."""

    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_runs = load_model_runs(report_dir / MODEL_RUNS_FILENAME)
    model_name = infer_model_name(config)

    model = _get_or_create_model(model_runs, model_name)
    model["name"] = model_name
    model["config_source"] = "train.py"
    model["config"] = _config_to_report_dict(config, device)

    history_row = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "train_loss": float(train_loss),
        "validation_loss": _optional_float(metrics.get("loss")),
        "validation_accuracy": _optional_float(metrics.get("accuracy")),
        "validation_macro_precision": _optional_float(metrics.get("macro_precision")),
        "validation_macro_recall": _optional_float(metrics.get("macro_recall")),
        "validation_macro_f1": _optional_float(metrics.get("macro_f1")),
    }

    history = [
        row for row in model.get("train_history", []) if int(row["epoch"]) != epoch
    ]
    history.append(history_row)
    history.sort(key=lambda row: int(row["epoch"]))
    model["train_history"] = history
    _refresh_model_summary(model)

    save_model_runs(report_dir / MODEL_RUNS_FILENAME, model_runs)
    write_model_csvs(model_runs, report_dir)


def record_test_metrics(
    config,
    metrics: dict[str, Any],
    rows: int,
    predictions_path: str | Path | None,
    device: str,
) -> None:
    """Upsert labeled test metrics into model_runs.json and derived test CSVs."""

    report_dir = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_runs = load_model_runs(report_dir / MODEL_RUNS_FILENAME)
    model_name = infer_model_name(config)

    model = _get_or_create_model(model_runs, model_name)
    model["name"] = model_name
    model.setdefault("config_source", "infer.py")
    model["config"] = _config_to_report_dict(config, device)
    model["test"] = {
        "predictions_path": str(predictions_path) if predictions_path else None,
        "rows": int(rows),
        "accuracy": _optional_float(metrics.get("accuracy")),
        "macro_precision": _optional_float(metrics.get("macro_precision")),
        "macro_recall": _optional_float(metrics.get("macro_recall")),
        "macro_f1": _optional_float(metrics.get("macro_f1")),
    }
    _refresh_model_summary(model)

    save_model_runs(report_dir / MODEL_RUNS_FILENAME, model_runs)
    write_model_csvs(model_runs, report_dir)


def refresh_report_assets(config, quiet: bool = True) -> bool:
    """Regenerate report JSON, CSV, and plot assets from all known model runs."""

    try:
        from scripts.generate_report_assets import generate_report_assets

        _, plots = generate_report_assets(
            config=config,
            model_runs_path=Path(config.report_dir) / MODEL_RUNS_FILENAME,
            output_dir=Path(config.report_dir),
        )
    except Exception as error:
        print(f"Could not refresh report assets: {error}")
        return False

    if not quiet:
        print(f"Refreshed report assets: {len(plots)} plots")
    return True


def load_model_runs(path: Path) -> dict[str, Any]:
    """Load model run metadata, with compatibility for the old experiment file."""

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        legacy_path = path.with_name("experiment_results.json")
        if legacy_path.exists():
            data = json.loads(legacy_path.read_text(encoding="utf-8"))
        else:
            data = {"notes": [], "models": []}

    if "models" not in data and "experiments" in data:
        data["models"] = data.pop("experiments")
    for model in data.get("models", []):
        if "experiment" in model:
            model.pop("experiment")
        model["name"] = _model_name_from_record(model)
        model.pop("id", None)
        _refresh_model_summary(model)
    return data


def save_model_runs(path: Path, model_runs: dict[str, Any]) -> None:
    """Write model run metadata as stable, readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model_runs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_model_csvs(model_runs: dict[str, Any], output_dir: Path) -> dict[str, str]:
    """Write report test CSVs derived directly from model_runs.json."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = []

    for model in model_runs.get("models", []):
        model_name = model["name"]
        if model.get("test"):
            test = model["test"]
            test_rows.append(
                {
                    "model_name": model_name,
                    "best_validation_epoch": model.get("best_validation_epoch"),
                    "best_validation_accuracy": model.get("best_validation_accuracy"),
                    "latest_epoch": model.get("latest_epoch"),
                    "latest_validation_accuracy": model.get(
                        "latest_validation_accuracy"
                    ),
                    "test_rows": test.get("rows"),
                    "test_accuracy": test.get("accuracy"),
                    "test_macro_precision": test.get("macro_precision"),
                    "test_macro_recall": test.get("macro_recall"),
                    "test_macro_f1": test.get("macro_f1"),
                    "predictions_path": test.get("predictions_path"),
                }
            )

    outputs = {}
    if test_rows:
        path = output_dir / "test_metrics.csv"
        _write_csv(path, test_rows)
        outputs["test_metrics_csv"] = str(path)
    return outputs


def _architecture_name(config) -> str:
    pooling = "CLS" if str(config.pooling).lower() == "cls" else "Mean"
    return (
        f"{pooling} Pooling, "
        f"{config.num_layers}L, "
        f"d_model={config.model_dimension}, "
        f"{config.num_heads}H, "
        f"d_ff={config.feed_forward_dimension}"
    )


def _model_name_from_record(model: dict[str, Any]) -> str:
    config = model.get("config") or {}
    required_keys = {
        "pooling",
        "num_layers",
        "model_dimension",
        "num_heads",
        "feed_forward_dimension",
    }
    if required_keys.issubset(config):
        pooling = "CLS" if str(config["pooling"]).lower() == "cls" else "Mean"
        return (
            f"{pooling} Pooling, "
            f"{config['num_layers']}L, "
            f"d_model={config['model_dimension']}, "
            f"{config['num_heads']}H, "
            f"d_ff={config['feed_forward_dimension']}"
        )

    legacy_name = model.get("name")
    if legacy_name in {"CLS pooling, manji model", "CLS Small"}:
        return "CLS Pooling, 2L, d_model=128, 4H, d_ff=512"
    if legacy_name in {"Mean pooling, veci model", "Mean Large"}:
        return "Mean Pooling, 4L, d_model=256, 8H, d_ff=1024"
    if legacy_name:
        return legacy_name
    raise ValueError("model record is missing both config and name")


def _config_to_report_dict(config, device: str) -> dict[str, Any]:
    values = asdict(config)
    report_keys = [
        "text_context_size",
        "vocab_size",
        "min_token_frequency",
        "model_dimension",
        "num_layers",
        "num_heads",
        "feed_forward_dimension",
        "dropout",
        "pooling",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "weight_decay",
        "max_grad_norm",
        "validation_fraction",
        "seed",
    ]
    report = {key: values[key] for key in report_keys}
    report["optimizer"] = "AdamW"
    report["loss_function"] = "CrossEntropyLoss"
    report["device"] = str(device)
    return report


def _get_or_create_model(
    model_runs: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    models = model_runs.setdefault("models", [])
    for model in models:
        if model.get("name") == model_name:
            return model
    model = {
        "name": model_name,
        "config_source": "train.py",
        "config": {},
        "train_history": [],
    }
    models.append(model)
    return model


def _refresh_model_summary(model: dict[str, Any]) -> None:
    history = model.get("train_history", [])
    if not history:
        return

    latest = max(history, key=lambda row: int(row["epoch"]))
    model["latest_epoch"] = int(latest["epoch"])
    model["latest_validation_accuracy"] = _optional_float(
        latest.get("validation_accuracy")
    )
    model["latest_validation_loss"] = _optional_float(latest.get("validation_loss"))

    comparable = [
        row for row in history if row.get("validation_accuracy") is not None
    ]
    if comparable:
        best = max(comparable, key=lambda row: float(row["validation_accuracy"]))
        model["best_validation_epoch"] = int(best["epoch"])
        model["best_validation_accuracy"] = _optional_float(
            best.get("validation_accuracy")
        )
        model["best_validation_loss"] = _optional_float(best.get("validation_loss"))


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
