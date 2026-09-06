"""Generate report tables and plots for the DBpedia classifier project."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from config import DBPEDIA_LABELS, get_config
from evaluate import classification_metrics
from reporting import (
    best_checkpoint_path,
    infer_model_slug,
    latest_checkpoint_path,
    load_model_runs,
    save_model_runs,
    write_model_csvs,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_project_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "model"


def _model_slug(model: dict) -> str:
    config = model.get("config") or {}
    required_keys = {
        "pooling",
        "num_layers",
        "model_dimension",
        "num_heads",
        "feed_forward_dimension",
    }
    if required_keys.issubset(config):
        return (
            f"{str(config['pooling']).lower()}_"
            f"d{config['model_dimension']}_"
            f"l{config['num_layers']}_"
            f"h{config['num_heads']}_"
            f"ff{config['feed_forward_dimension']}"
        )
    return _safe_slug(model["name"])


def _short_model_label(model_name: str, model_lookup: dict[str, dict]) -> str:
    config = (model_lookup.get(model_name) or {}).get("config") or {}
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
            f"{pooling} "
            f"{config['num_layers']}L "
            f"d{config['model_dimension']} "
            f"h{config['num_heads']} "
            f"ff{config['feed_forward_dimension']}"
        )
    return model_name


def _metric_ylim(values: pd.Series, padding: float = 0.002) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0
    lower = max(0.0, float(values.min()) - padding)
    upper = min(1.0, float(values.max()) + padding)
    if lower == upper:
        lower = max(0.0, lower - padding)
        upper = min(1.0, upper + padding)
    return lower, upper


def _metric_groups(
    dataframe: pd.DataFrame,
    columns: list[tuple[str, str]],
    tolerance: float = 1e-7,
) -> list[tuple[str, pd.Series]]:
    groups: list[tuple[list[str], pd.Series]] = []
    for column, label in columns:
        if column not in dataframe or not dataframe[column].notna().any():
            continue
        series = pd.to_numeric(dataframe[column], errors="coerce")
        matched = False
        for labels, representative in groups:
            comparable = pd.concat([representative, series], axis=1).dropna()
            if comparable.empty:
                continue
            if (comparable.iloc[:, 0] - comparable.iloc[:, 1]).abs().max() <= tolerance:
                labels.append(label)
                matched = True
                break
        if not matched:
            groups.append(([label], series))
    return [(" / ".join(labels), series) for labels, series in groups]


def _save_current_figure_to(paths: list[Path], dpi: int = 160) -> list[str]:
    import matplotlib.pyplot as plt

    saved_paths = []
    for path in paths:
        plt.savefig(path, dpi=dpi)
        saved_paths.append(str(path))
    plt.close()
    return saved_paths


def _load_checkpoint_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    return {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "vocab_size": checkpoint.get("vocab_size"),
        "pad_token_id": checkpoint.get("pad_token_id"),
        "validation_metrics": checkpoint.get("validation_metrics"),
        "best_accuracy": checkpoint.get("best_accuracy"),
        "best_metrics": checkpoint.get("best_metrics"),
        "config": checkpoint.get("config"),
    }


def _artifact_manifest(model_runs: dict, output_dir: Path) -> dict[str, object]:
    artifacts: dict[str, object] = {
        "dataset_distribution": {
            "csv": str(output_dir / "dataset_distribution.csv"),
            "png": str(output_dir / "dataset_distribution.png"),
        },
        "loss_curve": {
            "csv": str(output_dir / "loss_curve.csv"),
            "png": str(output_dir / "loss_curve.png"),
        },
        "validation_metrics_curve": {
            "csv": str(output_dir / "validation_metrics_curve.csv"),
            "png": str(output_dir / "validation_metrics_curve.png"),
        },
        "test_metrics": {
            "csv": str(output_dir / "test_metrics.csv"),
            "png": str(output_dir / "test_metrics.png"),
        },
        "test_per_class_f1": {
            "csv": str(output_dir / "test_per_class_f1.csv"),
            "png": str(output_dir / "test_per_class_f1.png"),
        },
    }
    by_model: dict[str, dict[str, dict[str, str]]] = {}
    for model in model_runs.get("models", []):
        model_name = model["name"]
        slug = _model_slug(model)
        artifacts_for_model: dict[str, dict[str, str]] = {}
        if model.get("train_history"):
            artifacts_for_model["loss_curve"] = {
                "csv": str(output_dir / f"loss_curve_{slug}.csv"),
                "png": str(output_dir / f"loss_curve_{slug}.png"),
            }
            artifacts_for_model["validation_metrics_curve"] = {
                "csv": str(output_dir / f"validation_metrics_curve_{slug}.csv"),
                "png": str(output_dir / f"validation_metrics_curve_{slug}.png"),
            }
        if model.get("test"):
            artifacts_for_model["test_metrics"] = {
                "csv": str(output_dir / f"test_metrics_{slug}.csv"),
                "png": str(output_dir / f"test_metrics_{slug}.png"),
            }
            artifacts_for_model["test_per_class_f1"] = {
                "csv": str(output_dir / f"test_per_class_f1_{slug}.csv"),
                "png": str(output_dir / f"test_per_class_f1_{slug}.png"),
            }
        if artifacts_for_model:
            by_model[model_name] = artifacts_for_model
    if by_model:
        artifacts["by_model"] = by_model
    return artifacts


def _history_frame_from_model_runs(model_runs: dict) -> pd.DataFrame:
    rows = []
    for model in model_runs.get("models", []):
        model_name = model["name"]
        for row in model.get("train_history", []):
            rows.append(
                {
                    "model_name": model_name,
                    **row,
                    "is_best_epoch": int(
                        row.get("epoch") == model.get("best_validation_epoch")
                    ),
                    "is_latest_epoch": int(
                        row.get("epoch") == model.get("latest_epoch")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _write_dataset_distribution(config, output_dir: Path) -> dict:
    rows = []
    train_dataframe = pd.read_parquet(config.train_file, columns=["label"])
    train_counts = train_dataframe["label"].value_counts().sort_index()
    for label, official_train_count in train_counts.items():
        validation_count = round(int(official_train_count) * config.validation_fraction)
        rows.append(
            {
                "split": "train",
                "label": int(label),
                "class_name": DBPEDIA_LABELS[int(label)],
                "count": int(official_train_count) - validation_count,
            }
        )
        rows.append(
            {
                "split": "validation",
                "label": int(label),
                "class_name": DBPEDIA_LABELS[int(label)],
                "count": validation_count,
            }
        )

    test_dataframe = pd.read_parquet(config.test_file, columns=["label"])
    test_counts = test_dataframe["label"].value_counts().sort_index()
    for label, count in test_counts.items():
        rows.append(
            {
                "split": "test",
                "label": int(label),
                "class_name": DBPEDIA_LABELS[int(label)],
                "count": int(count),
            }
        )
    distribution = pd.DataFrame(rows)
    distribution.to_csv(output_dir / "dataset_distribution.csv", index=False)
    return {
        "train_rows": int(distribution[distribution["split"] == "train"]["count"].sum()),
        "validation_rows": int(
            distribution[distribution["split"] == "validation"]["count"].sum()
        ),
        "test_rows": int(distribution[distribution["split"] == "test"]["count"].sum()),
        "csv": str(output_dir / "dataset_distribution.csv"),
    }


def _write_test_metrics(
    predictions_path: Path,
    output_dir: Path,
    model_slug: str = "current",
) -> dict | None:
    resolved_predictions_path = _resolve_project_path(predictions_path)
    if not resolved_predictions_path.exists():
        return None
    predictions = pd.read_parquet(resolved_predictions_path)
    required_columns = {"predicted_label", "true_label"}
    if not required_columns.issubset(predictions.columns):
        return None

    metrics = classification_metrics(
        predictions=predictions["predicted_label"],
        labels=predictions["true_label"],
        num_classes=len(DBPEDIA_LABELS),
        class_names=DBPEDIA_LABELS,
    )
    per_class = pd.DataFrame(metrics["per_class"])
    per_class_path = output_dir / f"test_per_class_f1_{model_slug}.csv"
    per_class.to_csv(per_class_path, index=False)
    return {
        "predictions_path": str(predictions_path),
        "rows": int(len(predictions)),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "test_per_class_f1_csv": str(per_class_path),
    }


def _write_environment(output_dir: Path) -> dict:
    mps_built = torch.backends.mps.is_built()
    mps_available = torch.backends.mps.is_available() or (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and mps_built
    )
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": torch.cuda.is_available(),
        "mps_built": mps_built,
        "mps_available": mps_available,
    }
    _write_json(output_dir / "environment.json", environment)
    return environment


def _generate_static_plots(output_dir: Path) -> list[str]:
    import matplotlib.pyplot as plt

    plots = []
    distribution_path = output_dir / "dataset_distribution.csv"
    if distribution_path.exists():
        distribution = pd.read_csv(distribution_path)
        pivot = distribution.pivot(index="class_name", columns="split", values="count")
        split_order = ["train", "validation", "test"]
        pivot = pivot[[column for column in split_order if column in pivot.columns]]
        pivot.plot(kind="bar", figsize=(11, 5))
        plt.xlabel("Class")
        plt.ylabel("Number of examples")
        plt.title("Class distribution by split")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = output_dir / "dataset_distribution.png"
        plt.savefig(path, dpi=160)
        plt.close()
        plots.append(str(path))

    return plots


def _write_model_tables(model_runs: dict, output_dir: Path) -> dict:
    outputs = {}
    per_class_rows = []
    per_model_test_csvs = {}
    per_model_per_class_csvs = {}

    for model in model_runs.get("models", []):
        model_name = model["name"]
        slug = _model_slug(model)
        test = model.get("test", {})
        if test:
            predictions_path_value = test.get("predictions_path")
            if predictions_path_value:
                predictions_path = _resolve_project_path(predictions_path_value)
            else:
                predictions_path = None
            if predictions_path and predictions_path.exists():
                predictions = pd.read_parquet(predictions_path)
                if {"predicted_label", "true_label"}.issubset(predictions.columns):
                    metrics = classification_metrics(
                        predictions=predictions["predicted_label"],
                        labels=predictions["true_label"],
                        num_classes=len(DBPEDIA_LABELS),
                        class_names=DBPEDIA_LABELS,
                    )
                    test.update(
                        {
                            "rows": int(len(predictions)),
                            "accuracy": metrics["accuracy"],
                            "macro_precision": metrics["macro_precision"],
                            "macro_recall": metrics["macro_recall"],
                            "macro_f1": metrics["macro_f1"],
                        }
                    )
                    model["test"] = test
                    model_test_path = output_dir / f"test_metrics_{slug}.csv"
                    pd.DataFrame(
                        [
                            {
                                "model_name": model_name,
                                "best_validation_epoch": model.get(
                                    "best_validation_epoch"
                                ),
                                "best_validation_accuracy": model.get(
                                    "best_validation_accuracy"
                                ),
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
                        ]
                    ).to_csv(model_test_path, index=False)
                    per_model_test_csvs[model_name] = str(model_test_path)
                    model_per_class_rows = []
                    for per_class in metrics["per_class"]:
                        row = {
                            "model_name": model_name,
                            **per_class,
                        }
                        per_class_rows.append(row)
                        model_per_class_rows.append(row)
                    model_per_class_path = output_dir / f"test_per_class_f1_{slug}.csv"
                    pd.DataFrame(model_per_class_rows).to_csv(
                        model_per_class_path,
                        index=False,
                    )
                    per_model_per_class_csvs[model_name] = str(model_per_class_path)

    if per_class_rows:
        path = output_dir / "test_per_class_f1.csv"
        pd.DataFrame(per_class_rows).to_csv(path, index=False)
        outputs["test_per_class_f1_csv"] = str(path)
        outputs["test_metrics_by_model_csvs"] = per_model_test_csvs
        outputs["test_per_class_f1_by_model_csvs"] = per_model_per_class_csvs

    outputs.update(write_model_csvs(model_runs, output_dir))
    if per_model_test_csvs:
        outputs["test_metrics_by_model_csvs"] = per_model_test_csvs
    if per_model_per_class_csvs:
        outputs["test_per_class_f1_by_model_csvs"] = per_model_per_class_csvs
    return outputs


def _generate_model_plots(output_dir: Path, model_runs: dict) -> list[str]:
    import matplotlib.pyplot as plt

    plots = []
    model_lookup = {model["name"]: model for model in model_runs.get("models", [])}
    model_slugs = {model["name"]: _model_slug(model) for model in model_runs.get("models", [])}
    history = _history_frame_from_model_runs(model_runs)
    if not history.empty:
        history = history.sort_values(["model_name", "epoch"])
        grouped_history = list(history.groupby("model_name", sort=False))

        if grouped_history:
            loss_columns = [
                "model_name",
                "epoch",
                "train_loss",
                "validation_loss",
                "is_best_epoch",
                "is_latest_epoch",
            ]
            history[[column for column in loss_columns if column in history]].to_csv(
                output_dir / "loss_curve.csv",
                index=False,
            )
            validation_columns = [
                "model_name",
                "epoch",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
                "is_best_epoch",
                "is_latest_epoch",
            ]
            history[
                [column for column in validation_columns if column in history]
            ].to_csv(output_dir / "validation_metrics_curve.csv", index=False)

            metric_columns = [
                ("validation_accuracy", "Accuracy"),
                ("validation_macro_f1", "Macro-F1"),
                ("validation_macro_precision", "Macro-precision"),
                ("validation_macro_recall", "Macro-recall"),
            ]
            fig, ax = plt.subplots(figsize=(13, 6))
            for model_name, group in grouped_history:
                label = _short_model_label(model_name, model_lookup)
                ax.plot(
                    group["epoch"],
                    group["train_loss"],
                    linestyle="--",
                    label=f"{label} train",
                )
                ax.plot(
                    group["epoch"],
                    group["validation_loss"],
                    label=f"{label} validation",
                )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss comparison")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            fig.tight_layout()
            plots.extend(_save_current_figure_to([output_dir / "loss_curve.png"]))

            fig, ax = plt.subplots(figsize=(13, 6))
            plotted_values = []
            comparison_metric_columns = [
                ("validation_accuracy", "Accuracy"),
                ("validation_macro_f1", "Macro-F1"),
            ]
            for model_name, group in grouped_history:
                model_label = _short_model_label(model_name, model_lookup)
                for metric_label, series in _metric_groups(
                    group,
                    comparison_metric_columns,
                ):
                    ax.plot(
                        group["epoch"],
                        series,
                        marker="o",
                        label=f"{model_label} {metric_label}",
                    )
                    plotted_values.append(series)
            if plotted_values:
                ax.set_ylim(*_metric_ylim(pd.concat(plotted_values)))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric value")
            ax.set_title("Validation metric comparison")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            fig.tight_layout()
            plots.extend(
                _save_current_figure_to([output_dir / "validation_metrics_curve.png"])
            )

        for model_name, group in grouped_history:
            label = _short_model_label(model_name, model_lookup)
            slug = model_slugs.get(model_name, _safe_slug(model_name))
            group[
                [column for column in loss_columns if column in group]
            ].to_csv(output_dir / f"loss_curve_{slug}.csv", index=False)
            group[
                [column for column in validation_columns if column in group]
            ].to_csv(
                output_dir / f"validation_metrics_curve_{slug}.csv",
                index=False,
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                group["epoch"],
                group["train_loss"],
                marker="o",
                linestyle="--",
                label="Training loss",
            )
            ax.plot(
                group["epoch"],
                group["validation_loss"],
                marker="o",
                label="Validation loss",
            )
            _mark_best_and_latest(
                ax,
                group,
                y_column="validation_loss",
                best_label="Best validation",
                latest_label="Last epoch",
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"Loss during training: {label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"loss_curve_{slug}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plots.append(str(path))

            fig, ax = plt.subplots(figsize=(8, 5))
            plotted_values = []
            for metric_label, series in _metric_groups(group, metric_columns):
                ax.plot(
                    group["epoch"],
                    series,
                    marker="o",
                    label=metric_label,
                )
                plotted_values.append(series)
            if plotted_values:
                ax.set_ylim(*_metric_ylim(pd.concat(plotted_values)))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Metric value")
            ax.set_title(f"Validation metrics: {label}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            path = output_dir / f"validation_metrics_curve_{slug}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            plots.append(str(path))

    test_path = output_dir / "test_metrics.csv"
    if test_path.exists():
        test_metrics = pd.read_csv(test_path)
        test_metrics["model_label"] = test_metrics["model_name"].map(
            lambda value: _short_model_label(value, model_lookup)
        )
        metric_columns = {
            "test_accuracy": "Accuracy",
            "test_macro_precision": "Macro-precision",
            "test_macro_recall": "Macro-recall",
            "test_macro_f1": "Macro-F1",
        }
        available_columns = [
            column for column in metric_columns if column in test_metrics.columns
        ]
        if available_columns:
            test_metric_groups = _metric_groups(
                test_metrics,
                [(column, metric_columns[column]) for column in available_columns],
            )
            plot_frame = pd.DataFrame(
                {
                    label: series.to_numpy()
                    for label, series in test_metric_groups
                },
                index=test_metrics["model_label"],
            )
            ax = plot_frame.plot(kind="bar", figsize=(11, 5))
            ax.set_xlabel("Model")
            ax.set_ylabel("Metric value")
            ax.set_title("Test metric comparison")
            ax.set_ylim(*_metric_ylim(plot_frame.stack()))
            ax.tick_params(axis="x", rotation=15)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
            plt.tight_layout()
            path = output_dir / "test_metrics.png"
            plt.savefig(path, dpi=160)
            plt.close()
            plots.append(str(path))

            for _, row in test_metrics.iterrows():
                model_name = row["model_name"]
                slug = model_slugs.get(model_name, _safe_slug(model_name))
                values = pd.DataFrame(
                    {
                        "metric": list(plot_frame.columns),
                        "value": [
                            plot_frame.loc[row["model_label"], column]
                            for column in plot_frame.columns
                        ],
                    }
                )
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(values["metric"], values["value"])
                ax.set_xlabel("Metric")
                ax.set_ylabel("Metric value")
                ax.set_title(
                    f"Test metrics: {_short_model_label(model_name, model_lookup)}"
                )
                ax.set_ylim(*_metric_ylim(values["value"]))
                ax.tick_params(axis="x", rotation=15)
                fig.tight_layout()
                path = output_dir / f"test_metrics_{slug}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                plots.append(str(path))

    per_class_path = output_dir / "test_per_class_f1.csv"
    if per_class_path.exists():
        per_class = pd.read_csv(per_class_path)
        per_class = per_class.sort_values(["model_name", "label"])
        per_class["model_label"] = per_class["model_name"].map(
            lambda value: _short_model_label(value, model_lookup)
        )
        grouped_per_class = list(per_class.groupby("model_name", sort=False))
        if grouped_per_class:
            for model_name, group in grouped_per_class:
                label = _short_model_label(model_name, model_lookup)
                slug = model_slugs.get(model_name, _safe_slug(model_name))
                fig, ax = plt.subplots(figsize=(11, 5))
                ax.bar(group["name"], group["f1"])
                ax.set_xlabel("Class")
                ax.set_ylabel("F1")
                ax.set_title(f"Test per-class F1: {label}")
                ax.set_ylim(*_metric_ylim(group["f1"], padding=0.004))
                ax.tick_params(axis="x", rotation=45)
                for tick in ax.get_xticklabels():
                    tick.set_horizontalalignment("right")
                fig.tight_layout()
                path = output_dir / f"test_per_class_f1_{slug}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                plots.append(str(path))

        pivot = per_class.pivot(
            index="name",
            columns="model_label",
            values="f1",
        ).reindex(DBPEDIA_LABELS)
        ax = pivot.plot(kind="bar", figsize=(14, 6))
        ax.set_xlabel("Class")
        ax.set_ylabel("F1")
        ax.set_title("Test per-class F1 comparison")
        ax.set_ylim(*_metric_ylim(pivot.stack(), padding=0.004))
        ax.tick_params(axis="x", rotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8)
        plt.tight_layout()
        plots.extend(_save_current_figure_to([output_dir / "test_per_class_f1.png"]))

    return plots


def _mark_best_and_latest(
    plt,
    group: pd.DataFrame,
    y_column: str,
    best_label: str,
    latest_label: str,
) -> None:
    best_rows = group[group["is_best_epoch"] == 1]
    latest_rows = group[group["is_latest_epoch"] == 1]
    if not best_rows.empty:
        row = best_rows.iloc[0]
        plt.scatter(
            [row["epoch"]],
            [row[y_column]],
            marker="*",
            s=180,
            edgecolors="black",
            linewidths=0.7,
            label=best_label,
            zorder=5,
        )
    if not latest_rows.empty:
        row = latest_rows.iloc[0]
        plt.scatter(
            [row["epoch"]],
            [row[y_column]],
            marker="X",
            s=100,
            edgecolors="black",
            linewidths=0.7,
            label=latest_label,
            zorder=5,
        )


def generate_report_assets(
    config=None,
    predictions_path: Path | str | None = None,
    model_runs_path: Path | str = "reports/model_runs.json",
    output_dir: Path | str = "reports",
) -> tuple[dict, list[str]]:
    config = config or get_config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        Path(predictions_path)
        if predictions_path
        else Path(config.run_dir) / f"test_{infer_model_slug(config)}.parquet"
    )

    model_runs_path = Path(model_runs_path)
    model_runs = load_model_runs(model_runs_path)
    model_outputs = {}
    if model_runs.get("models"):
        model_outputs = _write_model_tables(model_runs, output_dir)
        save_model_runs(model_runs_path, model_runs)

    dataset = _write_dataset_distribution(config, output_dir)
    environment = _write_environment(output_dir)
    test = (
        None
        if model_runs.get("models")
        else _write_test_metrics(predictions_path, output_dir)
    )

    plots = []
    try:
        plots.extend(_generate_static_plots(output_dir))
        if model_runs.get("models"):
            plots.extend(_generate_model_plots(output_dir, model_runs))
    except ModuleNotFoundError as error:
        print(f"Skipping plots because a dependency is missing: {error}")

    summary = {
        "dataset": dataset,
        "environment": environment,
        "test": test,
        "models": model_outputs,
        "plots": plots,
        "artifacts": _artifact_manifest(model_runs, output_dir),
        "checkpoints": {
            "best": _load_checkpoint_summary(best_checkpoint_path(config)),
            "latest": _load_checkpoint_summary(latest_checkpoint_path(config)),
        },
    }
    _write_json(output_dir / "metrics_summary.json", summary)

    return summary, plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        default=None,
        help="Prediction parquet path. Defaults to runs/test_<model_slug>.parquet.",
    )
    parser.add_argument("--models", default="reports/model_runs.json")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    _, plots = generate_report_assets(
        config=get_config(),
        predictions_path=args.predictions,
        model_runs_path=args.models,
        output_dir=args.output_dir,
    )

    print(f"Wrote summary to {Path(args.output_dir) / 'metrics_summary.json'}")
    if plots:
        print("Wrote plots:")
        for plot in plots:
            print(f"  {plot}")


if __name__ == "__main__":
    main()
