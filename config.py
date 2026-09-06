"""Configuration for the DBpedia title/content classifier."""

from dataclasses import dataclass
from pathlib import Path


DBPEDIA_LABELS: tuple[str, ...] = (
    "Company",
    "EducationalInstitution",
    "Artist",
    "Athlete",
    "OfficeHolder",
    "MeanOfTransportation",
    "Building",
    "NaturalPlace",
    "Village",
    "Animal",
    "Plant",
    "Album",
    "Film",
    "WrittenWork",
)


@dataclass(frozen=True)
class ProjectConfig:
    """Central project configuration for DBpedia classification."""

    # Model
    text_context_size: int = 192
    num_classes: int = len(DBPEDIA_LABELS)
    class_names: tuple[str, ...] = DBPEDIA_LABELS
    model_dimension: int = 128
    num_layers: int = 2
    num_heads: int = 4
    feed_forward_dimension: int = 512
    dropout: float = 0.1
    pooling: str = "mean"

    # Training
    batch_size: int = 64
    num_epochs: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 561

    # Data
    train_file: Path = Path("data/train.parquet")
    test_file: Path = Path("data/test.parquet")

    # Tokenizer
    tokenizer_dir: Path = Path("tokenizers")
    tokenizer_filename: str = "dbpedia_text_tokenizer.json"
    vocab_size: int = 30000
    min_token_frequency: int = 2

    # Checkpoints
    checkpoint_dir: Path = Path("checkpoints")
    best_checkpoint_filename: str | None = None
    latest_checkpoint_filename: str | None = None
    resume_from_checkpoint: Path | None = None
    checkpoint_every_steps: int = 1000
    save_on_interrupt: bool = True
    run_dir: Path = Path("runs")

    # Reports
    report_dir: Path = Path("reports")

    # Device/runtime
    device: str = "auto"
    num_workers: int = 0
    pin_memory: bool = True
    torch_num_threads: int | None = None
    torch_num_interop_threads: int | None = None


def get_config() -> ProjectConfig:
    """Return the default project configuration."""

    return ProjectConfig()
