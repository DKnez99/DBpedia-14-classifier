# DBpedia Title/Content Classification With a Transformer

This project is an educational implementation of a small encoder-only Transformer for classifying DBpedia article title and content into one of 14 DBpedia ontology categories.

The goal is to understand the full path from raw natural language text to class probabilities:

```text
title + content
    -> tokens
    -> token embeddings
    -> positional encoding
    -> Transformer encoder
    -> pooled text representation
    -> classification layer
    -> one of 14 DBpedia classes
```

The task is supervised multiclass text classification. During training, the model compares its predicted class distribution with the correct DBpedia label using cross entropy loss.

## Data

The dataset comes from `fancyzhx/dbpedia_14` and is stored locally as parquet files:

```text
data/
  train.parquet
  test.parquet
  DBPEDIA_SOURCE.txt
```

Each split contains three columns:

```text
label    integer class ID from 0 to 13
title    article title
content  article content
```

The official train split has 560,000 examples. The official test split has 70,000 examples. The classes are balanced, with 40,000 official training examples and 5,000 test examples per class. During local training, 10% of the official train split is reserved for validation, so the actual model-fitting subset has 36,000 examples per class and the validation subset has 4,000 examples per class.

## Classes

The 14 target classes are:

```text
0   Company
1   EducationalInstitution
2   Artist
3   Athlete
4   OfficeHolder
5   MeanOfTransportation
6   Building
7   NaturalPlace
8   Village
9   Animal
10  Plant
11  Album
12  Film
13  WrittenWork
```

## Input Text

The model receives one text sequence per example. The title and content are joined with explicit field labels:

```text
title: <title>
content: <content>
```

Keeping the field names in the input is useful because the title is usually a compact entity name, while the content contains a longer description. The encoder still sees one token sequence, but the labels give it a simple signal about where each field starts.

## Tokenization

This project uses one tokenizer for article text. There is no target tokenizer because classification predicts a fixed class ID, not an output sequence.

The tokenizer should include these special tokens:

```text
[UNK]  unknown token
[PAD]  padding token
[CLS]  sequence-level classification token
[SEP]  separator token
```

The expected encoded sequence is:

```text
[CLS] title/content tokens [SEP] [PAD] ...
```

Sequences are padded or truncated to `text_context_size`. The default is 192 tokens, which is enough for nearly all examples in the local DBpedia files while avoiding the much higher attention cost of 512-token sequences.

Training dataloaders batch-tokenize examples with `tokenizer.encode_batch(...)`. This avoids tokenizing one row at a time in `Dataset.__getitem__` on every epoch and keeps the dataset object simple.

## Model

The architecture is encoder-only:

```text
input token IDs
    -> token embeddings
    -> positional encoding
    -> Transformer encoder blocks
    -> pooling
    -> linear classification head
    -> 14 logits
```

The encoder uses a padding mask so attention ignores `[PAD]` positions.

The classification head should consume either:

- the final hidden state of the `[CLS]` token, or
- a masked mean pool over all non-padding tokens.

The pooling strategy is controlled by `ProjectConfig.pooling`. The current default is `"mean"`, which pools information from all non-padding tokens. `"cls"` is also supported and uses only the final hidden state of the `[CLS]` token.

## Train, Validation, and Test Splits

The DBpedia dataset provides only `train` and `test` parquet files. The project should not tune model choices directly on the official test split, because that would make the final test score too optimistic.

Instead, training should split a small part of `data/train.parquet` into a local validation set:

```text
official train split
    -> local train subset
    -> local validation subset

official test split
    -> final evaluation only
```

The default configuration reserves 10% of the official train split for validation. With 560,000 training rows, that gives:

```text
504,000 local training examples      36,000 per class
 56,000 local validation examples     4,000 per class
 70,000 final test examples           5,000 per class
```

The local training subset is the only subset used to update model weights. The validation subset is used after each epoch to measure generalization on examples that were not used for gradient updates; it controls best-checkpoint selection and exposes overfitting when training loss keeps falling but validation loss rises. The final test split is kept separate until the end and is used only for the final reported score.

The split must be deterministic using the configured random seed. A deterministic split makes results reproducible and ensures that checkpoint selection is based on the same validation examples across runs.

Because DBpedia is balanced by class, the validation split should preserve class balance. The preferred implementation is a stratified split by `label`, so each class contributes the same proportion of examples to validation.

## Training Objective

For each batch:

```text
input_ids, attention_mask, label
    -> model
    -> logits with shape [batch_size, 14]
    -> cross entropy loss against label
```

The main validation metric should be accuracy. Macro-F1 is also useful because it reports per-class behavior equally, even though DBpedia is balanced.

## Training

Start training with:

```bash
./.venv/bin/python train.py
```

The training loop saves:

```text
checkpoints/latest_<model_slug>.pt
checkpoints/best_<model_slug>.pt
```

The `.pt` files are PyTorch checkpoint files. They contain the trained model weights, optimizer state, epoch number, global step, validation metrics, best validation metrics, serialized configuration, vocabulary size, padding token ID, and tokenizer path. They are enough to resume training or load a trained classifier, as long as the matching tokenizer file is present at `tokenizers/dbpedia_text_tokenizer.json`.

The model slug is derived from the architecture, for example `cls_d256_l4_h8_ff1024`.

`latest_<model_slug>.pt` is updated every epoch. `best_<model_slug>.pt` is updated only when validation accuracy improves.
`latest_<model_slug>.pt` is also updated every `checkpoint_every_steps` optimizer steps and when training is stopped with `Ctrl+C`. The default step interval is 1000 optimizer steps.

Checkpoint files are intentionally trackable in git for this school project so trained models can be submitted with the code. They are large; if the remote repository rejects them, store them with Git LFS or attach them separately.

Resume training with:

```bash
./.venv/bin/python train.py --resume-from-checkpoint checkpoints/latest_cls_d256_l4_h8_ff1024.pt
```

When the latest checkpoint was saved during an epoch, training restores the model, optimizer, global step, epoch number, and deterministic dataloader position, then continues from the next batch.

Useful configuration knobs in `config.py`:

```text
text_context_size    shorter is faster, longer preserves more article content
vocab_size           larger covers more subwords, smaller trains faster
pooling              "cls" or "mean"
model_dimension      width of token representations
num_layers           number of Transformer encoder blocks
num_heads            number of attention heads
feed_forward_dimension hidden size inside each encoder block
batch_size           increase if memory allows
device               "auto", "cpu", "cuda", or "mps"
num_workers          DataLoader worker processes
torch_num_threads    CPU compute threads used by PyTorch
torch_num_interop_threads CPU inter-op threads used by PyTorch
num_epochs           total number of epochs to train
learning_rate        optimizer step size
weight_decay         AdamW regularization
max_grad_norm        gradient clipping threshold
checkpoint_every_steps optimizer-step interval for updating latest checkpoint
save_on_interrupt    update latest checkpoint on Ctrl+C
```

The default configuration is the small mean-pooling model:

```text
text_context_size       192
vocab_size              30000
model_dimension         128
num_layers              2
num_heads               4
feed_forward_dimension  512
pooling                 mean
batch_size              64
num_epochs              4
learning_rate           5e-4
```

For faster CPU experiments, reduce `num_epochs`, `vocab_size`, or `text_context_size`. For stronger results on a GPU, increase `model_dimension` to 256, `num_layers` to 4, and possibly `batch_size` if memory allows.

Device selection defaults to `auto`: CUDA is used when available, then MPS, then CPU. You can verify available devices with:

```bash
./.venv/bin/python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_built()); print(torch.backends.mps.is_available())"
```

Override the selected device with:

```bash
./.venv/bin/python train.py --device mps
```

CPU training can use explicit thread settings:

```bash
./.venv/bin/python train.py --device cpu --torch-threads 8 --torch-interop-threads 2
```

For this project, `num_workers` may or may not help because tokenization is already batched in the collator. Start with the default `0`; try `--num-workers 2` and keep it only if epoch time improves.

## Collecting Report Metrics

To train a model and collect the data needed for the report, run the full workflow:

```bash
./.venv/bin/python train.py

./.venv/bin/python infer.py \
  --input-parquet data/test.parquet
```

`train.py` writes checkpoints and updates training/validation report data after every completed epoch:

```text
checkpoints/best_<model_slug>.pt
checkpoints/latest_<model_slug>.pt
reports/model_runs.json
```

`infer.py` writes full test predictions and, when the input parquet has labels and no `--limit` is used, updates test metrics:

```text
runs/test_<model_slug>.parquet
reports/model_runs.json
reports/test_metrics.csv
```

The default parquet filename is derived from the model architecture. For example, the current config writes a path like:

```text
runs/test_cls_d256_l4_h8_ff1024.parquet
```

Pass `--output-parquet` only when you want to override that default path.

`scripts/generate_report_assets.py` derives the remaining report tables and graphs:

```text
reports/metrics_summary.json
reports/environment.json
reports/dataset_distribution.csv
reports/loss_curve.csv
reports/loss_curve.png
reports/loss_curve_<model_slug>.csv
reports/loss_curve_<model_slug>.png
reports/validation_metrics_curve.csv
reports/validation_metrics_curve.png
reports/validation_metrics_curve_<model_slug>.csv
reports/validation_metrics_curve_<model_slug>.png
reports/test_metrics.csv
reports/test_metrics.png
reports/test_metrics_<model_slug>.csv
reports/test_metrics_<model_slug>.png
reports/test_per_class_f1.csv
reports/test_per_class_f1.png
reports/test_per_class_f1_<model_slug>.csv
reports/test_per_class_f1_<model_slug>.png
reports/dataset_distribution.png
```

`train.py` refreshes these report assets after each completed epoch, and `infer.py` refreshes them after a full labeled parquet inference run. Use `--no-report-refresh` on either command only when you want to skip plot/summary regeneration during a quick local run.

You can still regenerate the report assets manually:

```bash
./.venv/bin/python scripts/generate_report_assets.py
```

The canonical model label used in JSON, CSV files, and graph legends is derived from the active architecture config, for example:

```text
Mean Pooling, 2L, d_model=128, 4H, d_ff=512
CLS Pooling, 4L, d_model=256, 8H, d_ff=1024
Mean Pooling, 4L, d_model=256, 8H, d_ff=1024
```

Use `--limit` only for quick inference checks. Limited inference runs are intentionally not recorded as final report test metrics because they are not computed on the full test split.

## Current Report Results

The current report includes four trained models:

```text
CLS Pooling, 2L, d_model=128, 4H, d_ff=512     test accuracy 0.9860
Mean Pooling, 2L, d_model=128, 4H, d_ff=512    test accuracy 0.9864
CLS Pooling, 4L, d_model=256, 8H, d_ff=1024    test accuracy 0.9858
Mean Pooling, 4L, d_model=256, 8H, d_ff=1024   test accuracy 0.9869
```

The best test result is currently the larger mean-pooling model: `Mean Pooling, 4L, d_model=256, 8H, d_ff=1024`.

## Inference

Run single-example inference from the best checkpoint:

```bash
./.venv/bin/python infer.py \
  --title "Henkel" \
  --content "Henkel AG & Company KGaA operates worldwide with leading brands and technologies."
```

For repeated predictions in Python, load the checkpoint once:

```python
from infer import DBpediaClassifierPredictor

predictor = DBpediaClassifierPredictor()
result = predictor.predict(
    title="Henkel",
    content="Henkel AG & Company KGaA operates worldwide with leading brands and technologies.",
)
print(result["category"], result["confidence"])
```

Run batch inference over a parquet file:

```bash
./.venv/bin/python infer.py \
  --device mps \
  --input-parquet data/test.parquet
```

If the input parquet has a `label` column, the output includes `true_label` and `correct`, and the CLI prints accuracy. Use `--limit` for a quick sample:

```bash
./.venv/bin/python infer.py \
  --device mps \
  --input-parquet data/test.parquet \
  --limit 1000
```

Batch inference defaults to summary output. It prints accuracy whenever the parquet file includes true labels, plus correct, incorrect, and predicted row counts. Control optional row output with `--print-mode`:

```bash
# Print only accuracy and row count.
./.venv/bin/python infer.py --input-parquet data/test.parquet

# Print predicted label/category and confidence for each row.
./.venv/bin/python infer.py --input-parquet data/test.parquet --print-mode compact

# Print title, content, prediction, confidence, and true label for every row.
./.venv/bin/python infer.py --input-parquet data/test.parquet --print-mode all

# Print detailed rows only where the prediction is wrong.
./.venv/bin/python infer.py --input-parquet data/test.parquet --print-mode wrong
```

## Project Files

```text
config.py      central configuration and DBpedia class names
data.py        parquet loading, deterministic validation split, and batched dataset collation
tokenizer.py   text tokenizer training/loading and single/batch classifier encoding
model.py       encoder-only Transformer classifier
train.py       training loop and checkpoint saving
evaluate.py    validation loop and classification metrics
infer.py       checkpoint loading, reusable predictor, and single-example categorization
utils.py       shared device, seed, config, and metric serialization helpers
```
