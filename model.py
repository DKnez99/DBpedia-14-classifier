"""Encoder-only Transformer classifier for DBpedia."""

import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    """Token embedding layer with Transformer-style scaling."""

    def __init__(self, vocab_size: int, model_dimension: int, pad_token_id: int) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.embedding = nn.Embedding(
            vocab_size,
            model_dimension,
            padding_idx=pad_token_id,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, model_dimension: int, context_size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros(context_size, model_dimension)
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float()
            * (-math.log(10000.0) / model_dimension)
        )

        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.size(1)
        x = x + self.positional_encoding[:, :sequence_length]
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """Thin wrapper around PyTorch layer normalization."""

    def __init__(self, model_dimension: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(model_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class MultiHeadAttention(nn.Module):
    """Batch-first self-attention with key padding mask support."""

    def __init__(self, model_dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dimension,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output, _ = self.attention(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return output


class FeedForward(nn.Module):
    """Position-wise feed-forward network used inside an encoder block."""

    def __init__(
        self,
        model_dimension: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(model_dimension, feed_forward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feed_forward_dimension, model_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(
        self,
        model_dimension: int,
        num_heads: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(model_dimension, num_heads, dropout)
        self.feed_forward = FeedForward(
            model_dimension,
            feed_forward_dimension,
            dropout,
        )
        self.attention_norm = LayerNormalization(model_dimension)
        self.feed_forward_norm = LayerNormalization(model_dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attention_input = self.attention_norm(x)
        x = x + self.dropout(
            self.self_attention(
                attention_input,
                attention_input,
                attention_input,
                key_padding_mask=key_padding_mask,
            )
        )

        feed_forward_input = self.feed_forward_norm(x)
        x = x + self.dropout(self.feed_forward(feed_forward_input))
        return x


class Encoder(nn.Module):
    """Stack of Transformer encoder blocks."""

    def __init__(
        self,
        model_dimension: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    model_dimension=model_dimension,
                    num_heads=num_heads,
                    feed_forward_dimension=feed_forward_dimension,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = LayerNormalization(model_dimension)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """Projection from pooled encoder output to DBpedia class logits."""

    def __init__(self, model_dimension: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model_dimension, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class DBpediaTransformerClassifier(nn.Module):
    """Encoder-only Transformer text classifier."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        context_size: int,
        model_dimension: int,
        num_layers: int,
        num_heads: int,
        feed_forward_dimension: int,
        dropout: float,
        pad_token_id: int,
        pooling: str = "cls",
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be either 'cls' or 'mean'")

        self.pad_token_id = pad_token_id
        self.pooling = pooling
        self.embeddings = InputEmbeddings(vocab_size, model_dimension, pad_token_id)
        self.positions = PositionalEncoding(model_dimension, context_size, dropout)
        self.encoder = Encoder(
            model_dimension=model_dimension,
            num_layers=num_layers,
            num_heads=num_heads,
            feed_forward_dimension=feed_forward_dimension,
            dropout=dropout,
        )
        self.classification_head = ClassificationHead(
            model_dimension=model_dimension,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        key_padding_mask = attention_mask == 0
        x = self.embeddings(input_ids)
        x = self.positions(x)
        encoded = self.encoder(x, key_padding_mask=key_padding_mask)
        pooled = self._pool(encoded, attention_mask)
        return self.classification_head(pooled)

    def _pool(self, encoded: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return encoded[:, 0]

        mask = attention_mask.unsqueeze(-1).type_as(encoded)
        summed = (encoded * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / token_counts


def build_classifier(config, vocab_size: int, pad_token_id: int = 1):
    """Build the DBpedia Transformer classifier."""

    if config.model_dimension % config.num_heads != 0:
        raise ValueError("model_dimension must be divisible by num_heads")

    model = DBpediaTransformerClassifier(
        vocab_size=vocab_size,
        num_classes=config.num_classes,
        context_size=config.text_context_size,
        model_dimension=config.model_dimension,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        feed_forward_dimension=config.feed_forward_dimension,
        dropout=config.dropout,
        pad_token_id=pad_token_id,
        pooling=config.pooling,
    )

    for parameter in model.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return model
