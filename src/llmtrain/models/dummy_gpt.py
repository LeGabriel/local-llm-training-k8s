from __future__ import annotations

from typing import Any

import torch
from torch import nn

from llmtrain.config.schemas import RunConfig
from llmtrain.models.base import ModelAdapter
from llmtrain.registry.models import register_model


class _TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        hidden = self.embed(input_ids)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        return self.lm_head(hidden)


@register_model("dummy_gpt")
class DummyGPTAdapter(ModelAdapter):
    """Tiny adapter for dry-run smoke tests."""

    def build_model(self, cfg: RunConfig) -> nn.Module:
        vocab_size = cfg.model.vocab_size or 128
        d_model_raw = cfg.model.d_model or 128
        d_model = min(d_model_raw, 128)
        n_heads = max(1, min(cfg.model.n_heads, d_model))
        if d_model % n_heads != 0:
            n_heads = 2 if d_model % 2 == 0 else 1
        return _TinyGPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads)

    def build_tokenizer(self, cfg: RunConfig) -> Any | None:
        return None

    def compute_loss(
        self, model: nn.Module, batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        if input_ids.dim() != 2 or labels.dim() != 2:
            raise ValueError(
                f"Expected input_ids and labels to be 2D (B, T); "
                f"got {tuple(input_ids.shape)} and {tuple(labels.shape)}."
            )
        if input_ids.shape != labels.shape:
            raise ValueError(
                "Expected input_ids and labels to have the same shape; "
                f"got {tuple(input_ids.shape)} vs {tuple(labels.shape)}."
            )
        if input_ids.dtype != torch.long or labels.dtype != torch.long:
            raise ValueError(
                "Expected input_ids and labels to be torch.long; "
                f"got {input_ids.dtype} and {labels.dtype}."
            )
        if attention_mask is not None:
            if attention_mask.dim() != 2:
                raise ValueError(
                    f"Expected attention_mask to be 2D (B, T); got {tuple(attention_mask.shape)}."
                )
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "Expected attention_mask to match input_ids shape; "
                    f"got {tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}."
                )
            if attention_mask.dtype not in (torch.bool, torch.long, torch.int64):
                raise ValueError(
                    f"Expected attention_mask to be bool or int64; got {attention_mask.dtype}."
                )
        logits = model(input_ids, attention_mask=attention_mask)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return loss, {"loss": loss.item()}
