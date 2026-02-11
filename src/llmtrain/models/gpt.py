from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from llmtrain.config.schemas import RunConfig
from llmtrain.models.base import ModelAdapter
from llmtrain.registry.models import register_model


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a static causal mask."""

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        self.resid_dropout = nn.Dropout(dropout)

        causal_mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        causal_mask = self.causal_mask
        if not isinstance(causal_mask, torch.Tensor):
            raise TypeError("Expected causal_mask buffer to be a torch.Tensor.")
        block_size = causal_mask.shape[-1]
        if seqlen > block_size:
            raise ValueError(f"Input sequence length {seqlen} exceeds block size {block_size}.")
        if attention_mask is not None and attention_mask.shape != (bsz, seqlen):
            raise ValueError(
                "Expected attention_mask to have shape (B, T); "
                f"got {tuple(attention_mask.shape)} for {(bsz, seqlen)}."
            )

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        mask_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(causal_mask[:, :, :seqlen, :seqlen], mask_value)

        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            padding_mask = attention_mask.bool()
            key_padding_mask = ~padding_mask[:, None, None, :]  # (B, 1, 1, T)
            attn_scores = attn_scores.masked_fill(key_padding_mask, mask_value)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = attn_probs @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.d_model)
        out = self.resid_dropout(self.out_proj(out))

        if padding_mask is not None:
            out = out * padding_mask[:, :, None].to(dtype=out.dtype)

        return out


class TransformerBlock(nn.Module):
    """GPT-style pre-norm transformer block."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, block_size: int, dropout: float
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp_fc = nn.Linear(d_model, d_ff)
        self.mlp_act = nn.GELU()
        self.mlp_proj = nn.Linear(d_ff, d_model)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)

        mlp_out = self.mlp_fc(self.ln_2(x))
        mlp_out = self.mlp_act(mlp_out)
        mlp_out = self.mlp_dropout(self.mlp_proj(mlp_out))
        x = x + mlp_out
        return x


class GPT(nn.Module):
    """Decoder-only GPT language model."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_residual_projections(self) -> None:
        scaled_std = 0.02 / math.sqrt(2 * self.n_layers)
        for block in self.blocks:
            block = cast(TransformerBlock, block)
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=scaled_std)
            nn.init.normal_(block.mlp_proj.weight, mean=0.0, std=scaled_std)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        if seqlen > self.block_size:
            raise ValueError(
                f"Input sequence length {seqlen} exceeds block size {self.block_size}."
            )

        position_ids = torch.arange(seqlen, device=input_ids.device)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)[None, :, :]
        x = self.drop(token_emb + pos_emb)

        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.ln_f(x)
        return self.lm_head(x)


@register_model("gpt")
class GPTAdapter(ModelAdapter):
    """Model adapter for the decoder-only GPT implementation."""

    def build_model(self, cfg: RunConfig) -> nn.Module:
        vocab_size = cfg.model.vocab_size or 128
        return GPT(
            vocab_size=vocab_size,
            block_size=cfg.model.block_size,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            d_ff=cfg.model.d_ff,
            dropout=cfg.model.dropout,
            tie_embeddings=cfg.model.tie_embeddings,
        )

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
        if input_ids.size(1) < 2:
            raise ValueError("Expected sequence length >= 2 for next-token loss.")

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
        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]

        loss_per_token = nn.functional.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.size(-1)),
            shifted_labels.reshape(-1),
            reduction="none",
        )

        if attention_mask is None:
            loss = loss_per_token.mean()
        else:
            token_mask = attention_mask[:, 1:].to(dtype=torch.bool).reshape(-1)
            valid_count = int(token_mask.sum().item())
            if valid_count == 0:
                raise ValueError("attention_mask has no valid target tokens after shift.")
            loss = loss_per_token[token_mask].mean()

        return loss, {"loss": float(loss.item())}
