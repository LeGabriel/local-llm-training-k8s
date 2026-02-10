"""Tests for GPT core model modules."""

from __future__ import annotations

import torch

from llmtrain.models.gpt import CausalSelfAttention


def test_causal_self_attention_shape_and_future_token_isolation() -> None:
    torch.manual_seed(0)
    batch_size = 2
    seqlen = 6
    d_model = 8

    attn = CausalSelfAttention(
        d_model=d_model,
        n_heads=2,
        block_size=seqlen,
        dropout=0.0,
    )
    attn.eval()

    x = torch.randn(batch_size, seqlen, d_model)
    out_original = attn(x)
    assert out_original.shape == x.shape

    # Modify only future tokens (>= t + 1) and ensure prefix outputs are unchanged.
    t = 2
    x_future_changed = x.clone()
    x_future_changed[:, t + 1 :, :] = torch.randn_like(x_future_changed[:, t + 1 :, :])
    out_future_changed = attn(x_future_changed)

    torch.testing.assert_close(
        out_original[:, : t + 1, :],
        out_future_changed[:, : t + 1, :],
        atol=1e-6,
        rtol=0.0,
    )
