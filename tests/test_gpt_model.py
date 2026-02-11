"""Tests for GPT core model modules."""

from __future__ import annotations

import pytest
import torch

from llmtrain.models.gpt import GPT, CausalSelfAttention, TransformerBlock


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


def test_transformer_block_preserves_shape_and_backpropagates_gradients() -> None:
    torch.manual_seed(0)
    batch_size = 2
    seqlen = 8
    d_model = 16

    block = TransformerBlock(
        d_model=d_model,
        n_heads=4,
        d_ff=32,
        block_size=seqlen,
        dropout=0.0,
    )
    block.train()

    x = torch.randn(batch_size, seqlen, d_model, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape

    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0

    # Verify gradients reach internal parameters (catches accidental detach).
    assert block.attn.qkv_proj.weight.grad is not None
    assert block.attn.out_proj.weight.grad is not None
    assert block.mlp_fc.weight.grad is not None
    assert block.mlp_proj.weight.grad is not None


def test_gpt_forward_returns_vocab_logits_shape() -> None:
    torch.manual_seed(0)
    batch_size = 3
    seqlen = 7
    vocab_size = 32

    model = GPT(
        vocab_size=vocab_size,
        block_size=seqlen,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        tie_embeddings=True,
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (batch_size, seqlen, vocab_size)


def test_gpt_weight_tying() -> None:
    """When tie_embeddings=True the LM head shares the token embedding weight tensor."""
    model = GPT(
        vocab_size=32,
        block_size=8,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        tie_embeddings=True,
    )
    assert model.lm_head.weight is model.token_embedding.weight

    # Untied variant must have independent tensors.
    model_untied = GPT(
        vocab_size=32,
        block_size=8,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
        tie_embeddings=False,
    )
    assert model_untied.lm_head.weight is not model_untied.token_embedding.weight


def test_gpt_raises_when_sequence_exceeds_block_size() -> None:
    """Forward must raise ValueError when T > block_size."""
    block_size = 8
    model = GPT(
        vocab_size=32,
        block_size=block_size,
        d_model=64,
        n_layers=1,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )
    too_long = torch.randint(0, 32, (1, block_size + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds block size"):
        model(too_long)


def test_gpt_logits_prefix_invariant_to_future_tokens() -> None:
    """End-to-end GPT causality: logits[:, :t+1] are unaffected by changes to tokens after t."""
    torch.manual_seed(42)
    vocab_size = 32
    seqlen = 8
    model = GPT(
        vocab_size=vocab_size,
        block_size=seqlen,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )
    model.eval()

    input_ids = torch.randint(0, vocab_size, (2, seqlen), dtype=torch.long)
    logits_original = model(input_ids)

    t = 3
    input_ids_modified = input_ids.clone()
    input_ids_modified[:, t + 1 :] = torch.randint(
        0, vocab_size, input_ids_modified[:, t + 1 :].shape, dtype=torch.long
    )
    logits_modified = model(input_ids_modified)

    torch.testing.assert_close(
        logits_original[:, : t + 1, :],
        logits_modified[:, : t + 1, :],
        atol=1e-6,
        rtol=0.0,
    )
