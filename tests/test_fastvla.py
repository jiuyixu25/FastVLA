"""
CPU sanity checks for FastVLA (no checkpoint, no simulator, no GPU).

Run from the repository root:
    python tests/test_fastvla.py      # or: pytest tests/test_fastvla.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fastvla import FastVLA

N, D, M = 512, 64, 72  # visual tokens (2 views x 256), hidden size, task tokens


def make_step(generator, n_placeholders=56):
    """Random [BOS | N visual | task] embeddings whose task span ends with zero-vector placeholders."""
    tokens = torch.randn(1, 1 + N + M, D, generator=generator)
    tokens[:, 1:1 + N] *= torch.rand(1, N, 1, generator=generator) ** (1 + 4 * torch.rand(1, generator=generator))
    tokens[:, 1 + N + M - n_placeholders - 1:-1] = 0.0
    positions = torch.arange(1 + N + M).unsqueeze(0)
    mask = torch.ones(1, 1 + N + M, dtype=torch.long)
    return tokens, positions, mask


def run_episode(pruner, steps=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    pruner.reset()
    kept = []
    for _ in range(steps):
        tokens, positions, mask = make_step(g)
        out, out_pos, out_mask = pruner(tokens, positions, mask)
        k = int(pruner.keep_mask.sum())
        keep_idx = torch.nonzero(pruner.keep_mask[0]).squeeze(1)

        # Sequence bookkeeping: [BOS | kept visual tokens in original order | task], original position ids
        assert out.shape[1] == out_pos.shape[1] == out_mask.shape[1] == 1 + k + M
        assert torch.equal(out[:, 0], tokens[:, 0]) and torch.equal(out[:, 1 + k:], tokens[:, 1 + N:])
        assert torch.equal(out[0, 1:1 + k], tokens[0, 1 + keep_idx])
        assert torch.equal(out_pos[0, 1:1 + k], 1 + keep_idx)
        kept.append(k)
    return kept


def test_rate_stays_in_band():
    rho, tau = 0.8, 0.1
    kept = run_episode(FastVLA(D, N, rho=rho, tau=tau, window=10))
    lo, hi = round(N * (1 - rho - tau)), round(N * (1 - rho + tau))
    assert all(lo <= k <= hi for k in kept), (kept, lo, hi)
    assert kept[0] == kept[1] == hi  # warm-up steps use rate rho - tau
    assert len(set(kept)) > 3  # the rate actually adapts


def test_tau_zero_is_fixed_rate():
    kept = run_episode(FastVLA(D, N, rho=0.7, tau=0.0, window=10))
    assert set(kept) == {round(N * 0.3)}


def test_reset_restarts_warmup():
    pruner = FastVLA(D, N, rho=0.6, tau=0.1, window=10)
    first = run_episode(pruner, seed=1)
    second = run_episode(pruner, seed=1)
    assert first == second


def test_disabled_is_passthrough():
    pruner = FastVLA(D, N, enable=False)
    tokens, positions, mask = make_step(torch.Generator().manual_seed(0))
    out, out_pos, out_mask = pruner(tokens, positions, mask)
    assert out is tokens and out_pos is positions and out_mask is mask


def test_zero_rows_do_not_change_importance():
    pruner = FastVLA(D, N)
    g = torch.Generator().manual_seed(0)
    visual, task = torch.randn(1, N, D, generator=g), torch.randn(1, 10, D, generator=g)
    padded = torch.cat([task, torch.zeros(1, 30, D)], dim=1)
    assert torch.allclose(pruner.importance(visual, task), pruner.importance(visual, padded))


def test_pruned_llama_model():
    """The pruned LLM equals the plain LLM run on the pruned sequence with the kept tokens' original positions."""
    try:
        from transformers import LlamaConfig, LlamaModel

        from prismatic.extern.hf.modeling_prismatic import PrunedLlamaModel
    except ImportError as e:  # the OpenVLA-OFT dependencies are not installed
        print(f"skipped test_pruned_llama_model ({e})")
        return

    torch.manual_seed(0)
    config = LlamaConfig(hidden_size=D, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, vocab_size=100)
    pruned = PrunedLlamaModel(config, num_patches=N).eval()
    plain = LlamaModel(config).eval()
    plain.load_state_dict(pruned.state_dict())
    pruned.pruner.configure(enable=True, rho=0.75, tau=0.1, window=10)

    tokens, positions, mask = make_step(torch.Generator().manual_seed(0))
    with torch.no_grad():
        out = pruned(inputs_embeds=tokens, attention_mask=mask).last_hidden_state
        keep = pruned.pruner.keep_mask[0]
        idx = torch.cat([torch.tensor([0]), 1 + torch.nonzero(keep).squeeze(1), torch.arange(1 + N, 1 + N + M)])
        ref = plain(inputs_embeds=tokens[:, idx], attention_mask=mask[:, idx], position_ids=idx.unsqueeze(0))
    assert torch.allclose(out, ref.last_hidden_state, atol=1e-5)
    assert pruned.num_forward == 1 and pruned.total_kept_patches == int(keep.sum())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK  {name}")
