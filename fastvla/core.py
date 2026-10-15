"""
core.py

FastVLA: training-free, task-conditioned, temporally adaptive visual token compression for VLA models.

FastVLA operates on the input sequence of a VLA's LLM backbone, laid out as

    [BOS | N visual tokens | task tokens]

and, at every action-prediction step, keeps only the visual tokens that the current decision needs:

    1. Importance   s_i = mean over task tokens t of softmax_i( <RMSNorm(t), RMSNorm(v_i)> / sqrt(d) )
                    (task-to-vision attention; zero-vector placeholder rows in the task span are skipped)
    2. Uniformity   U(t) = K_eff / K, where K_eff = 1 / sum_j q_j^2 is the inverse-Simpson effective number of
                    the top-K importance scores normalized to sum to one, with K = ceil((1 - (rho + tau)) * N), the
                    smallest budget the rate band can request
    3. Rate         alpha(t) = clip((U(t) - min W) / (max W - min W), 0, 1), where W holds U of the previous
                    `window` steps; rate(t) = rho + tau - 2 * tau * alpha(t), i.e. rate(t) lies in
                    [rho - tau, rho + tau] (the first two steps of an episode use rho - tau)
    4. Selection    keep the round((1 - rate(t)) * N) most important visual tokens, in their original order

Hyperparameters: `rho` (target compression rate), `tau` (allowed per-step deviation from rho) and `window`
(length of the sliding calibration window).
"""

import math
from collections import deque

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Parameter-free RMSNorm (computed in float32, returned in the input dtype)."""
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.to(dtype)


class FastVLA(nn.Module):
    """Per-step visual token pruning for one episode stream (batch size 1).

    Call `reset()` at the start of every episode and the module (or `select`) once per action-prediction step.
    """

    def __init__(
        self,
        hidden_size: int,
        num_patches: int,
        rho: float = 0.8,
        tau: float = 0.1,
        window: int = 10,
        enable: bool = True,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches  # N: number of visual tokens in the sequence (all camera views)
        self.scale = 1.0 / math.sqrt(hidden_size)
        self.configure(rho=rho, tau=tau, window=window, enable=enable)

    def configure(self, rho=None, tau=None, window=None, enable=None) -> None:
        """Set hyperparameters (between episodes, never mid-episode)."""
        if rho is not None:
            self.rho = float(rho)
        if tau is not None:
            self.tau = float(tau)
        if window is not None:
            self.window = int(window)
        if enable is not None:
            self.enable = bool(enable)
        assert 0.0 <= self.rho - self.tau and self.rho + self.tau <= 1.0, "need 0 <= rho - tau and rho + tau <= 1"
        assert self.window >= 1, "window must be >= 1"
        self.reset()

    def reset(self) -> None:
        """Start a new episode: clear the calibration window and the step counter."""
        self.history = deque(maxlen=self.window)  # U of the previous `window` steps
        self.step = 0
        self.keep_mask = None  # (1, N) boolean mask of the visual tokens kept at the last step

    def importance(self, visual: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """Task-to-vision attention importance of every visual token. Returns (1, N)."""
        nonzero = task.norm(dim=-1) > 1e-6
        if nonzero.any():
            task = task[nonzero].unsqueeze(0)
        visual = rms_norm(visual)
        task = rms_norm(task)
        attn = torch.softmax(torch.matmul(task, visual.transpose(-2, -1)) * self.scale, dim=-1)
        return attn.mean(dim=1)

    def uniformity(self, scores: torch.Tensor) -> float:
        """Normalized inverse-Simpson effective number U(t) of the top-K scores, K = ceil((1 - (rho + tau)) * N)."""
        K = max(1, math.ceil(scores.shape[-1] * (1 - (self.rho + self.tau)) - 1e-9))
        top, _ = torch.topk(scores, K, largest=True, sorted=True)
        q = top / top.sum()
        return (1.0 / (q * q).sum().item()) / K

    def compression_rate(self, u: float) -> float:
        """Map U(t) to this step's compression rate via min-max calibration over the sliding window."""
        if self.step < 2:
            rate = self.rho - self.tau  # warm-up: the window is not populated yet
        else:
            lo, hi = min(self.history), max(self.history)
            if abs(hi - lo) < 1e-12:
                rate = self.rho
            else:
                alpha = max(0.0, min(1.0, (u - lo) / (hi - lo)))
                rate = self.rho + self.tau - 2 * self.tau * alpha
        self.history.append(u)
        return rate

    @torch.no_grad()
    def select(self, visual: torch.Tensor, task: torch.Tensor) -> torch.Tensor:
        """One FastVLA step on (1, N, d) visual and (1, M, d) task tokens. Returns the (1, N) boolean keep mask."""
        assert visual.shape[0] == 1, "FastVLA processes one episode at a time (batch size 1)"
        num_visual = visual.shape[1]

        # Score the visual tokens, then set this step's compression rate from how concentrated the scores are
        scores = self.importance(visual, task)
        rate = self.compression_rate(self.uniformity(scores))
        self.step += 1

        # Keep the top-k visual tokens (as a boolean mask, which preserves their original order)
        k = max(1, round(num_visual * (1.0 - rate)))
        keep = torch.zeros(1, num_visual, dtype=torch.bool, device=visual.device)
        keep.scatter_(1, scores.topk(k, dim=1, largest=True).indices, True)
        self.keep_mask = keep
        return keep

    def forward(self, tokens: torch.Tensor, position_ids: torch.Tensor, attention_mask=None):
        """Prune the visual tokens of a [BOS | N visual | task] sequence.

        Returns `(tokens, position_ids, attention_mask)` with the dropped visual tokens removed; kept tokens keep
        their original position ids. Sequences without a full visual span (e.g. cached decoding steps) and disabled
        pruning pass through unchanged.
        """
        bsz, seq_len, dim = tokens.shape
        if not self.enable or seq_len <= self.num_patches + 1:
            return tokens, position_ids, attention_mask

        split = [1, self.num_patches, seq_len - self.num_patches - 1]
        bos, visual, task = torch.split(tokens, split, dim=1)
        keep = self.select(visual, task)

        tokens = torch.cat([bos, visual[keep].view(bsz, -1, dim), task], dim=1)
        bos_pos, visual_pos, task_pos = torch.split(position_ids, split, dim=1)
        position_ids = torch.cat([bos_pos, visual_pos[keep].view(bsz, -1), task_pos], dim=1)
        if attention_mask is not None:
            bos_mask, visual_mask, task_mask = torch.split(attention_mask, split, dim=1)
            attention_mask = torch.cat([bos_mask, visual_mask[keep].view(bsz, -1), task_mask], dim=1)

        return tokens, position_ids, attention_mask
