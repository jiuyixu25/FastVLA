"""
fastvla_cogact.py

FastVLA for CogACT on SIMPLER.

CogACT reads its cognition feature from the last position of the VLM prefill sequence
[BOS | visual tokens | prompt, 29871, EOS]; FastVLA prunes the visual tokens of that prefill pass, which leaves the
last position untouched. Usage, after setting up CogACT's SIMPLER evaluation (https://github.com/microsoft/CogACT):
in `SimplerEnv/simpler_env/main_inference.py`, right after the `CogACTInference` model is created, add

    from fastvla_cogact import install_fastvla
    install_fastvla(model)

and put this directory on PYTHONPATH. Hyperparameters are read from the environment unless passed explicitly:
FASTVLA_RHO (default 0.5), FASTVLA_TAU (0.1), FASTVLA_WINDOW (10), FASTVLA_ENABLE (1).
"""

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))  # FastVLA repository; appended so it never shadows CogACT

import torch

from fastvla import FastVLA


def _cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0


def install_fastvla(policy, rho=None, tau=None, window=None, enable=None) -> FastVLA:
    """Attach FastVLA to a `CogACTInference` policy. Returns the pruner."""
    rho = float(os.environ.get("FASTVLA_RHO", 0.5)) if rho is None else rho
    tau = float(os.environ.get("FASTVLA_TAU", 0.1)) if tau is None else tau
    window = int(os.environ.get("FASTVLA_WINDOW", 10)) if window is None else window
    enable = os.environ.get("FASTVLA_ENABLE", "1") == "1" if enable is None else enable

    vlm = policy.vla.vlm
    llama = vlm.llm_backbone.llm.model  # Hugging Face LlamaModel
    pruner = FastVLA(llama.config.hidden_size, num_patches=0, rho=rho, tau=tau, window=window, enable=enable)

    # The number of visual tokens is read from the projector output
    original_projector_forward = vlm.projector.forward

    def projector_forward(x):
        out = original_projector_forward(x)
        pruner.num_patches = out.shape[1]
        return out

    vlm.projector.forward = projector_forward

    # Prune the visual tokens at the prefill forward pass of the LLM backbone
    original_forward = llama.forward

    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None, **kwargs):
        num_patches = pruner.num_patches
        if (
            pruner.enable
            and num_patches > 0
            and inputs_embeds is not None
            and inputs_embeds.shape[1] > num_patches + 1
            and _cache_length(past_key_values) == 0
        ):
            positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
            mask = attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None
            inputs_embeds, position_ids, mask = pruner(inputs_embeds, positions, mask)
            attention_mask = mask if mask is not None else attention_mask
            cache_position = None
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position, **kwargs,
        )

    llama.forward = forward

    # SIMPLER calls policy.reset(task_description) at the start of every episode
    original_reset = policy.reset

    def reset(*args, **kwargs):
        pruner.reset()
        return original_reset(*args, **kwargs)

    policy.reset = reset

    print(f"[FastVLA] CogACT: enable={enable}, rho={rho}, tau={tau}, window={window}", flush=True)
    return pruner
