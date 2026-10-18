"""
run_libero_eval_fastvla.py

FastVLA for the base OpenVLA model on LIBERO, using the official OpenVLA evaluation script unchanged.

Base OpenVLA predicts 7 discrete action tokens autoregressively from one 224x224 image (256 visual tokens). Its LLM
input at the first (prefill) forward pass is [BOS | 256 visual tokens | prompt], so FastVLA prunes the visual tokens
there; the decoding steps then attend to the pruned KV cache.

Run from the root of the official OpenVLA repository (https://github.com/openvla/openvla), with LIBERO installed:

    python /path/to/FastVLA/integrations/openvla/run_libero_eval_fastvla.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial --center_crop True \
        --fastvla_rho 0.35 --fastvla_tau 0.1 --fastvla_window 10

The --fastvla_{enable,rho,tau,window} arguments are consumed here; all other arguments go to the official
experiments/robot/libero/run_libero_eval.py.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())  # official OpenVLA repository (experiments.*, prismatic.*)
sys.path.append(str(Path(__file__).resolve().parents[2]))  # FastVLA repository; appended so it never shadows the above

import torch

from fastvla import FastVLA

NUM_PATCHES = 256  # base OpenVLA: one image -> 256 visual tokens
LLM_HIDDEN, LLM_FFN, LLM_LAYERS = 4096, 11008, 32  # Llama-2-7B


def parse_fastvla_args(argv):
    """Split the --fastvla_* arguments from the arguments of the official evaluation script."""
    args = {"enable": True, "rho": 0.35, "tau": 0.1, "window": 10}
    rest, it = [], iter(argv)
    for a in it:
        key = a[len("--fastvla_"):] if a.startswith("--fastvla_") else None
        if key in args:
            value = next(it)
            args[key] = value.lower() in ("1", "true", "yes") if key == "enable" else type(args[key])(value)
        else:
            rest.append(a)
    return args, rest


def cache_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values.get_seq_length()
    return past_key_values[0][0].shape[2] if len(past_key_values) > 0 else 0


def install(vla, pruner, stats):
    """Prune visual tokens at the prefill forward pass of the LLM backbone (decoding steps are untouched)."""
    llama = vla.language_model.model
    original_forward = llama.forward

    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None,
                use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                cache_position=None, **kwargs):
        prefill = (
            inputs_embeds is not None
            and inputs_embeds.shape[1] > NUM_PATCHES + 1
            and cache_length(past_key_values) == 0
        )
        if prefill:
            kept = NUM_PATCHES
            if pruner.enable:
                split = [1, NUM_PATCHES, inputs_embeds.shape[1] - NUM_PATCHES - 1]
                bos, visual, prompt = torch.split(inputs_embeds, split, dim=1)
                keep = pruner.select(visual, prompt)
                kept = int(keep.sum())
                inputs_embeds = torch.cat([bos, visual[keep].unsqueeze(0), prompt], dim=1)
                # As in the unpruned pipeline: contiguous positions (so that the cached decoding steps stay
                # consistent) and an attention mask one token shorter than the embeddings (OpenVLA appends the
                # empty token 29871 to `input_ids` but not to `attention_mask`)
                position_ids, cache_position = None, None
                attention_mask = torch.ones(1, inputs_embeds.shape[1] - 1, dtype=torch.long, device=inputs_embeds.device)
            n = inputs_embeds.shape[1]
            stats["kept"].append(kept)
            stats["flops"].append(LLM_LAYERS * n * (4 * LLM_HIDDEN**2 + 2 * n * LLM_HIDDEN + 3 * LLM_HIDDEN * LLM_FFN))
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict, cache_position=cache_position, **kwargs,
        )

    llama.forward = forward


def main():
    fastvla_args, rest = parse_fastvla_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *rest]

    import experiments.robot.libero.run_libero_eval as official  # official OpenVLA LIBERO evaluation

    pruner = None
    stats = {"kept": [], "flops": [], "latency": []}

    # Attach FastVLA to the model built by the official script
    original_get_model = official.get_model

    def get_model(cfg, **kwargs):
        nonlocal pruner
        model = original_get_model(cfg, **kwargs)
        pruner = FastVLA(model.language_model.config.hidden_size, NUM_PATCHES, rho=fastvla_args["rho"],
                         tau=fastvla_args["tau"], window=fastvla_args["window"], enable=fastvla_args["enable"])
        install(model, pruner, stats)
        print(f"[FastVLA] {fastvla_args}", flush=True)
        return model

    official.get_model = get_model

    # Episode boundaries: the official script calls env.reset() at the start of every episode
    original_get_env = official.get_libero_env

    def get_libero_env(*args, **kwargs):
        env, task_description = original_get_env(*args, **kwargs)
        original_reset = env.reset

        def reset(*reset_args, **reset_kwargs):
            if pruner is not None:
                pruner.reset()
            return original_reset(*reset_args, **reset_kwargs)

        env.reset = reset
        return env, task_description

    official.get_libero_env = get_libero_env

    # Time every action-prediction call (base OpenVLA executes one action per call)
    original_get_action = official.get_action

    def get_action(*args, **kwargs):
        start = time.time()
        action = original_get_action(*args, **kwargs)
        stats["latency"].append(time.time() - start)
        return action

    official.get_action = get_action

    official.eval_libero()

    if stats["latency"]:
        latency = sum(stats["latency"]) / len(stats["latency"])
        print(
            f"[FastVLA] action-prediction latency: {latency * 1000:.2f} ms | control frequency: {1 / latency:.2f} Hz | "
            f"LLM TFLOPs: {sum(stats['flops']) / len(stats['flops']) * 1e-12:.3f} | "
            f"visual tokens kept: {sum(stats['kept']) / len(stats['kept']):.1f} / {NUM_PATCHES}",
            flush=True,
        )


if __name__ == "__main__":
    main()
