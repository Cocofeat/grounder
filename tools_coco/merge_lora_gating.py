#!/usr/bin/env python3
"""Merge LoRA weights in a gating checkpoint and save a clean model.

Usage:
    python tools_coco/merge_lora_gating.py \
        --input work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating \
        --output work_dirs/internvl_chat_v2_5/internvl3_coco_2025_12_14_multi_gating_merged
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def merge_lora_state_dict(state_dict, lora_alpha=16, lora_rank=8):
    """Merge LoRA weights into base weights and strip PEFT key prefixes.

    Key patterns in PEFT-wrapped checkpoint:
      language_model.base_model.model.model.layers.X.mlp.down_proj.base_layer.weight  (base W)
      language_model.base_model.model.model.layers.X.mlp.down_proj.lora_A.default.weight
      language_model.base_model.model.model.layers.X.mlp.down_proj.lora_B.default.weight
      language_model.base_model.model.model.layers.X.input_layernorm.weight  (no LoRA)
      language_model.base_model.model.lm_head.weight  (no LoRA)
      vision_model.*  (no PEFT prefix)
      mlp1.*  (no PEFT prefix)
      gate_module.*  (no PEFT prefix)

    After merge:
      language_model.model.layers.X.mlp.down_proj.weight  (merged)
      language_model.model.layers.X.input_layernorm.weight
      language_model.lm_head.weight
      vision_model.*, mlp1.*, gate_module.*  (unchanged)
    """
    scaling = lora_alpha / lora_rank

    # Collect LoRA pairs: {base_key_prefix: (lora_A, lora_B)}
    lora_a = {}
    lora_b = {}
    for key in state_dict:
        if '.lora_A.default.weight' in key:
            prefix = key.replace('.lora_A.default.weight', '')
            lora_a[prefix] = state_dict[key]
        elif '.lora_B.default.weight' in key:
            prefix = key.replace('.lora_B.default.weight', '')
            lora_b[prefix] = state_dict[key]

    merged = {}
    processed_prefixes = set()

    for key, tensor in state_dict.items():
        # Skip LoRA weights (will be merged into base)
        if '.lora_A.default.weight' in key or '.lora_B.default.weight' in key:
            continue

        if '.base_layer.' in key:
            # This is a base weight/bias wrapped by PEFT
            # Extract suffix (.weight or .bias) and prefix
            suffix = '.weight' if key.endswith('.weight') else '.bias'
            base_tag = '.base_layer' + suffix
            prefix = key.replace(base_tag, '')

            if suffix == '.weight' and prefix in lora_a and prefix in lora_b:
                # Merge: W = W_base + scaling * (B @ A)
                w_base = tensor.float()
                a = lora_a[prefix].float()
                b = lora_b[prefix].float()
                w_merged = w_base + scaling * (b @ a)
                tensor = w_merged.to(state_dict[key].dtype)
                processed_prefixes.add(prefix)

            # Strip .base_layer and PEFT prefix
            new_key = key.replace(base_tag, suffix)
            new_key = _strip_peft_prefix(new_key)
            merged[new_key] = tensor
        else:
            # Non-LoRA key, just strip PEFT prefix
            new_key = _strip_peft_prefix(key)
            merged[new_key] = tensor

    print(f"Merged {len(processed_prefixes)} LoRA pairs (scaling={scaling})")
    return merged


def _strip_peft_prefix(key):
    """Remove PEFT base_model.model prefix from language_model keys."""
    # language_model.base_model.model.XXX -> language_model.XXX
    if key.startswith('language_model.base_model.model.'):
        return 'language_model.' + key[len('language_model.base_model.model.'):]
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Input checkpoint directory')
    parser.add_argument('--output', required=True, help='Output directory for merged model')
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-rank', type=int, default=8)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load all shards
    print(f"Loading checkpoint from {args.input}")
    idx_path = os.path.join(args.input, 'model.safetensors.index.json')
    idx = json.load(open(idx_path))

    # Collect all unique shard files
    shard_files = sorted(set(idx['weight_map'].values()))
    state_dict = {}
    for shard in shard_files:
        print(f"  Loading {shard}")
        st = load_file(os.path.join(args.input, shard))
        state_dict.update(st)

    print(f"Total keys loaded: {len(state_dict)}")

    # Merge LoRA
    merged = merge_lora_state_dict(state_dict, args.lora_alpha, args.lora_rank)
    print(f"Merged model keys: {len(merged)}")

    # Verify no PEFT artifacts remain
    peft_keys = [k for k in merged if 'base_model' in k or 'lora_' in k or 'base_layer' in k]
    if peft_keys:
        print(f"WARNING: {len(peft_keys)} keys still have PEFT artifacts:")
        for k in peft_keys[:5]:
            print(f"  {k}")
    else:
        print("Clean: no PEFT artifacts in merged keys")

    # Save merged weights
    out_path = os.path.join(args.output, 'model.safetensors')
    print(f"Saving merged model to {out_path}")
    save_file(merged, out_path)

    # Copy config files
    for fname in ['config.json', 'generation_config.json', 'tokenizer_config.json',
                   'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                   'merges.txt']:
        src = os.path.join(args.input, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, fname))
            print(f"  Copied {fname}")

    # Update config to remove PEFT references
    cfg_path = os.path.join(args.output, 'config.json')
    cfg = json.load(open(cfg_path))
    # Set architecture to Gating model
    cfg['architectures'] = ['InternVLChatModelGating']
    json.dump(cfg, open(cfg_path, 'w'), indent=2)
    print(f"Updated config.json")

    print(f"\nDone! Merged checkpoint saved to: {args.output}")


if __name__ == '__main__':
    main()
