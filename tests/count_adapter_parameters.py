#!/usr/bin/env python3
"""Count parameters in a LoRA adapter_model.safetensors file."""

import sys
import argparse
from collections import defaultdict
from safetensors import safe_open


def count_params(path: str) -> None:
    total = 0
    by_type = defaultdict(int)   # "lora_A", "lora_B", "other"
    by_layer = defaultdict(int)  # layer name -> param count

    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            tensor = f.get_tensor(key)
            n = tensor.numel()
            total += n

            # Classify by adapter type
            if "lora_A" in key:
                by_type["lora_A"] += n
            elif "lora_B" in key:
                by_type["lora_B"] += n
            else:
                by_type["other"] += n

            # Group by module (strip .lora_A/lora_B.weight suffix)
            parts = key.split(".")
            # Find lora_A or lora_B position
            for marker in ("lora_A", "lora_B"):
                if marker in parts:
                    idx = parts.index(marker)
                    layer = ".".join(parts[:idx])
                    by_layer[layer] += n
                    break
            else:
                by_layer[key] += n

    print(f"File: {path}")
    print(f"Total tensors: {len(keys)}")
    print(f"Total parameters: {total:,}")
    print(f"  lora_A: {by_type['lora_A']:>12,}")
    print(f"  lora_B: {by_type['lora_B']:>12,}")
    if by_type["other"]:
        print(f"  other:  {by_type['other']:>12,}")
    print()

    # Infer rank from lora_A shapes
    ranks = set()
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            if "lora_A" in key:
                t = f.get_tensor(key)
                ranks.add(t.shape[0])  # lora_A: (rank, in_features)
    if ranks:
        print(f"Detected LoRA rank(s): {sorted(ranks)}")
    print()

    # Per-module breakdown
    # print(f"{'Module':<60} {'Params':>12}")
    # print("-" * 74)
    attn_layers = 0
    mlp_layers = 0
    for layer, n in sorted(by_layer.items(), key=lambda x: -x[1]):
        if "attn" in layer:
            attn_layers += n
        else:
            assert "mlp" in layer, f"Unexpected layer name: {layer}"
            mlp_layers += n

    print(f"Attention-related parameters: {attn_layers:,}")
    print(f"MLP-related parameters:       {mlp_layers:,}")


def main():
    # parser = argparse.ArgumentParser(description="Count LoRA adapter parameters.")
    # parser.add_argument("path", nargs="?",
    #                     default="adapter_model.safetensors",
    #                     help="Path to adapter_model.safetensors")
    # args = parser.parse_args()
    import glob
    paths = glob.glob("adapters/*.safetensors")
    for path in paths:
        count_params(path)


if __name__ == "__main__":
    main()
