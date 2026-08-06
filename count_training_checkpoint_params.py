#!/usr/bin/env python3
"""
Count WL-WMP training-checkpoint parameters without starting Isaac Gym.

Expected checkpoint keys from rsl_rl/runners/wmp_runner.py:
  - model_state_dict
  - world_model_dict
  - depth_predictor

Paper accounting convention:
  WL-WMP training-time method parameters =
      actor_critic + world_model

The training-only pseudo-depth synthesizer (depth_predictor) is printed
separately because it is shared across the matched comparison and should not
be mixed into the method-specific total unless the paper explicitly chooses
that convention.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint across old and new PyTorch versions."""
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint root must be a dict, got {type(obj)!r}")
    return obj


def tensor_numel(state: Any) -> int:
    """Count tensor elements in a state_dict-like mapping."""
    if not isinstance(state, dict):
        return 0
    return sum(
        int(value.numel())
        for value in state.values()
        if torch.is_tensor(value)
    )


def grouped_counts(state: Any) -> list[tuple[str, int]]:
    """Group a state dict by the first key component."""
    groups: dict[str, int] = defaultdict(int)
    if not isinstance(state, dict):
        return []
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        prefix = str(key).split(".", 1)[0]
        groups[prefix] += int(value.numel())
    return sorted(groups.items(), key=lambda item: (-item[1], item[0]))


def print_section(label: str, state: Any) -> int:
    count = tensor_numel(state)
    print(f"\n{label}: {count:,}")
    for prefix, value in grouped_counts(state):
        print(f"  {prefix:<32s} {value:>15,d}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count WL-WMP training-time parameters from a .pt checkpoint."
    )
    parser.add_argument("checkpoint", type=Path, help="Path to model_*.pt")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")
    print("Available keys:", ", ".join(sorted(map(str, checkpoint.keys()))))

    actor_critic = print_section(
        "Actor-critic state (model_state_dict)",
        checkpoint.get("model_state_dict"),
    )
    world_model = print_section(
        "World-model state (world_model_dict)",
        checkpoint.get("world_model_dict"),
    )
    depth_predictor = print_section(
        "Shared pseudo-depth synthesizer (depth_predictor)",
        checkpoint.get("depth_predictor"),
    )

    method_total = actor_critic + world_model
    total_with_depth = method_total + depth_predictor

    print("\n" + "=" * 72)
    print(f"WL-WMP training-time method total:        {method_total:,}")
    print(f"Shared depth predictor, reported apart:   {depth_predictor:,}")
    print(f"Method + depth predictor:                 {total_with_depth:,}")
    print("=" * 72)

    missing = [
        key for key in ("model_state_dict", "world_model_dict", "depth_predictor")
        if key not in checkpoint
    ]
    if missing:
        print("\nWARNING: missing expected keys:", ", ".join(missing))


if __name__ == "__main__":
    main()
