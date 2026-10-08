#!/usr/bin/env python3
"""
Sample text from a Text8 flow model (e.g. with the DiT encoder).

Usage:
    python sample_text8.py --config configs/text8_sample_1step.yml
    python sample_text8.py --config configs/text8_sample_ode.yml

Note: Lightning .ckpt files store the training weights. To exactly reproduce
the TensorBoard validation samples, EMA weights are required, but the current
checkpoints do not serialize EMA.
"""

import argparse
import os
import sys
import time
from configuration import parse_run_args

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from models import get_flow_model
from utils import load_config, seed_all

TEXT8_CHARS = list("_abcdefghijklmnopqrstuvwxyz")


def load_model_from_ckpt(ckpt_path, config, device):
    """Load weights from .pt or Lightning .ckpt (matches eval_text8_bpc.py)."""
    model = get_flow_model(config.model, config.encoder).to(device)

    if ckpt_path.endswith(".ckpt"):
        from models.text8_module import Text8Module

        module = Text8Module.load_from_checkpoint(ckpt_path, map_location=device)
        module = module.to(device)
        model.load_state_dict(module.model.state_dict())
        print("[info] loaded weights from .ckpt (non-EMA)")
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        print("[info] loaded weights from .pt")

    model.eval()
    return model


def ids_to_strings(ids: torch.Tensor) -> list[str]:
    """ids: (N, D) integer class indices."""
    rows = ids.tolist()
    return ["".join(TEXT8_CHARS[i] for i in row) for row in rows]


def main():
    parser = argparse.ArgumentParser(description="Sample from a Text8 flow model.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint path: .ckpt or .pt")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/text8_sample_1step.yml",
        help="YAML containing the model recipe and run settings",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--method",
        type=str,
        choices=["euler", "ode"],
        default="euler",
        help="euler: explicit Euler; ode: adaptive solver via torchdiffeq",
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=None,
        help="number of sequences to generate; defaults to config.sample.n_sample",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=None,
        help="number of integration steps; defaults to config.sample.n_step",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="optional: write samples line by line to this UTF-8 text file",
    )
    args = parse_run_args(parser, required=('ckpt',))

    seed_all(args.seed)
    config = load_config(args.config)

    sample_cfg = getattr(config, "sample", None)
    n_sample = args.n_sample
    if n_sample is None:
        n_sample = sample_cfg.n_sample if sample_cfg is not None else 4
    n_steps = args.n_steps
    if n_steps is None:
        n_steps = sample_cfg.n_step if sample_cfg is not None else 256

    print(f"device={args.device}  method={args.method}  n_sample={n_sample}  n_steps={n_steps}")

    model = load_model_from_ckpt(args.ckpt, config, args.device)

    seq_len = None
    if hasattr(config, "datasets") and hasattr(config.datasets, "seq_len"):
        seq_len = int(config.datasets.seq_len)

    with torch.no_grad():
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        t0 = time.perf_counter()
        traj = model.sample(args.method, n_sample, n_steps, args.device)
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.synchronize(args.device)
        t1 = time.perf_counter()

    sample_sec = t1 - t0
    print(
        f"Sampling time: {sample_sec:.4f} s | {n_sample} sequences | "
        f"{sample_sec / n_sample * 1000:.3f} ms/sequence"
    )
    if seq_len is not None:
        n_char = n_sample * seq_len
        print(
            f"  ({seq_len} chars/sequence, {n_char} chars total, "
            f"{sample_sec / n_char * 1000:.4f} ms/char)"
        )

    ids = traj.argmax(dim=-1)
    texts = ids_to_strings(ids)

    for i, t in enumerate(texts):
        print(f"[{i}] {t}")

    if args.out:
        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
