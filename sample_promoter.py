#!/usr/bin/env python3
"""
Promoter sampling script (generation + timing only, no SP-MSE).

Usage examples:
  python sample_promoter.py --config configs/promoter_sample_1step.yml
  python sample_promoter.py --config configs/promoter_sample_ode.yml

Notes:
  - Promoter is conditional generation; `signal` is taken from the dataset as cond.
  - By default the first n_sample signals are taken in order from the valid split
    (or the split selected by run.split); set run.shuffle to draw different samples
    on each repeat.
  - Only model.sample(...) is timed; CUDA is synchronized before and after.
"""

import argparse
import os
import sys
import time
from configuration import parse_run_args
from typing import Optional

import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets import get_dataset  # noqa: E402
from models import get_flow_model  # noqa: E402
from utils import load_config, recursive_to_device, seed_all  # noqa: E402

DNA_CHARS = ['A', 'C', 'G', 'T']


def load_model_from_ckpt(ckpt_path: str, config, device: str):
    """Same loading logic as sample_text8.py / eval_promoter_sp_mse.py."""
    model = get_flow_model(config.model, config.encoder).to(device)

    if ckpt_path.endswith('.ckpt'):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('state_dict', ckpt)
        cleaned = {}
        prefix = 'model.'
        for k, v in state.items():
            cleaned[k[len(prefix):] if k.startswith(prefix) else k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f'[Info] missing keys: {len(missing)} (e.g. {missing[:3]})')
        if unexpected:
            print(f'[Info] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})')
        print('[Info] Loaded weights from .ckpt (non-EMA)')
    else:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state)
        print('[Info] Loaded weights from .pt')

    model.eval()
    return model


def collect_cond_args(dataset, n_sample: int, batch_size: int, device: str,
                      shuffle: bool, seed: int):
    """Take up to n_sample cond_args entries from dataset (sequential or after shuffle)."""
    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=4, generator=g if shuffle else None,
    )
    cond_acc: Optional[list] = None
    n = 0
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError('promoter dataset batch must be (x, signal)')
        _, *cond_args = batch
        cond_args = recursive_to_device(cond_args, device)
        take = min(n_sample - n, cond_args[0].size(0))
        if take <= 0:
            break
        if cond_acc is None:
            cond_acc = [c[:take] for c in cond_args]
        else:
            for i in range(len(cond_args)):
                cond_acc[i] = torch.cat([cond_acc[i], cond_args[i][:take]], dim=0)
        n += take
        if n >= n_sample:
            break
    if cond_acc is None or n == 0:
        raise RuntimeError('Failed to collect cond_args from dataset; check --split / data')
    return tuple(cond_acc), n


def ids_to_dna(ids: torch.Tensor):
    rows = ids.tolist()
    return [''.join(DNA_CHARS[int(b)] for b in row) for row in rows]


def time_sample(model, method: str, n_sample: int, n_steps: int, device: str,
                cond_args, warmup: int, repeat: int):
    """Time model.sample(...); returns (last traj, [seconds per repeat])."""
    is_cuda = torch.cuda.is_available() and str(device).startswith('cuda')

    for _ in range(warmup):
        with torch.no_grad():
            _ = model.sample(method, n_sample, n_steps, device, *cond_args)
    if is_cuda:
        torch.cuda.synchronize(device)

    secs = []
    traj = None
    for _ in range(repeat):
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            traj = model.sample(method, n_sample, n_steps, device, *cond_args)
        if is_cuda:
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        secs.append(t1 - t0)
    return traj, secs


def main():
    parser = argparse.ArgumentParser(description='Promoter sampling + timing')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint path (.pt or .ckpt)')
    parser.add_argument('--config', type=str, default='configs/promoter_sample_1step.yml')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, choices=['ode', 'euler'], default='ode',
                        help='sampling method (default: ode)')
    parser.add_argument('--n_sample', type=int, default=16, help='number of sequences to generate')
    parser.add_argument('--n_steps', type=int, default=None,
                        help='integration steps; default = config.visualizer.n_step (300)')
    parser.add_argument('--split', type=str, default='valid',
                        choices=['train', 'valid', 'test'],
                        help='split to take cond signals from (default: valid)')
    parser.add_argument('--shuffle', action='store_true',
                        help='shuffle the dataset before taking (default: take first n_sample in order)')
    parser.add_argument('--warmup', type=int, default=1, help='warmup repeats before timing')
    parser.add_argument('--repeat', type=int, default=1, help='timed repeats (mean / std reported)')
    parser.add_argument('--print_seq', type=int, default=4,
                        help='number of generated sequences to print (0 disables)')
    parser.add_argument('--print_seq_len', type=int, default=120,
                        help='max chars to print per sequence (0 = full)')
    parser.add_argument('--fasta_out', type=str, default=None,
                        help='optional: write generated sequences to this FASTA file')
    args = parse_run_args(parser, required=('ckpt',))

    seed_all(args.seed)
    config = load_config(args.config)

    n_steps = args.n_steps
    if n_steps is None:
        n_steps = (
            getattr(config, 'visualizer', {}).get('n_step', 300)
            if hasattr(config, 'visualizer') else 300
        )

    print(f'device={args.device}  method={args.method}  '
          f'n_sample={args.n_sample}  n_steps={n_steps}  '
          f'split={args.split}  shuffle={args.shuffle}  '
          f'warmup={args.warmup}  repeat={args.repeat}')

    print('Loading dataset...')
    train_set, valid_set, test_set = get_dataset(config.datasets)
    split_map = {'train': train_set, 'valid': valid_set, 'test': test_set}
    dataset = split_map[args.split]
    print(f'Dataset({args.split}) size: {len(dataset)}')

    print('Building model + loading ckpt...')
    model = load_model_from_ckpt(args.ckpt, config, args.device)

    print('Collecting cond signals from dataset...')
    cond_args, n_got = collect_cond_args(
        dataset, args.n_sample,
        batch_size=min(args.n_sample, max(1, args.n_sample)),
        device=args.device, shuffle=args.shuffle, seed=args.seed,
    )
    if n_got != args.n_sample:
        print(f'[warn] only got {n_got} cond entries (requested {args.n_sample}); '
              f'sampling with the actual count')
    n_sample_eff = n_got

    print('Sampling + timing...')
    traj, secs = time_sample(
        model, args.method, n_sample_eff, n_steps, args.device,
        cond_args, args.warmup, args.repeat,
    )
    seq_len = traj.size(1) if traj.dim() >= 2 else 0

    import statistics
    mean_s = statistics.fmean(secs)
    std_s = statistics.pstdev(secs) if len(secs) > 1 else 0.0
    print('\n===== sampling time =====')
    print(f'per run: ' + ', '.join(f'{s:.4f}s' for s in secs))
    print(f'mean: {mean_s:.4f} s  +/- {std_s:.4f} s  (repeat={len(secs)})')
    print(f'      | {n_sample_eff} seqs | {mean_s / n_sample_eff * 1000:.3f} ms/seq')
    if seq_len > 0:
        n_char = n_sample_eff * seq_len
        print(f'      | {seq_len} bases/seq, {n_char} bases total, '
              f'{mean_s / n_char * 1000:.4f} ms/base')

    ids = traj.argmax(-1).cpu()
    seqs = ids_to_dna(ids)

    if args.print_seq > 0:
        n_show = min(args.print_seq, len(seqs))
        print(f'\n===== generated sequences (first {n_show}) =====')
        for i in range(n_show):
            s = seqs[i]
            disp = s if args.print_seq_len <= 0 else (
                s if len(s) <= args.print_seq_len else s[:args.print_seq_len] + '...'
            )
            print(f'[{i}] len={len(s)}  {disp}')

    if args.fasta_out:
        out_path = os.path.abspath(args.fasta_out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            for i, s in enumerate(seqs):
                f.write(f'>sample_{i} method={args.method} n_steps={n_steps} len={len(s)}\n')
                f.write(s + '\n')
        print(f'\nWrote {len(seqs)} sequences to FASTA: {out_path}')


if __name__ == '__main__':
    main()
