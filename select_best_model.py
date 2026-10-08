#!/usr/bin/env python3
"""
Evaluate all checkpoints in an experiment directory and select the best model
by FID. Supports repeating the evaluation N times for stable estimates with
variance.

Usage:
    python select_best_model.py --config configs/bmnist_select_1step.yml
"""
import argparse
import glob
import json
import os
from configuration import parse_run_args
from utils import seed_all

import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

from models import get_flow_model
from datasets import get_dataset
from evaluation.fid import get_fid, calculate_activation_statistics, InceptionV3

VALID_FID_STAT_PATH = 'bmnist_valid_fid.npz'


def ensure_fid_stats(config, device, fid_stat_path=VALID_FID_STAT_PATH):
    """Precompute and cache validation-set FID statistics if not already on disk."""
    if os.path.exists(fid_stat_path):
        return
    print(f'FID stats not found at {fid_stat_path}, computing from validation set...')
    from torch.utils.data import DataLoader
    _, valid_set, _ = get_dataset(config.datasets)
    loader = DataLoader(valid_set, batch_size=512)
    gt = []
    for samples, *_ in tqdm(loader, desc='Loading valid data'):
        img = (samples[..., 0] > samples[..., 1]).float().view(-1, 1, 28, 28).expand(-1, 3, -1, -1)
        gt.append(img)
    gt = torch.cat(gt, dim=0)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()
    mu, sigma = calculate_activation_statistics(gt, inception, device, batch_size=256)
    np.savez(fid_stat_path, mu=mu, sigma=sigma)
    print(f'FID stats saved to {fid_stat_path}')


def compute_fid_for_checkpoint(ckpt_path, model, device, total_sample, batch_size, n_steps, method='euler'):
    """Load a checkpoint into `model`, sample, and return the FID score."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    samples = []
    with torch.no_grad():
        for start in tqdm(range(0, total_sample, batch_size), desc='Sampling', leave=False):
            take = min(batch_size, total_sample - start)
            s = model.sample(method, n_sample=take, n_steps=n_steps, device=device)
            samples.append(s.detach().cpu())

    samples = torch.cat(samples, dim=0)
    img = (samples[..., 0] > samples[..., 1]).float().view(-1, 1, 28, 28).expand(-1, 3, -1, -1)
    fid = get_fid(img, device, batch_size=256, dataset='bmnist_valid')
    return fid


def list_checkpoints(exp_dir, ckpt_pattern):
    ckpt_files = sorted(glob.glob(os.path.join(exp_dir, ckpt_pattern)))
    ckpt_files = [f for f in ckpt_files if f.endswith('.pt')]
    if not ckpt_files:
        raise FileNotFoundError(f'No .pt files matching "{ckpt_pattern}" in {exp_dir}')
    return ckpt_files


def load_config(base_ckpt, ckpt_files):
    if base_ckpt:
        return torch.load(base_ckpt, map_location='cpu', weights_only=False)
    for f in ckpt_files:
        c = torch.load(f, map_location='cpu', weights_only=False)
        if isinstance(c, dict) and 'config' in c:
            return c
    raise RuntimeError('None of the checkpoints contain a config. '
                       'Please specify --base_ckpt explicitly.')


def evaluate_once(model, ckpt_files, device, total_sample, batch_size, n_steps, method):
    """Evaluate every checkpoint once and return a sorted list of results."""
    results = []
    for ckpt_path in ckpt_files:
        name = os.path.basename(ckpt_path)
        print(f'\n{"="*60}')
        print(f'Evaluating: {ckpt_path}')
        print(f'{"="*60}')
        fid = compute_fid_for_checkpoint(
            ckpt_path, model, device,
            total_sample, batch_size, n_steps, method,
        )
        print(f'  FID = {fid:.4f}')
        results.append({'checkpoint': ckpt_path, 'name': name, 'fid': fid})

    results.sort(key=lambda r: r['fid'])
    return results


def print_run_summary(results):
    print(f'\n{"="*60}')
    print('Results (sorted by FID, lower is better):')
    print(f'{"="*60}')
    for i, r in enumerate(results):
        marker = ' <-- BEST' if i == 0 else ''
        print(f'  {r["fid"]:>10.4f}  {r["checkpoint"]}{marker}')
    best = results[0]
    print(f'\nBest model: {best["checkpoint"]}  (FID = {best["fid"]:.4f})')


def aggregate_runs(all_runs):
    ckpt_fids = {}
    for run in all_runs:
        for entry in run:
            ckpt_fids.setdefault(entry['checkpoint'], []).append(entry['fid'])

    summary = []
    for ckpt, fids in sorted(ckpt_fids.items(), key=lambda kv: np.mean(kv[1])):
        summary.append({
            'checkpoint': ckpt,
            'fid_mean': float(np.mean(fids)),
            'fid_std': float(np.std(fids)),
            'fid_runs': [float(x) for x in fids],
        })
    return summary


def print_aggregate_summary(summary, all_runs):
    n = len(all_runs)
    print(f'\n{"="*60}')
    print(f'Aggregated results over {n} runs (mean ± std):')
    print(f'{"="*60}')
    for s in summary:
        print(f'  {s["fid_mean"]:>10.4f} ± {s["fid_std"]:<8.4f}  {s["checkpoint"]}')

    best = summary[0]
    print(f'\nBest model (by mean FID): {best["checkpoint"]}  '
          f'({best["fid_mean"]:.4f} ± {best["fid_std"]:.4f})')

    print(f'\n{"="*60}')
    print('Per-run best:')
    print(f'{"="*60}')
    for i, run in enumerate(all_runs):
        b = run[0]
        print(f'  Run {i+1}: {b["checkpoint"]}  (FID = {b["fid"]:.4f})')


def main():
    parser = argparse.ArgumentParser(description='Select the best checkpoint by FID.')
    parser.add_argument('exp_dir', type=str, nargs='?', help='Experiment directory under logs/')
    parser.add_argument('--config', type=str, default=None, help='YAML with checkpoint-selection run settings')
    parser.add_argument('--base_ckpt', type=str, default=None,
                        help='Base checkpoint for reading config. If omitted, '
                             'uses the config from the first .pt file found in exp_dir.')
    parser.add_argument('--ckpt_pattern', type=str, default='*.pt',
                        help='Glob pattern for checkpoint files (default: *.pt)')
    parser.add_argument('--total_sample', type=int, default=1000,
                        help='Number of samples to generate per checkpoint')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for sampling')
    parser.add_argument('--n_steps', type=int, default=1,
                        help='Number of steps for sampling (Euler steps or ODE trajectory points)')
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'ode'],
                        help='Sampling method: euler or ode (default: euler)')
    parser.add_argument('-n', '--n_runs', type=int, default=1,
                        help='Repeat the full evaluation N times and aggregate (default: 1)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parse_run_args(parser, required=('exp_dir',))
    seed_all(args.seed)

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt_files = list_checkpoints(args.exp_dir, args.ckpt_pattern)
    print(f'Found {len(ckpt_files)} checkpoints in {args.exp_dir}')

    config_ckpt = load_config(args.base_ckpt, ckpt_files)
    config = EasyDict(config_ckpt['config'])
    model = get_flow_model(config.model, config.encoder).to(device)
    ensure_fid_stats(config, device)

    all_runs = []
    for i in range(args.n_runs):
        if args.n_runs > 1:
            print(f'\n{"#"*60}')
            print(f'  Run {i+1}/{args.n_runs}')
            print(f'{"#"*60}')
        results = evaluate_once(
            model, ckpt_files, device,
            args.total_sample, args.batch_size, args.n_steps, args.method,
        )
        print_run_summary(results)
        all_runs.append(results)

        out_path = os.path.join(args.exp_dir, 'fid_selection.json')
        with open(out_path, 'w') as f:
            json.dump({
                'best_checkpoint': results[0]['checkpoint'],
                'best_fid': results[0]['fid'],
                'total_sample': args.total_sample,
                'n_steps': args.n_steps,
                'all_results': results,
            }, f, indent=2)
        print(f'Results saved to {out_path}')

    if args.n_runs > 1:
        summary = aggregate_runs(all_runs)
        print_aggregate_summary(summary, all_runs)

        best = summary[0]
        out_path = os.path.join(args.exp_dir, 'fid_selection_repeat.json')
        with open(out_path, 'w') as f:
            json.dump({
                'n_runs': args.n_runs,
                'best_checkpoint': best['checkpoint'],
                'best_fid_mean': best['fid_mean'],
                'best_fid_std': best['fid_std'],
                'per_checkpoint': summary,
                'per_run': [{'best': r[0]['checkpoint'], 'fid': r[0]['fid']} for r in all_runs],
            }, f, indent=2)
        print(f'\nAggregated results saved to {out_path}')


if __name__ == '__main__':
    main()
