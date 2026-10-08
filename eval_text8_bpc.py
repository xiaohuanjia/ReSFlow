#!/usr/bin/env python3
"""
Text8 Bits-Per-Character (BPC) evaluation.


Usage:
    python eval_text8_bpc.py --config configs/text8_eval_bpc.yml
"""


import argparse
import math
import os
import sys
from typing import Optional
from configuration import parse_run_args

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets.prepare import prepare_text8  # noqa: E402
from models import get_flow_model  # noqa: E402
from utils import load_config, seed_all  # noqa: E402


class Text8NonOverlap(Dataset):
    """Non-overlapping seq_len slices of the first n_sequences * seq_len chars."""

    def __init__(self, root: str, split: str, vocab_size: int, seq_len: int, n_sequences: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        fname = os.path.join(root, f'{split}.bin')
        prepare_text8(root)
        data = np.memmap(fname, dtype=np.uint16, mode='r')
        max_n = data.size // seq_len
        if n_sequences > max_n:
            print(f'[warn] requested {n_sequences} but only {max_n} available; truncating.')
            n_sequences = max_n
        self.n_sequences = n_sequences
        self.data = data[: n_sequences * seq_len]

    def __len__(self):
        return self.n_sequences

    def __getitem__(self, idx):
        s = idx * self.seq_len
        seq = torch.from_numpy(self.data[s: s + self.seq_len].astype(np.int64))
        seq = F.one_hot(seq, self.vocab_size).float()
        return seq


@torch.no_grad()
def compute_bpc_ode(
    model,
    p1: torch.Tensor,
    s_max: float = 10.0,
    s_min: float = 1e-3,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    log_eps: float = 1e-3,
    n_noise: int = 1,
) -> torch.Tensor:
    """Per-character NLL in nats, shape (B, D); BPC = nll / log(2)."""
    device = p1.device
    nll_acc = torch.zeros(p1.size()[:-1], device=device, dtype=torch.float)

    p1_hat = model.preprocess(p1)

    for _ in range(n_noise):
        noise = model.sample_prior(*p1.size(), device=device, eps=model.eps)
        p0_hat = model.preprocess(noise)

        def ode_fn(t, _):
            _t = (-t).exp()
            pt = model.interpolate(p1_hat, p0_hat, _t)
            vf = model(_t, pt)
            pred_p1 = model.postprocess(model.proj_x(model.exp(pt, vf * _t, model.eps)))
            integrand = -(pred_p1 * p1).sum(-1).clamp(min=log_eps).log()
            return integrand

        nll = odeint(
            ode_fn,
            torch.zeros(p1.size()[:-1], device=device, dtype=torch.float),
            t=torch.linspace(s_min, s_max, 2, device=device, dtype=torch.float),
            method='dopri5',
            atol=atol,
            rtol=rtol,
        )[-1]
        nll_acc = nll_acc + nll

    return nll_acc / n_noise


def load_model_from_ckpt(ckpt_path: str, config, device: str, use_ema: bool = True):
    """Support Lightning .ckpt (with EMA + hparams) and bare .pt."""

    if ckpt_path.endswith('.ckpt'):
        from models.text8_module import Text8Module

        module = Text8Module.load_from_checkpoint(ckpt_path, map_location=device)
        module = module.to(device)
        model = module.model.to(device)

        if use_ema:
            try:
                module.ema.copy_to(model.parameters())
                print('[info] applied EMA weights')
            except Exception as e:
                print(f'[warn] failed to apply EMA, using training weights: {e}')
        else:
            print('[info] EMA skipped, using training weights')
    else:
        if config is None:
            raise ValueError('bare .pt checkpoint requires --config')
        model = get_flow_model(config.model, config.encoder).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if use_ema and 'ema' in ckpt:
                state = ckpt['ema']
                print('[info] using EMA weights from .pt')
            elif 'model' in ckpt:
                state = ckpt['model']
                print('[info] using .pt[model] weights')
            else:
                state = ckpt
                print('[info] using full .pt state_dict')
        else:
            state = ckpt
        model.load_state_dict(state)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def resolve_config(ckpt_path: str, config_path: Optional[str]):
    """Prefer --config; otherwise try to recover from .ckpt hparams."""
    if config_path:
        cfg = load_config(config_path)
        print(f'[info] config: {config_path}')
        return cfg

    if ckpt_path.endswith('.ckpt'):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'hyper_parameters' in ckpt:
            from easydict import EasyDict
            cfg = EasyDict(ckpt['hyper_parameters'])
            print('[info] recovered config from .ckpt hparams')
            return cfg
    raise ValueError('no --config provided and ckpt has no embedded hparams; '
                     'please specify --config explicitly.')


def main():
    parser = argparse.ArgumentParser(
        description='Text8 BPC evaluation (paper B.2 Eq.(45), Dopri5).',
    )
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--config', type=str, default=None,
                        help='Required for .pt or for .ckpt without embedded hparams.')
    parser.add_argument('--data_root', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_sequences', type=int, default=4000,
                        help='Paper protocol: 4000 x 256 = 1,024,000 chars.')
    parser.add_argument('--s_max', type=float, default=10.0)
    parser.add_argument('--s_min', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-5)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--log_eps', type=float, default=1e-3,
                        help='Lower clamp before log inside the integrand.')
    parser.add_argument('--n_noise', type=int, default=1,
                        help='Number of mu_0 Monte-Carlo samples per datum.')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_ema', action='store_true')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_run_args(parser, required=('ckpt',))

    seed_all(args.seed)

    cfg = resolve_config(args.ckpt, args.config)

    ds_cfg = cfg.datasets
    seq_len = int(getattr(ds_cfg, 'seq_len', 256))
    vocab_size = int(getattr(cfg.encoder, 'vocab_size',
                             getattr(ds_cfg, 'vocab_size', 27)))
    data_root = args.data_root or ds_cfg.root

    print('==================== Text8 BPC eval ====================')
    print(f'ckpt        : {args.ckpt}')
    print(f'config      : {args.config or "<auto from ckpt>"}')
    print(f'device      : {args.device}')
    print(f'data root   : {data_root}  split={args.split}')
    print(f'seq_len     : {seq_len}    vocab_size={vocab_size}')
    print(f'#sequences  : {args.n_sequences}  (= {args.n_sequences * seq_len:,} chars)')
    print(f'batch_size  : {args.batch_size}')
    print(f's range     : [{args.s_min}, {args.s_max}]   atol={args.atol}  rtol={args.rtol}')
    print(f'n_noise     : {args.n_noise}     log_eps={args.log_eps}')
    print(f'use_ema     : {not args.no_ema}')
    print('========================================================')

    dataset = Text8NonOverlap(
        root=data_root, split=args.split,
        vocab_size=vocab_size, seq_len=seq_len,
        n_sequences=args.n_sequences,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=False, pin_memory=True,
    )

    model = load_model_from_ckpt(
        args.ckpt, cfg, args.device, use_ema=not args.no_ema,
    )

    total_nll = 0.0
    total_chars = 0
    log2 = math.log(2.0)

    pbar = tqdm(loader, desc='BPC eval')
    for batch in pbar:
        p1 = batch.to(args.device, non_blocking=True)
        nll_per_char = compute_bpc_ode(
            model, p1,
            s_max=args.s_max, s_min=args.s_min,
            atol=args.atol, rtol=args.rtol,
            log_eps=args.log_eps, n_noise=args.n_noise,
        )
        total_nll += nll_per_char.double().sum().item()
        total_chars += nll_per_char.numel()
        running_bpc = (total_nll / total_chars) / log2
        pbar.set_postfix({'BPC': f'{running_bpc:.4f}'})

    bpc = (total_nll / total_chars) / log2
    print('--------------------------------------------------------')
    print(f'evaluated chars : {total_chars:,}')
    print(f'mean NLL (nat)  : {total_nll / total_chars:.6f}')
    print(f'BPC             : {bpc:.6f}')
    print('--------------------------------------------------------')


if __name__ == '__main__':
    main()
