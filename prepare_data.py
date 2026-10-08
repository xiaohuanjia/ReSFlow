#!/usr/bin/env python3
"""Optionally populate data caches before starting training or evaluation."""

import argparse
from pathlib import Path

from datasets.prepare import (GENOME_FILE, prepare_bmnist, prepare_genome_memmap,
                              prepare_promoter_resources, prepare_text8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('bmnist', 'text8', 'promoter', 'all'), default='all')
    parser.add_argument('--data_root', default='./data', help='Parent directory containing task subdirectories.')
    args = parser.parse_args()
    root = Path(args.data_root)
    if args.dataset in ('bmnist', 'all'):
        for split in ('train', 'valid', 'test'):
            prepare_bmnist(root / 'bmnist', split)
    if args.dataset in ('text8', 'all'):
        prepare_text8(root / 'text8')
    if args.dataset in ('promoter', 'all'):
        import pyfaidx
        promoter_root = root / 'promoter'
        prepare_promoter_resources(promoter_root)
        with pyfaidx.Fasta(str(promoter_root / GENOME_FILE)) as genome:
            prepare_genome_memmap(genome, promoter_root / (GENOME_FILE + '.mmap'))
    print(f'[data] {args.dataset} preparation complete in {root}')


if __name__ == '__main__':
    main()
