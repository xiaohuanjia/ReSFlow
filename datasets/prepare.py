"""Automatic preparation shared by dataset loaders and standalone scripts."""

import gzip
import os
from pathlib import Path
import pickle
import shutil
import tarfile
import zipfile

import numpy as np

from ._download import data_lock, download_file, nonempty


BMNIST_URL = ('https://www.cs.toronto.edu/~larocheh/public/datasets/'
              'binarized_mnist/binarized_mnist_{}.amat')
TEXT8_URL = 'https://mattmahoney.net/dc/text8.zip'
PROMOTER_URL = 'https://zenodo.org/api/records/7943307/files/data.tar.gz/content'
PROMOTER_SIZE = 10162295808
PROMOTER_MD5 = 'aa53f13ddcf1474962548bc02cbb9b93'
GENOME_FILE = 'Homo_sapiens.GRCh38.dna.primary_assembly.fa'
TSS_FILE = 'FANTOM_CAT.lv3_robust.tss.sortedby_fantomcage.hg38.v4.tsv'
SIGNAL_FILES = ('agg.plus.bw.bedgraph.bw', 'agg.minus.bw.bedgraph.bw')
BLACKLIST_FILES = ('fantom.blacklist8.plus.bed.gz', 'fantom.blacklist8.minus.bed.gz')
SEI_FILES = ('best.sei.model.pth.tar', 'target.sei.names')
SELENE_URL = 'https://raw.githubusercontent.com/FunctionLab/selene/master/selene_sdk/sequences/data/{}'


def prepare_bmnist(root, split):
    if split not in ('train', 'valid', 'test'):
        raise ValueError(f'Unknown BMNIST split: {split}')
    root = Path(root)
    source = root / f'binarized_mnist_{split}.amat'
    cache = root / f'binarized_mnist_{split}.npy'
    with data_lock(root / '.bmnist.lock'):
        if nonempty(cache):
            try:
                data = np.load(cache, mmap_mode='r', allow_pickle=False)
                if data.ndim == 2 and data.shape[1] == 784 and data.dtype == np.float32:
                    return cache
            except (OSError, ValueError):
                pass
        download_file(BMNIST_URL.format(split), source)
        data = np.loadtxt(source, dtype=np.float32, ndmin=2)
        if data.shape[1] != 784 or not np.isin(data, (0, 1)).all():
            raise ValueError(f'Invalid binarized MNIST file: {source}')
        partial = Path(str(cache) + '.part')
        with partial.open('wb') as handle:
            np.save(handle, data, allow_pickle=False)
        os.replace(partial, cache)
    return cache


def prepare_text8(root):
    root = Path(root)
    with data_lock(root / '.text8.lock'):
        outputs = [root / f'{split}.bin' for split in ('train', 'valid', 'test')]
        if all(nonempty(path) and path.stat().st_size % 2 == 0 for path in outputs) and nonempty(root / 'meta.pkl'):
            return
        raw = root / 'text8'
        if not nonempty(raw):
            archive = download_file(TEXT8_URL, root / 'text8.zip')
            with zipfile.ZipFile(archive) as handle:
                # Only the corpus is extracted, regardless of other archive paths.
                with handle.open('text8') as source, Path(str(raw) + '.part').open('wb') as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            os.replace(str(raw) + '.part', raw)
        data = np.frombuffer(raw.read_bytes(), dtype=np.uint8)
        alphabet = b' abcdefghijklmnopqrstuvwxyz'
        if not np.isin(data, np.frombuffer(alphabet, dtype=np.uint8)).all():
            raise ValueError(f'Unexpected characters in Text8 corpus: {raw}')
        lookup = np.zeros(256, dtype=np.uint16)
        lookup[np.frombuffer(alphabet, dtype=np.uint8)] = np.arange(27, dtype=np.uint16)
        boundaries = (0, int(len(data) * .9), int(len(data) * .95), len(data))
        for index, path in enumerate(outputs):
            partial = Path(str(path) + '.part')
            with partial.open('wb') as handle:
                for start in range(boundaries[index], boundaries[index + 1], 1024 * 1024):
                    lookup[data[start:min(start + 1024 * 1024, boundaries[index + 1])]].tofile(handle)
            os.replace(partial, path)
        chars = alphabet.decode('ascii')
        meta = {'vocab_size': 27, 'stoi': {c: i for i, c in enumerate(chars)},
                'itos': {i: c for i, c in enumerate(chars)}}
        partial = root / 'meta.pkl.part'
        with partial.open('wb') as handle:
            pickle.dump(meta, handle)
        os.replace(partial, root / 'meta.pkl')
        print(f'[data] Text8 splits and vocabulary prepared in {root}', flush=True)


def _extract_resources(archive, destinations):
    """Flatten selected regular files without trusting archive paths/links."""
    pending = {Path(name).name: Path(target) for name, target in destinations.items()}
    with tarfile.open(archive, mode='r|gz') as handle:
        for member in handle:
            basename = Path(member.name).name
            compressed = basename.endswith('.fa.gz') and basename[:-3] in pending
            key = basename[:-3] if compressed else basename
            if not member.isfile() or key not in pending:
                continue
            destination = pending.pop(key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = Path(str(destination) + '.part')
            with handle.extractfile(member) as source, partial.open('wb') as target:
                if compressed:
                    with gzip.GzipFile(fileobj=source) as uncompressed:
                        shutil.copyfileobj(uncompressed, target, 8 * 1024 * 1024)
                else:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
            os.replace(partial, destination)
            print(f'[data] Extracted {destination.name}', flush=True)
            if not pending:
                break
    if pending:
        raise FileNotFoundError(f'Required files absent from {archive}: {", ".join(sorted(pending))}')


def prepare_promoter_resources(root, ref_file=GENOME_FILE, tsses_file=TSS_FILE,
                               fantom_files=SIGNAL_FILES, fantom_blacklist_files=BLACKLIST_FILES,
                               sei_model_path=None, sei_feat_path=None):
    """Retrieve missing benchmark/Sei resources and prepare BED/FASTA indexes."""
    root = Path(root)
    destinations = {GENOME_FILE: root / ref_file, TSS_FILE: root / tsses_file}
    destinations.update({name: root / target for name, target in zip(SIGNAL_FILES, fantom_files)})
    destinations.update({name: root / target for name, target in zip(BLACKLIST_FILES, fantom_blacklist_files)})
    destinations[SEI_FILES[0]] = Path(sei_model_path) if sei_model_path else root / SEI_FILES[0]
    destinations[SEI_FILES[1]] = Path(sei_feat_path) if sei_feat_path else root / SEI_FILES[1]
    with data_lock(root / '.promoter.lock'):
        _ensure_promoter_files(root, destinations)
        # pyfaidx creates/rebuilds the FASTA index, including custom FASTA paths.
        import pyfaidx
        with pyfaidx.Fasta(str(root / ref_file)):
            pass
        for filename in fantom_blacklist_files:
            ensure_tabix_index(root / filename)
        ensure_blacklist(root, 'hg38')


def _ensure_promoter_files(root, destinations):
    missing = {name: target for name, target in destinations.items() if not nonempty(target)}
    if missing:
        archive = download_file(PROMOTER_URL, root / '.downloads' / 'data.tar.gz',
                                size=PROMOTER_SIZE, md5=PROMOTER_MD5)
        _extract_resources(archive, missing)


def prepare_sei_resources(root, model_path, feature_path):
    root = Path(root)
    with data_lock(root / '.promoter.lock'):
        _ensure_promoter_files(root, {SEI_FILES[0]: Path(model_path), SEI_FILES[1]: Path(feature_path)})


def ensure_tabix_index(filename):
    with data_lock(str(filename) + '.index.lock'):
        _ensure_tabix_index(filename)


def _ensure_tabix_index(filename):
    filename = Path(filename)
    if nonempty(str(filename) + '.tbi'):
        return
    import pysam
    plain = Path(str(filename) + '.indexing.bed')
    compressed = Path(str(filename) + '.indexing.gz')
    try:
        with gzip.open(filename, 'rb') as source, plain.open('wb') as target:
            shutil.copyfileobj(source, target, 8 * 1024 * 1024)
        pysam.tabix_compress(str(plain), str(compressed), force=True)
        pysam.tabix_index(str(compressed), preset='bed', force=True)
        os.replace(compressed, filename)
        os.replace(str(compressed) + '.tbi', str(filename) + '.tbi')
    finally:
        plain.unlink(missing_ok=True)
        compressed.unlink(missing_ok=True)
        Path(str(compressed) + '.tbi').unlink(missing_ok=True)


def ensure_blacklist(root, assembly):
    filename = {'hg38': 'hg38.blacklist.bed.gz',
                'hg19': 'hg19_blacklist_ENCFF001TDO.bed.gz'}[assembly]
    path = Path(root) / filename
    with data_lock(str(path) + '.prepare.lock'):
        download_file(SELENE_URL.format(filename), path)
        ensure_tabix_index(path)
    return path


def prepare_genome_memmap(genome, path, bases=('A', 'C', 'G', 'T')):
    """Build the existing (4, genome length) float32 cache in bounded memory."""
    path = Path(path)
    chromosomes = sorted(genome.keys())
    lengths = [len(genome[chrom]) for chrom in chromosomes]
    total = sum(lengths)
    shape = (len(bases), total)
    expected = len(bases) * total * np.dtype(np.float32).itemsize
    with data_lock(str(path) + '.lock'):
        if not path.is_file() or path.stat().st_size != expected:
            partial = Path(str(path) + '.part')
            table = np.full((256, len(bases)), 1. / len(bases), dtype=np.float32)
            for index, base in enumerate(bases):
                for letter in (base.upper(), base.lower()):
                    table[ord(letter)] = 0
                    table[ord(letter), index] = 1
            cache = np.memmap(partial, dtype=np.float32, mode='w+', shape=shape)
            offset = 0
            for chromosome, length in zip(chromosomes, lengths):
                print(f'[data] Encoding genome {chromosome}', flush=True)
                for start in range(0, length, 1024 * 1024):
                    end = min(start + 1024 * 1024, length)
                    sequence = genome[chromosome][start:end].seq.encode('ascii')
                    cache[:, offset + start:offset + end] = table[np.frombuffer(sequence, dtype=np.uint8)].T
                offset += length
            cache.flush()
            del cache
            os.replace(partial, path)
    return np.memmap(path, dtype=np.float32, mode='r', shape=shape)
