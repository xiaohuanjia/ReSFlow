"""Run with: python -m unittest discover -s tests -v."""
import ast
from contextlib import contextmanager
import gzip
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import pickle
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from datasets import BinaryMNIST, Text8Dataset
from datasets import prepare
from datasets._download import download_file


@contextmanager
def http_fixture(payload, *, ignore_range=False, interrupt=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            header = self.headers.get('Range')
            requests.append(header)
            offset = int(header.split('=')[1].split('-')[0]) if header and not ignore_range else 0
            self.send_response(206 if header and not ignore_range else 200)
            self.send_header('Content-Length', str(len(payload) - offset))
            if header and not ignore_range:
                self.send_header('Content-Range', f'bytes {offset}-{len(payload)-1}/{len(payload)}')
            self.end_headers()
            if interrupt and len(requests) == 1:
                self.wfile.write(payload[:len(payload)//2])
            else:
                self.wfile.write(payload[offset:])
            self.wfile.flush()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/data', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def archive_bytes(files):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as handle:
        for name, payload in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))
    return output.getvalue()


class DownloadTests(unittest.TestCase):
    def test_resume_checksum_and_reuse(self):
        payload = b'0123456789' * 100
        with tempfile.TemporaryDirectory() as directory, http_fixture(payload) as (url, requests):
            path = Path(directory) / 'data'
            Path(str(path) + '.part').write_bytes(payload[:123])
            download_file(url, path, size=len(payload), md5=hashlib.md5(payload).hexdigest())
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(requests, ['bytes=123-'])
            download_file(url, path, size=len(payload), md5=hashlib.md5(payload).hexdigest())
            self.assertEqual(len(requests), 1)
            self.assertFalse(Path(str(path) + '.part').exists())

    def test_server_ignoring_range_restarts(self):
        payload = b'complete response'
        with tempfile.TemporaryDirectory() as directory, http_fixture(payload, ignore_range=True) as (url, _):
            path = Path(directory) / 'data'
            Path(str(path) + '.part').write_bytes(b'partial')
            download_file(url, path)
            self.assertEqual(path.read_bytes(), payload)

    def test_interrupted_transfer_retries(self):
        payload = b'x' * 10000
        with tempfile.TemporaryDirectory() as directory, http_fixture(payload, interrupt=True) as (url, requests):
            path = Path(directory) / 'data'
            with patch('datasets._download.time.sleep'):
                download_file(url, path, size=len(payload), md5=hashlib.md5(payload).hexdigest())
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(requests, [None, 'bytes=5000-'])

    def test_wrong_checksum_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory, http_fixture(b'wrong') as (url, _):
            path = Path(directory) / 'data'
            with self.assertRaises(RuntimeError):
                download_file(url, path, md5=hashlib.md5(b'right').hexdigest(), retries=1)
            self.assertFalse(path.exists())

    def test_concurrent_downloads_share_cache(self):
        payload = b'data' * 10000
        with tempfile.TemporaryDirectory() as directory, http_fixture(payload) as (url, requests):
            path = Path(directory) / 'data'
            failures = []
            def run():
                try:
                    download_file(url, path, size=len(payload))
                except Exception as error:
                    failures.append(error)
            threads = [threading.Thread(target=run) for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)
                self.assertFalse(thread.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(len(requests), 1)


class DatasetTests(unittest.TestCase):
    def test_bmnist_download_preprocess_and_loader(self):
        row = np.arange(784) % 2
        payload = (' '.join(map(str, row)) + '\n').encode() * 2
        with tempfile.TemporaryDirectory() as directory, http_fixture(payload) as (url, requests):
            with patch.object(prepare, 'BMNIST_URL', url):
                dataset = BinaryMNIST(directory, 'train')
                cache = prepare.prepare_bmnist(directory, 'train')
                self.assertEqual(len(dataset), 2)
                np.testing.assert_array_equal(dataset[0][0].numpy()[:, 0], row)
                np.testing.assert_array_equal(dataset[0][0].numpy().sum(-1), np.ones(784))
                self.assertEqual(np.load(cache).dtype, np.float32)
                self.assertEqual(len(requests), 1)

    def test_text8_download_split_and_recover_missing_output(self):
        corpus = b' abcdefghijklmnopqrstuvwxyz' * 100
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w') as archive:
            archive.writestr('text8', corpus)
            archive.writestr('../outside', 'not extracted')
        with tempfile.TemporaryDirectory() as directory, http_fixture(output.getvalue()) as (url, requests):
            root = Path(directory)
            with patch.object(prepare, 'TEXT8_URL', url):
                dataset = Text8Dataset(directory, 'train', seq_len=27)
                self.assertEqual(dataset[0][0].shape, (27, 27))
                sizes = [np.fromfile(root / (split + '.bin'), np.uint16).size for split in ('train', 'valid', 'test')]
                self.assertEqual(sizes, [2430, 135, 135])
                meta = pickle.loads((root / 'meta.pkl').read_bytes())
                self.assertEqual(meta['stoi'][' '], 0)
                self.assertEqual(meta['stoi']['z'], 26)
                expected = (root / 'valid.bin').read_bytes()
                (root / 'valid.bin').unlink()
                prepare.prepare_text8(root)
                self.assertEqual((root / 'valid.bin').read_bytes(), expected)
                self.assertEqual(len(requests), 1)
                self.assertFalse((root.parent / 'outside').exists())

    def test_text8_existing_zip_without_corpus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root / 'text8.zip', 'w') as archive:
                archive.writestr('text8', b' abcdefghijklmnopqrstuvwxyz' * 20)
            with patch('datasets._download.urllib.request.urlopen', side_effect=AssertionError('Unexpected network')):
                prepare.prepare_text8(root)
            self.assertTrue((root / 'test.bin').exists())

    def test_invalid_text8_characters_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'text8').write_bytes(b'ABC invalid')
            with self.assertRaises(ValueError):
                prepare.prepare_text8(root)
            self.assertFalse((root / 'train.bin').exists())

    def test_bpc_loader_prepares_missing_data(self):
        # Exercise the actual standalone dataset without importing CUDA/model code.
        path = Path(__file__).resolve().parents[1] / 'eval_text8_bpc.py'
        tree = ast.parse(path.read_text())
        node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Text8NonOverlap')
        import os
        import torch
        import torch.nn.functional as F
        from torch.utils.data import Dataset
        scope = dict(Dataset=Dataset, np=np, torch=torch, F=F, os=os, prepare_text8=prepare.prepare_text8)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'text8').write_bytes(b' abcdefghijklmnopqrstuvwxyz' * 100)
            dataset = scope['Text8NonOverlap'](directory, 'test', 27, 27, 3)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset[0].shape, (27, 27))


class PromoterTests(unittest.TestCase):
    def test_selected_extraction_custom_paths_and_gzip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'data.tar.gz'
            archive.write_bytes(archive_bytes({'nested/genome.fa.gz': gzip.compress(b'>chr1\nACGT\n'),
                                               '../outside': b'ignored', 'nested/model': b'weights'}))
            prepare._extract_resources(archive, {'genome.fa': root / 'custom.fa', 'model': root / 'custom.pt'})
            self.assertEqual((root / 'custom.fa').read_bytes(), b'>chr1\nACGT\n')
            self.assertEqual((root / 'custom.pt').read_bytes(), b'weights')
            self.assertFalse((root.parent / 'outside').exists())

    def test_missing_archive_member_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'data.tar.gz'
            archive.write_bytes(archive_bytes({'some/file': b'data'}))
            with self.assertRaises(FileNotFoundError):
                prepare._extract_resources(archive, {'required': root / 'required'})

    def test_full_promoter_preparation_and_dataset(self):
        import pyBigWig
        import pysam
        from datasets.promoter import TSSDatasetS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genome = ''.join(f'>{chrom}\n' + ('ACGTN' * 500) + '\n' for chrom in ('chr1', 'chr8', 'chr9', 'chr10')).encode()
            tsses = b'chr\tTSS\tstrand\nchr1\t1250\t+\nchr8\t1250\t+\nchr9\t1250\t-\nchr10\t1250\t+\n'
            bw = root / 'fixture.bw'
            handle = pyBigWig.open(str(bw), 'w')
            handle.addHeader([(chrom, 2500) for chrom in ('chr1', 'chr8', 'chr9', 'chr10')])
            for chrom in ('chr1', 'chr8', 'chr9', 'chr10'):
                handle.addEntries(chrom, 0, values=[1.] * 2500, span=1, step=1)
            handle.close()
            files = {f'data/promoter_design/{prepare.GENOME_FILE}': genome,
                     f'data/promoter_design/{prepare.TSS_FILE}': tsses}
            for name in prepare.SIGNAL_FILES:
                files[f'data/promoter_design/{name}'] = bw.read_bytes()
            for name in prepare.BLACKLIST_FILES:
                files[f'data/promoter_design/{name}'] = gzip.compress(b'chr1\t0\t10\n')
            for name in prepare.SEI_FILES:
                files[f'data/promoter_design/{name}'] = b'fixture'
            payload = archive_bytes(files)
            with http_fixture(payload) as (url, requests), http_fixture(gzip.compress(b'chr1\t0\t10\n')) as (blacklist_url, _):
                with patch.multiple(prepare, PROMOTER_URL=url, PROMOTER_SIZE=len(payload), PROMOTER_MD5=hashlib.md5(payload).hexdigest(), SELENE_URL=blacklist_url):
                    train = TSSDatasetS(str(root), 'train', seqlength=32)
                    valid = TSSDatasetS(str(root), 'valid', seqlength=32)
                    test = TSSDatasetS(str(root), 'test', seqlength=32)
                    self.assertEqual([len(train), len(valid), len(test)], [1, 1, 2])
                    sequence, signal = train[0]
                    self.assertEqual(sequence.shape, (32, 4))
                    self.assertEqual(signal.shape, (32, 1))
                    np.testing.assert_allclose(sequence.numpy().sum(-1), 1)
                    np.testing.assert_allclose(signal.numpy(), 1)
                    self.assertEqual(test[1][0].shape, (32, 4))
                    self.assertEqual(len(requests), 1)
                    self.assertTrue((root / (prepare.GENOME_FILE + '.fai')).exists())
                    self.assertEqual((root / (prepare.GENOME_FILE + '.mmap')).stat().st_size, 4 * 10000 * 4)
                    with pysam.TabixFile(str(root / prepare.BLACKLIST_FILES[0])) as tabix:
                        self.assertEqual(list(tabix.fetch('chr1', 0, 20)), ['chr1\t0\t10'])
                    # A custom Sei destination can be recovered from the same cached archive.
                    model = root / 'custom' / 'sei.pt'
                    features = root / 'custom' / 'features.txt'
                    prepare.prepare_sei_resources(root, model, features)
                    self.assertEqual(model.read_bytes(), b'fixture')
                    self.assertEqual(len(requests), 1)

    def test_chunked_genome_encoding_rebuilds_incomplete_cache(self):
        import pyfaidx
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fasta = root / 'tiny.fa'
            fasta.write_text('>chr1\n' + 'A' * (1024 * 1024) + 'cGn\n>chr2\nTACN\n')
            cache = root / 'tiny.mmap'
            cache.write_bytes(b'broken')
            with pyfaidx.Fasta(str(fasta)) as genome:
                data = prepare.prepare_genome_memmap(genome, cache)
                np.testing.assert_allclose(data[:, 1024 * 1024:1024 * 1024 + 3].T,
                                           [[0,1,0,0], [0,0,1,0], [.25,.25,.25,.25]])
                np.testing.assert_allclose(data[:, -4:].T,
                                           [[0,0,0,1], [1,0,0,0], [0,1,0,0], [.25,.25,.25,.25]])
                timestamp = cache.stat().st_mtime_ns
                del data
                prepare.prepare_genome_memmap(genome, cache)
                self.assertEqual(cache.stat().st_mtime_ns, timestamp)


if __name__ == '__main__':
    unittest.main()
