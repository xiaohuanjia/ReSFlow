"""CPU smoke checks for the saved evaluation notebook; no pretrained FID download."""
import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import torch

from configuration import load_config
from models.categorical import SphereCategoricalFlow

ROOT = Path(__file__).resolve().parents[1]


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, points, time):
        return self.linear(points)


def tiny_model(*args):
    return SphereCategoricalFlow(TinyEncoder(), (28, 28), 2, ot=False)


class SmallFeatures(torch.nn.Module):
    """Only the pretrained feature network is replaced in the CPU smoke run."""
    BLOCK_INDEX_BY_DIM = {2048: 0}

    def __init__(self, *args):
        super().__init__()

    def forward(self, images):
        return [torch.stack([images.mean((1, 2, 3)), images[:, :, :14].mean((1, 2, 3))], 1)[:, :, None, None]]


class NotebookTests(unittest.TestCase):
    def notebook(self):
        return nbformat.read(ROOT / 'eval_bmnist.ipynb', as_version=4)

    def test_schema_and_evaluation_only_structure(self):
        notebook = self.notebook()
        nbformat.validate(notebook)
        headings = [cell.source.splitlines()[0] for cell in notebook.cells
                    if cell.cell_type == 'markdown' and cell.source.startswith('## ')]
        self.assertEqual(headings, ['## 1. Load models and data', '## 2. Precompute FID statistics',
                                    '## 3. Sample and evaluate'])
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                tree = ast.parse(cell.source)
                compile(tree, f'cell_{index}', 'exec')
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)
                attributes = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)]
                for forbidden in ('backward', 'get_loss', 'train', 'environ'):
                    self.assertNotIn(forbidden, attributes)

    def test_all_cells_sampling_metrics_exports_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'configs').mkdir()
            (root / 'data').mkdir()
            rng = np.random.default_rng(3)
            np.save(root / 'data/binarized_mnist_test.npy', rng.integers(0, 2, (12, 784)).astype(np.float32))
            weights = root / 'weights.pt'
            torch.save({'model': tiny_model().state_dict()}, weights)
            config = load_config(ROOT / 'configs/bmnist_eval_notebook.yml')
            config.datasets.root = str(root / 'data')
            settings = config.evaluation
            settings.device = 'cpu'
            settings.output_dir = str(root / 'outputs')
            settings.checkpoints = [{'name': 'ReSFlow', 'path': str(weights)}]
            settings.total_sample = 9
            settings.sample_batch_size = 4
            settings.fid_batch_size = 5
            settings.data_batch_size = 5
            settings.seeds = [42, 43]
            settings.grid_n_sample = 9
            (root / 'configs/bmnist_eval_notebook.yml').write_text(json.dumps(config))
            old_cwd = Path.cwd()
            scope = {}
            try:
                os.chdir(root)
                with patch('models.get_flow_model', side_effect=tiny_model), \
                     patch('evaluation.inception.InceptionV3', SmallFeatures), \
                     patch('matplotlib.pyplot.show'), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    for index, cell in enumerate(self.notebook().cells):
                        if cell.cell_type == 'code':
                            exec(compile(cell.source, f'cell_{index}', 'exec'), scope)
                    results = scope['results']
                    self.assertEqual(len(results), 4)
                    self.assertTrue((results.n_samples == 9).all())
                    self.assertTrue(np.isfinite(results.fid).all())
                    one_step = results[results.method == 'euler']
                    self.assertTrue((one_step.nfe_mean_per_batch == 1).all())
                    self.assertTrue((results[results.method == 'ode'].nfe_mean_per_batch > 1).all())
                    cache = root / 'outputs/bmnist_test_fid.npz'
                    previous_mtime = cache.stat().st_mtime_ns
                    # Reuse matching statistics without recomputing features.
                    with patch.object(scope['fid_network'], 'forward', side_effect=AssertionError('cache miss')):
                        mu, sigma = scope['reference_statistics'](
                            scope['reference_loader'], scope['fid_network'], scope['device'], 5,
                            cache, scope['reference_metadata'])
                    self.assertEqual(previous_mtime, cache.stat().st_mtime_ns)
                    np.testing.assert_allclose(mu, scope['mu_ref'])
                    self.assertEqual(len(list((root / 'outputs').glob('*.png'))), 4)
                    for filename in ('metrics.csv', 'metrics.json', 'summary.csv', 'evaluation_config.json'):
                        self.assertTrue((root / 'outputs' / filename).is_file(), filename)
                    self.assertEqual(len(scope['models']['ReSFlow']._forward_pre_hooks), 0)
                    import matplotlib.pyplot as plt
                    plt.close('all')
            finally:
                os.chdir(old_cwd)


if __name__ == '__main__':
    unittest.main()
