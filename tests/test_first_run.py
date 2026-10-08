"""Small CPU checks for documented training/checkpoint/evaluation transitions."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from easydict import EasyDict
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from configuration import load_config
from models import get_flow_model
from models.dit.transformer import DDiTBlock
from models.dit.rotary import Rotary

ROOT = Path(__file__).resolve().parents[1]


class FirstRunTests(unittest.TestCase):
    def test_native_attention_matches_explicit_attention(self):
        torch.manual_seed(1)
        block = DDiTBlock(16, 2, 8, dropout=0).eval()
        torch.nn.init.normal_(block.adaLN_modulation.weight, std=.1)
        points = torch.randn(2, 4, 16, requires_grad=True)
        condition = torch.randn(2, 8)
        cos_sin = Rotary(8)(points)
        actual = block(points, cos_sin, condition)
        def attention(query, key, value, **kwargs):
            weights = (query @ key.transpose(-1, -2) / query.shape[-1]**.5).softmax(-1)
            return weights @ value
        with patch('models.dit.transformer.F.scaled_dot_product_attention', side_effect=attention):
            reference = block(points, cos_sin, condition)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-6)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(points.grad).all())

    def test_documented_checkpoint_paths_connect(self):
        for train_name, reflow_name, sample_name, extension in [
            ('bmnist_train.yml','bmnist_reflow_run.yml','bmnist_infer_1step.yml','100000.pt'),
            ('text8_sphere_lightning.yml','text8_reflow_resume.yml','text8_sample_1step.yml','last.ckpt'),
            ('promoter_train.yml','promoter_reflow_run.yml','promoter_sample_1step.yml','latest.pt'),
        ]:
            train = load_config(ROOT/'configs'/train_name)
            reflow = load_config(ROOT/'configs'/reflow_name)
            sample = load_config(ROOT/'configs'/sample_name)
            self.assertEqual(Path(reflow.run.resume), Path(train.run.logdir)/train.run.savename/extension)
            checkpoint = sample.run.get('ckpt', sample.run.get('resume'))
            self.assertEqual(Path(checkpoint), Path(reflow.run.logdir)/reflow.run.savename/f'reflow_r{reflow.reflow.k}_final.pt')
            self.assertEqual(train.model, reflow.model)
            self.assertEqual(train.encoder, reflow.encoder)
            self.assertEqual(reflow.model, sample.model)
            self.assertEqual(reflow.encoder, sample.encoder)
        lightning = load_config(ROOT/'configs/text8_sphere_lightning.yml')
        self.assertEqual(lightning.train.batch_size * lightning.trainer.devices, 512)

    def test_lightning_base_to_reflow_to_bpc(self):
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.loggers import TensorBoardLogger
        from models.text8_module import Text8Module
        from reflow import run_reflow_pipeline
        from eval_text8_bpc import compute_bpc_ode
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cfg = load_config(ROOT/'configs/text8_sphere_lightning.yml')
            cfg.model.data_dims = [4]
            cfg.encoder.update(hidden_size=16, cond_dim=8, length=4, n_blocks=1, n_heads=2, dropout=0.)
            cfg.sample.update(n_sample=1, n_step=1)
            cfg.train.val_check_interval = 1
            cfg.train.lr_warmup_steps = 1
            batch = F.one_hot(torch.randint(27, (4, 4)), 27).float()
            loader = DataLoader(TensorDataset(batch), batch_size=2)
            module = Text8Module(**cfg)
            checkpoint_callback = ModelCheckpoint(dirpath=directory, save_last=True, monitor='val_loss', save_top_k=1)
            trainer = pl.Trainer(accelerator='cpu', devices=1, precision='32-true', max_steps=2,
                                 max_epochs=-1, limit_val_batches=1, val_check_interval=1,
                                 logger=TensorBoardLogger(directory, name='smoke'),
                                 callbacks=[checkpoint_callback], enable_progress_bar=False,
                                 enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(module, loader, loader)
            path = Path(directory)/'last.ckpt'
            self.assertTrue(path.is_file())
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            native = get_flow_model(cfg.model, cfg.encoder)
            native.load_state_dict({key[6:]:value for key,value in checkpoint['state_dict'].items() if key.startswith('model.')})
            reflow_cfg = copy.deepcopy(cfg)
            reflow_cfg.reflow = EasyDict(k=1, n_pairs=3, gen_batch_size=2, gen_method='euler',
                                        gen_n_steps=1, seed=0, max_iter=2, batch_size=2,
                                        lr=1e-4, log_freq=1, save_freq=1, vis_freq=None)
            logdir = Path(directory)/'reflow';logdir.mkdir()
            result = run_reflow_pipeline(native, reflow_cfg, 'cpu', logdir=str(logdir))
            final = logdir/'reflow_r1_final.pt'
            self.assertTrue(final.is_file())
            evaluated = get_flow_model(cfg.model, cfg.encoder)
            evaluated.load_state_dict(torch.load(final, map_location='cpu', weights_only=False)['model'])
            samples = evaluated.eval().sample('euler', 2, 1, 'cpu')
            self.assertEqual(tuple(samples.shape), (2, 4, 27))
            bpc = compute_bpc_ode(evaluated, batch[:1], s_max=.1, s_min=.01, atol=1e-3, rtol=1e-3)
            self.assertTrue(torch.isfinite(bpc).all())

    def test_checkpoint_selection_keeps_partial_batch(self):
        from select_best_model import compute_fid_for_checkpoint
        cfg = EasyDict(type='sphere', data_dims=[28,28], n_class=2, ot=False)
        class Encoder(torch.nn.Module):
            def forward(self, points, time):
                return torch.zeros_like(points)
        from models.categorical import SphereCategoricalFlow
        model = SphereCategoricalFlow(Encoder(), cfg.data_dims, 2)
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            path = Path(directory)/'weights.pt'
            torch.save({'model': model.state_dict()}, path)
            with patch('select_best_model.get_fid', return_value=0.) as fid:
                compute_fid_for_checkpoint(str(path), model, 'cpu', 9, 4, 1)
            self.assertEqual(fid.call_args.args[0].shape[0], 9)


if __name__ == '__main__':
    unittest.main()
