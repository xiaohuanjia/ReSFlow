"""Persisted experiment settings: inheritance, CLI routing, and inference."""
import argparse
import ast
from contextlib import redirect_stderr, redirect_stdout
import copy
import io
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from configuration import load_config, parse_run_args

ROOT = Path(__file__).resolve().parents[1]
ROUTES = {
 'bmnist_train.yml': 'main.py',
 'bmnist_reflow_run.yml': 'main.py',
 'bmnist_infer_1step.yml': 'main.py',
 'bmnist_infer_ode.yml': 'main.py',
 'bmnist_select_1step.yml': 'select_best_model.py',
 'text8_reflow_resume.yml': 'main.py',
 'text8_sphere_lightning.yml': 'main_lightning.py',
 'text8_sample_1step.yml': 'sample_text8.py',
 'text8_sample_ode.yml': 'sample_text8.py',
 'text8_eval_bpc.yml': 'eval_text8_bpc.py',
 'promoter_train.yml': 'main.py',
 'promoter_reflow_run.yml': 'main.py',
 'promoter_sample_1step.yml': 'sample_promoter.py',
 'promoter_sample_ode.yml': 'sample_promoter.py',
 'promoter_eval_1step.yml': 'main.py',
 'promoter_eval_ode.yml': 'main.py',
}


def argument_builder(script):
    """Execute the actual parser statements, avoiding GPU/model imports."""
    tree = ast.parse((ROOT / script).read_text())
    if script in ('main.py', 'main_lightning.py'):
        body = next(node.body for node in tree.body if isinstance(node, ast.If))
    else:
        body = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    statements = []
    for node in body:
        statements.append(copy.deepcopy(node))
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'args' for target in node.targets):
            break
    code = ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[]))
    scope = dict(argparse=argparse, torch=torch, parse_run_args=parse_run_args)
    def build(argv):
        with patch.object(sys, 'argv', [script, *argv]):
            exec(compile(code, script, 'exec'), scope)
        return scope['args']
    return build


class ConfigurationTests(unittest.TestCase):
    def test_relative_inheritance_and_nested_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'base.yml').write_text('model:\n  type: sphere\n  n_class: 27\nrun:\n  method: euler\n  n_steps: 1\n')
            (root / 'child.yml').write_text('extends: base.yml\nmodel:\n  n_class: 4\nrun:\n  method: ode\n')
            result = load_config(root / 'child.yml')
            self.assertEqual(result.model.type, 'sphere')
            self.assertEqual(result.model.n_class, 4)
            self.assertEqual(result.run.method, 'ode')
            self.assertEqual(result.run.n_steps, 1)
            self.assertNotIn('extends', result)

    def test_cycles_and_non_mapping_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.yml').write_text('extends: b.yml\n')
            (root / 'b.yml').write_text('extends: a.yml\n')
            with self.assertRaisesRegex(ValueError, 'Circular'):
                load_config(root / 'a.yml')
            (root / 'bad.yml').write_text('[1, 2]')
            with self.assertRaisesRegex(ValueError, 'mapping'):
                load_config(root / 'bad.yml')

    def test_every_profile_routes_all_runtime_settings(self):
        for name, script in ROUTES.items():
            with self.subTest(profile=name):
                path = ROOT / 'configs' / name
                config = load_config(path)
                flags = [str(path)] if script in ('main.py', 'main_lightning.py') else ['--config', str(path)]
                args = argument_builder(script)(flags)
                for key, value in config.run.items():
                    self.assertEqual(getattr(args, key), value, key)
                self.assertIn('datasets', config)
                self.assertIn('model', config)
                self.assertIn('encoder', config)

    def test_explicit_cli_override_remains_compatible(self):
        path = ROOT / 'configs' / 'promoter_sample_1step.yml'
        args = argument_builder('sample_promoter.py')(['--config', str(path), '--device', 'cpu', '--n_sample', '2', '--shuffle'])
        self.assertEqual(args.device, 'cpu')
        self.assertEqual(args.n_sample, 2)
        self.assertTrue(args.shuffle)
        self.assertEqual(args.method, 'euler')

    def test_invalid_values_and_missing_checkpoint_fail_before_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.yml'
            cases = [({'method':'bad'}, 'method'), ({'n_steps':0}, 'positive'),
                     ({'shuffle':'false'}, 'boolean'), ({'ckpt':None}, 'required'),
                     ({'typo':1}, 'Unsupported')]
            for run, message in cases:
                with self.subTest(run=run):
                    base = {'ckpt':'weights.pt', **run}
                    path.write_text(yaml.safe_dump({'run':base}))
                    error = io.StringIO()
                    with redirect_stderr(error), self.assertRaises(SystemExit) as exit:
                        argument_builder('sample_promoter.py')(['--config', str(path)])
                    self.assertEqual(exit.exception.code, 2)
                    self.assertIn(message, error.getvalue())

    def test_environment_variables_do_not_override_profiles(self):
        with patch.dict(os.environ, {'METHOD':'ode', 'N_STEPS':'999', 'SPLIT':'valid'}):
            args = argument_builder('main.py')([str(ROOT / 'configs/bmnist_infer_1step.yml')])
        self.assertEqual((args.method, args.n_steps, args.split), ('euler', 1, 'test'))
        tree = ast.parse((ROOT / 'main.py').read_text())
        self.assertFalse(any(isinstance(node, ast.Attribute) and node.attr in ('environ', 'getenv') for node in ast.walk(tree)))

    def test_readme_experiment_commands_reference_real_profiles(self):
        text = (ROOT / 'README.md').read_text().split('## Training and evaluation\n')[1].split('## Citation\n')[0]
        commands = [line.strip() for line in text.splitlines() if line.strip().startswith('python ')]
        documented_profiles = set(ROUTES)
        self.assertEqual({Path(command.split()[-1]).name for command in commands}, documented_profiles)
        for command in commands:
            tokens = command.split()
            self.assertEqual(len(tokens), 4 if tokens[2] == '--config' else 3)
            filename = tokens[-1]
            self.assertTrue((ROOT / filename).is_file())
            self.assertEqual(ROUTES[Path(filename).name], tokens[1])
        self.assertNotRegex((ROOT / 'README.md').read_text(), r'\b(?:METHOD|N_STEPS|SPLIT)=')

    def test_main_inference_passes_yaml_settings_to_visualizer(self):
        from utils import count_parameters, get_optimizer, get_scheduler, recursive_to_device, seed_all
        tree = ast.parse((ROOT / 'main.py').read_text())
        body = next(node.body for node in tree.body if isinstance(node, ast.If))
        code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), 'main.py', 'exec')
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))
        for name, method, n_steps, conditional in [
                ('bmnist_infer_1step.yml','euler',1,False),
                ('bmnist_infer_ode.yml','ode',200,False),
                ('promoter_eval_1step.yml','euler',1,True),
                ('promoter_eval_ode.yml','ode',300,True)]:
            with self.subTest(profile=name), tempfile.TemporaryDirectory() as directory:
                model = Model()
                visualizer = Mock()
                dataset = torch.utils.data.TensorDataset(torch.zeros(2, 1))
                scope = dict(argparse=argparse, os=os, torch=torch, time=Mock(),
                             SummaryWriter=Mock(), DataLoader=torch.utils.data.DataLoader, parse_run_args=parse_run_args,
                             load_config=load_config, seed_all=seed_all,
                             get_dataset=Mock(return_value=(dataset,dataset,dataset)),
                             get_vis=Mock(return_value=visualizer),
                             get_flow_model=Mock(return_value=model),
                             get_optimizer=get_optimizer, get_scheduler=get_scheduler,
                             count_parameters=count_parameters, recursive_to_device=recursive_to_device)
                with patch.object(sys, 'argv', ['main.py',str(ROOT/'configs'/name),'--device','cpu','--logdir',directory]), \
                     patch.object(torch, 'load', return_value={'model':model.state_dict()}), \
                     patch('os.path.isfile', return_value=True), \
                     patch.dict(os.environ, {'METHOD':'bogus','N_STEPS':'999','SPLIT':'valid'}), redirect_stdout(io.StringIO()):
                    exec(code, scope)
                self.assertEqual(visualizer.n_step, n_steps)
                args, kwargs = visualizer.call_args
                self.assertIs(args[0], model)
                if conditional:
                    self.assertIs(args[1], scope['test_loader'])
                    self.assertEqual(args[2], method)
                    self.assertIsNone(kwargs['max_batch'])
                else:
                    self.assertEqual(args[1], method)
                scope['time'].sleep.assert_called_once()


if __name__ == '__main__':
    unittest.main()
