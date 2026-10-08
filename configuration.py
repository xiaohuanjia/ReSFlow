"""YAML experiment inheritance and persisted command-line defaults."""

import argparse
from copy import deepcopy
from pathlib import Path

from easydict import EasyDict
import yaml


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(config_path, _parents=()):
    """Resolve `extends` relative to the YAML file, then merge nested keys."""
    path = Path(config_path).resolve()
    if path in _parents:
        raise ValueError(f'Circular configuration inheritance: {path}')
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    parent = config.pop('extends', None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError(f'extends must be a filename: {path}')
        inherited = load_config(path.parent / parent, (*_parents, path))
        config = _merge(inherited, config)
    return EasyDict(config)


def parse_run_args(parser, argv=None, required=()):
    """Use YAML `run` settings as defaults; explicit CLI flags still override."""
    preliminary = parser.parse_args(argv)
    filename = getattr(preliminary, 'config', None)
    if filename:
        run = load_config(filename).get('run', {})
        if not isinstance(run, dict):
            parser.error('run must be a mapping')
        actions = {action.dest: action for action in parser._actions}
        values = {}
        for key, value in run.items():
            if key not in actions or key in ('config', 'help'):
                parser.error(f'Unsupported run setting for this entry point: {key}')
            action = actions[key]
            if value is not None:
                if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
                    if not isinstance(value, bool):
                        parser.error(f'run.{key} must be a boolean')
                elif action.type:
                    try:
                        value = action.type(str(value))
                    except (ValueError, TypeError):
                        parser.error(f'Invalid value for run.{key}: {value!r}')
                if action.choices and value not in action.choices:
                    parser.error(f'Invalid run.{key}: {value!r}; choose from {action.choices}')
            values[key] = value
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    for key in required:
        if not getattr(args, key, None):
            parser.error(f'{key} is required; set run.{key} in YAML or pass it explicitly')
    for key in ('n_steps', 'n_sample', 'n_sequences', 'total_sample', 'batch_size', 'repeat', 'n_runs', 'n_noise'):
        value = getattr(args, key, None)
        if value is not None and value <= 0:
            parser.error(f'{key} must be positive')
    for key in ('num_workers', 'warmup', 'max_batch'):
        value = getattr(args, key, None)
        if value is not None and value < 0:
            parser.error(f'{key} must be nonnegative')
    return args
