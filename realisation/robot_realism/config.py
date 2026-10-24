from copy import deepcopy
from pathlib import Path
import os
import re

from .common import digest


def load_config(path, *, input_root=None, output_root=None):
    import yaml
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1 YAML configuration")
    for key in ("input", "output", "runtime", "generation", "controls", "selection"):
        if not isinstance(config.get(key), dict):
            raise ValueError(f"Missing configuration section: {key}")
    for section, key in (("input", "root"), ("output", "root"), ("output", "run_dir"),
                         ("output", "scratch"), ("runtime", "python"), ("runtime", "cache")):
        value = Path(config[section][key]).expanduser()
        # Resolving venv/bin/python follows its symlink into the base image and
        # silently drops the venv. Preserve the executable path itself.
        config[section][key] = (os.path.abspath(path.parent / value) if (section, key) == ('runtime', 'python')
                                else str((path.parent / value).resolve()))
    if input_root:
        config["input"]["root"] = str(Path(input_root).expanduser().resolve())
    if output_root:
        config["output"]["root"] = str(Path(output_root).expanduser().resolve())
    roots = [Path(config["input"]["root"])] + [Path(config["output"][k]) for k in ("root", "run_dir", "scratch")]
    for i, first in enumerate(roots):
        for second in roots[i + 1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                raise ValueError(f"Input/output/run_dir/scratch must not contain each other: {first}, {second}")
    for root in roots[:2]:
        for runtime in (Path(config['runtime']['cache']), Path(config['runtime']['python']).parent):
            if runtime.is_relative_to(root) or root.is_relative_to(runtime):
                raise ValueError("Runtime environment/cache must be separate from input and output")
    cameras = config['input']['cameras']
    if not cameras or len(cameras) != len(set(cameras)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', c) for c in cameras):
        raise ValueError("cameras must be distinct safe names")
    modes = config['generation']['controls']
    if not modes or len(set(modes)) != len(modes) or set(modes) - {'depth', 'edge'}:
        raise ValueError("This adapter supports [depth], [edge], or [depth, edge]")
    if config['generation']['resolution'] not in ('480', '720'):
        raise ValueError("resolution must be the string '480' or '720'")
    if type(config['generation']['steps']) is not int or config['generation']['steps'] < 1:
        raise ValueError("steps must be a positive integer")
    if type(config['generation']['guidance']) is not int or not 0 <= config['generation']['guidance'] <= 7:
        raise ValueError("guidance must be an integer between 0 and 7")
    for key in ('depth_weight', 'edge_weight'):
        if not 0 < config['controls'][key] <= 1:
            raise ValueError(f"{key} must be >0 and <=1")
    if not 0 < config['controls']['near_m'] < config['controls']['far_m']:
        raise ValueError("Expected 0 < near_m < far_m")
    if not 0 <= config['controls']['canny_low'] < config['controls']['canny_high'] <= 255:
        raise ValueError("Expected 0 <= canny_low < canny_high <= 255")
    for key in ('enable_guardrails', 'offload_guardrails'):
        if type(config['runtime'][key]) is not bool:
            raise ValueError(f"runtime.{key} must be true or false")
    for name in ('prompt', 'negative_prompt'):
        if not isinstance(config['generation'][name], str) or not config['generation'][name].strip():
            raise ValueError(f"{name} must be nonempty text")
    if type(config['selection']['seed']) is not int:
        raise ValueError('selection.seed must be an integer')
    if 'prompting' in config:
        policy = config['prompting']
        if not isinstance(policy, dict) or policy.get('mode') not in ('mixed', 'paired_general'):
            raise ValueError('prompting.mode must be mixed or paired_general')
        if cameras != ['agentview']:
            raise ValueError('The prompt experiment requires only agentview')
        for key in ('selection_manifest', 'specific_manifest'):
            if not isinstance(policy.get(key), str) or not policy[key].strip():
                raise ValueError(f'prompting.{key} must be a path')
            policy[key] = str((path.parent / Path(policy[key]).expanduser()).resolve())
    if 'execution' in config:
        execution = config['execution']
        if not isinstance(execution, dict) or execution.get('guardrail_policy') not in ('stop', 'record_and_skip'):
            raise ValueError('execution.guardrail_policy must be stop or record_and_skip')
        if execution['guardrail_policy'] == 'record_and_skip':
            if cameras != ['agentview']:
                raise ValueError('record_and_skip is supported only for agentview')
            if not config['runtime']['enable_guardrails']:
                raise ValueError('record_and_skip requires enabled guardrails')
    return config


def config_id(config):
    value = deepcopy(config)
    value['output'] = {'root': value['output']['root']}
    if 'prompting' in config:
        from .prompting import prompt_inputs
        value['prompt_inputs_sha256'] = prompt_inputs(config)
    return digest(value)
