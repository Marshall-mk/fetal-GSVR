import os
import argparse
import yaml
import warnings
from datetime import datetime

from model import train

# --- PREAMBLE ---
warnings.filterwarnings("ignore", message=".*torch._prims_common.check.*")
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*")

# --- CLI OVERRIDES ---
def apply_overrides(cfg, overrides):
    """Apply dot-notation CLI overrides to a config dict.

    Parses pairs like ['--training.batch_size', '500000'] and sets the
    corresponding nested key, auto-casting to the existing value's type.
    """
    i = 0
    while i < len(overrides):
        key = overrides[i]
        if not key.startswith('--'):
            i += 1
            continue
        key = key[2:]  # strip leading --
        if i + 1 >= len(overrides) or overrides[i + 1].startswith('--'):
            raise ValueError(f"Override '{key}' has no value")
        raw_value = overrides[i + 1]
        i += 2

        parts = key.split('.')
        d = cfg
        for part in parts[:-1]:
            if part not in d:
                raise KeyError(f"Config key '{key}': '{part}' not found")
            d = d[part]

        final_key = parts[-1]
        if final_key not in d:
            raise KeyError(f"Config key '{key}': '{final_key}' not found")

        existing = d[final_key]
        if isinstance(existing, bool):
            d[final_key] = raw_value.lower() in ('true', '1', 'yes')
        elif isinstance(existing, int):
            d[final_key] = int(raw_value)
        elif isinstance(existing, float):
            d[final_key] = float(raw_value)
        else:
            d[final_key] = raw_value

        print(f"Override: {key} = {d[final_key]!r}")


# --- MAIN ---
def main(config_path, overrides=None):
    parser = argparse.ArgumentParser()
    # If called via subprocess with --config, that value takes precedence.
    # If called manually with a function argument, the default is used.
    parser.add_argument('--config', type=str, default=config_path)
    parser.add_argument('--exp_name', type=str, default=None)

    # Use parse_known_args to ensure flexibility if extra flags are ever passed
    args, unknown = parser.parse_known_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    cli_overrides = overrides if overrides is not None else unknown
    if cli_overrides:
        apply_overrides(cfg, cli_overrides)

    for i, subject in enumerate(cfg['data']['subjects']):
        if not subject['enabled']: continue
        print(f"--- Starting Subject {i+1} of {len(cfg['data']['subjects'])} ---")
        exp_name = args.exp_name if args.exp_name else cfg['data']['subjects'][i]['name']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_name = f"{exp_name}_{timestamp}.nii.gz"
        output_file_path = os.path.join(cfg['experiment']['output_root'], exp_name, output_file_name)
        output_file_path = os.path.abspath(output_file_path)
        print(f"Output file path: {output_file_path}")
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        with open(output_file_path.replace('.nii.gz', '.yaml'), 'w') as f: yaml.dump(cfg, f)

        stack_paths = cfg['data']['subjects'][i]['input_stacks']
        mask_paths = cfg['data']['subjects'][i]['input_masks']
        if not mask_paths: mask_paths = []

        try:
            train(stack_paths=stack_paths, mask_paths=mask_paths, config=cfg, output_file_path=output_file_path)
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Stack paths: {stack_paths}")
            print(f"Mask paths: {mask_paths}")
            print(f"Config: {cfg}")
            print(f"Output file path: {output_file_path}")
            raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)

    args, unknown = parser.parse_known_args()

    main(args.config, overrides=unknown)
