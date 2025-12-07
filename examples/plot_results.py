# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import csv
import ast
import re
from typing import Any


def parse_config(config_str: str) -> dict[str, Any]:
    try:
        return ast.literal_eval(config_str)
    except Exception as e:
        print(f"Failed to parse config: {config_str}, error: {e}")
        return {}


def load_results(result_dir: Path) -> list[dict[str, Any]]:
    data = []
    # Regex for filename
    # Example: ZeRO-2.0_batch1_seq512_paramnone_results_1764966389.csv
    filename_pattern = re.compile(
        r"^ZeRO-(?P<strategy>[\d\.]+)_batch(?P<batch>\d+)_seq(?P<seq>\d+)_param(?P<param>[a-zA-Z0-9_]+)_results_(?P<timestamp>\d+)\.csv$"
    )

    for p in result_dir.iterdir():
        if not p.is_file():
            continue

        match = filename_pattern.match(p.name)
        if not match:
            continue

        try:
            with open(p, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('success', 'True') != 'True':
                        continue

                    cfg_str = row.get('config', '{}')
                    cfg = parse_config(cfg_str)
                    if not cfg:
                        continue

                    strategy = float(cfg['strategy'])
                    batch_size = int(cfg['batch_size'])
                    seq_len = int(cfg['seq_len'])
                    total_tokens = batch_size * seq_len

                    avg_step_time = float(row['avg_step_time'])
                    tokens_per_sec = float(row['tokens_per_sec'])
                    peak_gpu_mem_mb = float(row['peak_gpu_mem_mb'])

                    checkpointing = cfg['checkpointing']

                    item = {
                        'strategy': strategy,
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'total_tokens': total_tokens,
                        'avg_step_time': avg_step_time,
                        'tokens_per_sec': tokens_per_sec,
                        'peak_gpu_mem_mb': peak_gpu_mem_mb,
                        'num_persistent_layers': cfg.get('num_persistent_layers'),
                        'forward_reduce_bucket_size': cfg.get('forward_reduce_bucket_size'),
                        'checkpointing': checkpointing
                    }
                    data.append(item)
        except Exception as e:
            print(f"Error reading {p.name}: {e}")
    return data


def plot_line_metrics(data, strategy, result_dir):
    ckpt_values = sorted(list(set(d['checkpointing'] for d in data)), key=lambda x: str(x))

    for ckpt in ckpt_values:
        ckpt_data = [d for d in data if d['checkpointing'] == ckpt]
        if not ckpt_data:
            continue

        ckpt_data.sort(key=lambda x: x['total_tokens'])
        unique_tokens = sorted(list(set(d['total_tokens'] for d in ckpt_data)))

        metrics = ['avg_step_time', 'tokens_per_sec', 'peak_gpu_mem_mb']

        for metric in metrics:
            plot_vals = []
            for t in unique_tokens:
                candidates = [d for d in ckpt_data if d['total_tokens'] == t]
                if candidates:
                    best = max(candidates, key=lambda x: x['tokens_per_sec'])
                    plot_vals.append(best[metric])
                else:
                    plot_vals.append(0)

            plt.figure(figsize=(10, 6))
            plt.plot(unique_tokens, plot_vals, marker='o', linestyle='-', linewidth=2)
            plt.xlabel('Total Tokens (Batch * Seq)')
            plt.ylabel(metric)
            plt.title(f'Strategy {strategy} (Ckpt={ckpt}): {metric} vs Total Tokens')
            plt.grid(True, alpha=0.3)

            out_path = result_dir / f'strategy_{strategy}_ckpt_{ckpt}_{metric}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")


def plot_heatmap_metrics(data, strategy, result_dir):
    if abs(strategy - 3.5) < 1e-5:
        param_name = 'num_persistent_layers'
        param_label = 'Num Persistent Layers'
    elif abs(strategy - 4.0) < 1e-5:
        param_name = 'forward_reduce_bucket_size'
        param_label = 'Forward Reduce Bucket Size'
    else:
        print(f"Unknown parameter for strategy {strategy}")
        return

    ckpt_values = sorted(list(set(d['checkpointing'] for d in data)), key=lambda x: str(x))

    for ckpt in ckpt_values:
        ckpt_data = [d for d in data if d['checkpointing'] == ckpt]
        if not ckpt_data:
            continue

        unique_tokens = sorted(list(set(d['total_tokens'] for d in ckpt_data)))
        unique_params = sorted(list(set(d[param_name] for d in ckpt_data if d[param_name] is not None)))

        if not unique_tokens or not unique_params:
            print(f"Not enough data for heatmap for strategy {strategy} (Ckpt={ckpt})")
            continue

        metrics = ['avg_step_time', 'tokens_per_sec', 'peak_gpu_mem_mb']

        for metric in metrics:
            grid = np.zeros((len(unique_params), len(unique_tokens)))
            grid[:] = np.nan

            for i, p_val in enumerate(unique_params):
                for j, t_val in enumerate(unique_tokens):
                    matches = [d for d in ckpt_data if d['total_tokens'] == t_val and d[param_name] == p_val]
                    if matches:
                        best_run = max(matches, key=lambda x: x['tokens_per_sec'])
                        grid[i, j] = best_run[metric]

            plt.figure(figsize=(12, 8))
            im = plt.imshow(grid, aspect='auto', origin='lower', cmap='viridis')

            plt.xticks(range(len(unique_tokens)), unique_tokens, rotation=45)
            plt.yticks(range(len(unique_params)), unique_params)

            plt.xlabel('Total Tokens')
            plt.ylabel(param_label)
            plt.title(f'Strategy {strategy} (Ckpt={ckpt}): {metric} Heatmap')
            plt.colorbar(im, label=metric)

            for i in range(len(unique_params)):
                for j in range(len(unique_tokens)):
                    val = grid[i, j]
                    if not np.isnan(val):
                        if val > 1000:
                            txt = f'{val:.0f}'
                        else:
                            txt = f'{val:.2f}'
                        plt.text(j, i, txt, ha='center', va='center', color='w', fontsize=8, fontweight='bold')

            plt.tight_layout()
            out_path = result_dir / f'strategy_{strategy}_ckpt_{ckpt}_{metric}_heatmap.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")


def plot_comparison(all_data, result_dir):
    strategies = [2.0, 3.0, 3.5, 4.0]
    metrics = ['avg_step_time', 'tokens_per_sec', 'peak_gpu_mem_mb']

    ckpt_values = sorted(list(set(d['checkpointing'] for d in all_data)), key=lambda x: str(x))

    for ckpt in ckpt_values:
        ckpt_data = [d for d in all_data if d['checkpointing'] == ckpt]
        if not ckpt_data:
            continue

        unique_tokens = sorted(list(set(d['total_tokens'] for d in ckpt_data)))
        if not unique_tokens:
            continue

        for metric in metrics:
            plt.figure(figsize=(14, 8))

            bar_width = 0.15
            indices = np.arange(len(unique_tokens))

            for i, strategy in enumerate(strategies):
                vals = []
                for t in unique_tokens:
                    candidates = [
                        d for d in ckpt_data if abs(d['strategy'] - strategy) < 1e-5 and d['total_tokens'] == t
                    ]
                    if candidates:
                        best_run = max(candidates, key=lambda x: x['tokens_per_sec'])
                        vals.append(best_run[metric])
                    else:
                        vals.append(0)

                if any(v > 0 for v in vals):
                    plt.bar(indices + i * bar_width, vals, width=bar_width, label=f'ZeRO-{strategy}')

            plt.xlabel('Total Tokens')
            plt.ylabel(metric)
            plt.title(f'Comparison of {metric} across Strategies (Ckpt={ckpt})')
            plt.xticks(indices + bar_width * 1.5, unique_tokens, rotation=45)
            plt.legend()
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()

            out_path = result_dir / f'comparison_ckpt_{ckpt}_{metric}.png'
            plt.savefig(out_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved {out_path}")


def plot_results(result_dir: str):
    result_dir: Path = Path(result_dir)
    if not result_dir.exists():
        print(f"Directory {result_dir} does not exist")
        return

    print(f"Loading results from {result_dir}...")
    all_data = load_results(result_dir)

    if not all_data:
        print("No valid data found.")
        return

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    strategies = (2.0, 3.0, 3.5, 4.0)
    for strategy in strategies:
        strat_data = [d for d in all_data if abs(d['strategy'] - strategy) < 1e-5]
        if strat_data:
            if abs(strategy - 2.0) < 1e-5 or abs(strategy - 3.0) < 1e-5:
                print(f"Plotting line charts for Strategy {strategy}...")
                plot_line_metrics(strat_data, strategy, plot_dir)
            elif abs(strategy - 3.5) < 1e-5 or abs(strategy - 4.0) < 1e-5:
                print(f"Plotting heatmaps for Strategy {strategy}...")
                plot_heatmap_metrics(strat_data, strategy, plot_dir)
        else:
            print(f"No data found for strategy {strategy}")

    print("Plotting comparison charts...")
    plot_comparison(all_data, plot_dir)


if __name__ == "__main__":
    plot_results("./results")
