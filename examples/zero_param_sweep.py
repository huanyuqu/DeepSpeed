# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Zero Parameter Sweep Experiment

This script explores two dimensions:
1. Zero strategy parameters:
   - Zero3.5: num_persistent_layers (layers that don't release params after backward allgather)
   - Zero4: forward_reduce_bucket_size (gradients deferred to forward reduce)
2. Compute token volume: batch_size * seq_len

Usage:
    deepspeed examples/zero_param_sweep.py --model examples/Qwen3-4B --strategy zero4

Output: Performance and memory metrics with visualization (scatter plot / heatmap)
"""

import time
import argparse
import gc
import json
import os
import tempfile
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional
import csv
import re

import torch
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen3
from deepspeed.utils.logging import set_log_level_from_string

os.environ["TOKENIZERS_PARALLELISM"] = "false"
apply_liger_kernel_to_qwen3()


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    strategy: float  # 2, 3, 3.5 or 4
    batch_size: int  # batch size per gpu
    seq_len: int
    world_size: int  # number of gpus
    num_layers: int = None  # number of transformer layers
    gradient_accumulation_steps: int = 16
    # Zero3.5 specific: number of persistent layers (not released after backward)
    num_persistent_layers: Optional[int | tuple[int, int]] = None
    # Zero4 specific: forward reduce bucket size
    forward_reduce_bucket_size: Optional[int | tuple[int, int]] = None
    checkpointing: bool = False

    @property
    def total_tokens(self) -> int:
        return self.batch_size * self.seq_len

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_deepspeed_config(exp_config: ExperimentConfig) -> dict[str, Any]:
    """Generate DeepSpeed config dict based on experiment configuration"""
    base_config = {
        "train_batch_size": (exp_config.batch_size * exp_config.world_size * exp_config.gradient_accumulation_steps),
        "train_micro_batch_size_per_gpu": exp_config.batch_size,
        "gradient_accumulation_steps": exp_config.gradient_accumulation_steps,
        "steps_per_print": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e7,
            "stage3_prefetch_bucket_size": 1,
            "stage3_param_persistence_threshold": 1,
            "stage3_max_live_parameters": 1,
            "stage3_max_reuse_distance": 1e6,
            "reduce_bucket_size": 1e7,
        }
    }

    if exp_config.checkpointing:
        base_config["activation_checkpointing"] = {
            "partition_activations": False,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": False,
            "number_checkpoints": exp_config.num_layers
        }

    if exp_config.strategy == 2:
        base_config["zero_optimization"]["keep_params_available"] = True
    elif exp_config.strategy in (3.5, 4):
        # Zero3.5: partition_params_backward controls whether to release params after backward
        # num_persistent_layers controls how many layers keep params
        base_config["zero_optimization"]["partition_params_backward"] = False
        if exp_config.num_persistent_layers is not None:
            # This parameter controls how many layers don't release after backward allgather
            base_config["zero_optimization"]["num_persistent_layers"] = exp_config.num_persistent_layers
        if exp_config.strategy == 4:
            # Zero4: forward_reduce defers gradient reduce to forward pass
            base_config["zero_optimization"]["forward_reduce"] = True
            if exp_config.forward_reduce_bucket_size is not None:
                base_config["zero_optimization"]["forward_reduce_bucket_size"] = exp_config.forward_reduce_bucket_size

    return base_config


@dataclass
class ExperimentResult:
    """Result of a single experiment run"""
    config: ExperimentConfig
    avg_step_time: float
    tokens_per_sec: float
    peak_gpu_mem_mb: float
    success: bool
    error_msg: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_single_experiment(model_name: str,
                          exp_config: ExperimentConfig,
                          steps: int = 32,
                          warmup: int = 3,
                          local_rank: int = -1,
                          memory_snapshot: bool = False,
                          profile: bool = False,
                          profile_dir: str = None) -> ExperimentResult:
    """Run a single benchmark experiment"""

    if local_rank == 0:
        print(f"\n{'='*60}")
        print(f"Running experiment: ZeRO-{exp_config.strategy}")
        print(f"  batch_size={exp_config.batch_size}, seq_len={exp_config.seq_len}")
        print(f"  total_tokens={exp_config.total_tokens}")
        if exp_config.strategy == 3.5:
            print(f"  num_persistent_layers={exp_config.num_persistent_layers}")
        elif exp_config.strategy == 4:
            print(f"  forward_reduce_bucket_size={exp_config.forward_reduce_bucket_size}")
        print(f"{'='*60}")

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, trust_remote_code=True)

    # Generate DeepSpeed config
    ds_config = generate_deepspeed_config(exp_config)

    # Write config to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(ds_config, f, indent=2)
        config_path = f.name

    if memory_snapshot and local_rank == 0:
        torch.cuda.memory._record_memory_history(max_entries=100000)  #ignore-cuda
    deepspeed.init_distributed()

    # Initialize model
    if ds_config.get("zero_optimization", {}).get("stage") == 3:
        with deepspeed.zero.Init(config_dict_or_path=config_path):
            model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True, dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float16)

    if local_rank == 0:
        print("Model initialization completed")

    if exp_config.checkpointing:
        model.gradient_checkpointing_enable()
        if local_rank == 0:
            print("Recomputation enabled")

    # Create synthetic dataset
    texts = ["Hello world"] * (exp_config.batch_size * (steps + warmup))
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=exp_config.seq_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=exp_config.batch_size, shuffle=False)

    # Initialize DeepSpeed
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(model=model,
                                                                model_parameters=model.parameters(),
                                                                config=config_path)

    model_engine.train()
    device = model_engine.device
    it = iter(dataloader)

    # Warmup phase
    for i in range(warmup):
        batch_input_ids, batch_attention_mask, batch_labels = next(it)
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model_engine(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        model_engine.backward(loss)
        model_engine.step()

    # Reset memory stats
    get_accelerator().reset_peak_memory_stats(device)

    if hasattr(model_engine, "global_steps"):
        model_engine.global_steps = 0
    if hasattr(model_engine, "micro_steps"):
        model_engine.micro_steps = 0
    if hasattr(model_engine.optimizer, "zero_grad"):
        model_engine.optimizer.zero_grad(set_to_none=True)

    # Benchmark phase
    times = []
    total_tokens = 0

    if profile:
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
        ],
                                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                                    on_trace_ready=lambda prof: prof.export_chrome_trace(
                                        f'{profile_dir}/trace_zero{exp_config.strategy}_rank0_{int(time.time())}.json')
                                    if local_rank == 0 else None,
                                    with_stack=True) as prof:

            for i in range(steps):
                batch_input_ids, batch_attention_mask, batch_labels = next(it)
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                get_accelerator().synchronize(device)
                t0 = time.perf_counter()

                outputs = model_engine(input_ids=batch_input_ids,
                                       attention_mask=batch_attention_mask,
                                       labels=batch_labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                model_engine.backward(loss)
                model_engine.step()

                get_accelerator().synchronize(device)
                t1 = time.perf_counter()
                step_time = t1 - t0
                times.append(step_time)
                total_tokens += batch_input_ids.numel()

                if local_rank == 0:
                    print(f"  step {i+1}/{steps}, loss={loss.item():.4f}, step_time={step_time:.4f}s")

                prof.step()
    else:
        for i in range(steps):
            batch_input_ids, batch_attention_mask, batch_labels = next(it)
            batch_input_ids = batch_input_ids.to(device)
            batch_attention_mask = batch_attention_mask.to(device)
            batch_labels = batch_labels.to(device)

            get_accelerator().synchronize(device)
            t0 = time.perf_counter()

            outputs = model_engine(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
            model_engine.backward(loss)
            model_engine.step()

            get_accelerator().synchronize(device)
            t1 = time.perf_counter()
            step_time = t1 - t0
            times.append(step_time)
            total_tokens += batch_input_ids.numel()

            if local_rank == 0:
                print(f"  step {i+1}/{steps}, loss={loss.item():.4f}, step_time={step_time:.4f}s")

    avg_time = sum(times) / len(times)
    throughput = total_tokens / sum(times)
    peak_mem = get_accelerator().max_memory_allocated(device) / (1024**2)

    if local_rank == 0:
        print(f"\nResult: avg_step_time={avg_time:.4f}s, tokens/sec={throughput:.1f}, peak_mem(MB)={peak_mem:.1f}")

    # Cleanup
    del model_engine, optimizer_engine, model, tokenizer, dataset, dataloader
    gc.collect()
    get_accelerator().empty_cache()

    if local_rank == 0 and memory_snapshot:
        snapshot_path = f"{profile_dir}/memory_snapshot_zero{exp_config.strategy}_{int(time.time())}.pickle"
        torch.cuda.memory._dump_snapshot(snapshot_path)  #ignore-cuda
        print(f"\nMemory snapshot saved to: {snapshot_path}")

    # Remove temp config file
    os.unlink(config_path)

    if deepspeed.comm.is_initialized():
        deepspeed.comm.barrier()
        deepspeed.comm.destroy_process_group()

    return ExperimentResult(config=exp_config,
                            avg_step_time=avg_time,
                            tokens_per_sec=throughput,
                            peak_gpu_mem_mb=peak_mem,
                            success=True)

    # except Exception as e:
    #     if local_rank == 0:
    #         print(f"Experiment failed: {str(e)}")

    #     gc.collect()
    #     get_accelerator().empty_cache()

    #     if deepspeed.comm.is_initialized():
    #         try:
    #             deepspeed.comm.barrier()
    #             deepspeed.comm.destroy_process_group()
    #         except:
    #             pass

    #     return ExperimentResult(
    #         config=exp_config,
    #         avg_step_time=0,
    #         tokens_per_sec=0,
    #         peak_gpu_mem_mb=0,
    #         success=False,
    #         error_msg=str(e)
    #     )


def generate_experiment_configs(strategy: float,
                                world_size: int,
                                num_layers: int,
                                batch_sizes: list[int],
                                seq_lens: list[int],
                                strategy_params: list[int],
                                gradient_accumulation_steps: int = 16,
                                checkpointing: bool = False) -> list[ExperimentConfig]:
    """Generate all experiment configurations for parameter sweep"""
    configs = []
    seen = set()

    for batch_size, seq_len, param in itertools.product(batch_sizes, seq_lens, strategy_params):
        total_tokens = batch_size * seq_len
        key = (total_tokens, param)
        if key not in seen:
            seen.add(key)
            if strategy == 3.5:
                config = ExperimentConfig(strategy=strategy,
                                          batch_size=batch_size,
                                          seq_len=seq_len,
                                          world_size=world_size,
                                          num_layers=num_layers,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          num_persistent_layers=param,
                                          checkpointing=checkpointing)
            elif strategy == 4:
                config = ExperimentConfig(strategy=strategy,
                                          batch_size=batch_size,
                                          seq_len=seq_len,
                                          world_size=world_size,
                                          num_layers=num_layers,
                                          gradient_accumulation_steps=gradient_accumulation_steps,
                                          forward_reduce_bucket_size=param,
                                          checkpointing=checkpointing)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            configs.append(config)

    return configs


def save_results(results: list[ExperimentResult], output_path: str):
    """Save experiment results to CSV file"""
    if not results:
        return

    output_path: Path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        fieldnames = list(results[0].to_dict().keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())

    print(f"\nResults saved to {output_path}")


def plot_results(results: list[ExperimentResult], output_dir: str, strategy: str):
    """Generate visualization plots for experiment results"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping visualization")
        return

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful results
    successful_results = [r for r in results if r.success]
    if not successful_results:
        print("No successful results to plot")
        return

    # Extract data
    total_tokens = [r.config.total_tokens for r in successful_results]
    throughput = [r.tokens_per_sec for r in successful_results]
    memory = [r.peak_gpu_mem_mb for r in successful_results]

    if strategy == "zero3.5":
        strategy_param = [r.config.num_persistent_layers for r in successful_results]
        param_name = "Persistent Layers"
    else:  # zero4
        strategy_param = [r.config.forward_reduce_bucket_size for r in successful_results]
        param_name = "Forward Reduce Bucket Size"

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Scatter - Total Tokens vs Throughput (colored by strategy param)
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(total_tokens, throughput, c=strategy_param, cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Total Tokens (batch_size × seq_len)')
    ax1.set_ylabel('Throughput (tokens/sec)')
    ax1.set_title(f'Throughput vs Token Volume\n(colored by {param_name})')
    plt.colorbar(scatter1, ax=ax1, label=param_name)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter - Total Tokens vs Memory (colored by strategy param)
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(total_tokens, memory, c=strategy_param, cmap='plasma', s=100, alpha=0.7)
    ax2.set_xlabel('Total Tokens (batch_size × seq_len)')
    ax2.set_ylabel('Peak GPU Memory (MB)')
    ax2.set_title(f'Memory Usage vs Token Volume\n(colored by {param_name})')
    plt.colorbar(scatter2, ax=ax2, label=param_name)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Scatter - Strategy Param vs Throughput (colored by total tokens)
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(strategy_param, throughput, c=total_tokens, cmap='coolwarm', s=100, alpha=0.7)
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Throughput (tokens/sec)')
    ax3.set_title(f'Throughput vs {param_name}\n(colored by Total Tokens)')
    plt.colorbar(scatter3, ax=ax3, label='Total Tokens')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter - Memory vs Throughput (Pareto frontier exploration)
    ax4 = axes[1, 1]
    scatter4 = ax4.scatter(memory, throughput, c=strategy_param, cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Peak GPU Memory (MB)')
    ax4.set_ylabel('Throughput (tokens/sec)')
    ax4.set_title(f'Memory-Throughput Trade-off\n(colored by {param_name})')
    plt.colorbar(scatter4, ax=ax4, label=param_name)
    ax4.grid(True, alpha=0.3)

    # Add annotations for best points
    best_throughput_idx = np.argmax(throughput)
    ax4.annotate(f'Best Throughput', (memory[best_throughput_idx], throughput[best_throughput_idx]),
                 textcoords="offset points",
                 xytext=(10, 10),
                 ha='left',
                 arrowprops=dict(arrowstyle='->', color='red'))

    plt.tight_layout()
    scatter_path = output_dir / f'{strategy}_scatter_plots.png'
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plots saved to {scatter_path}")

    # Generate heatmaps if we have enough data
    batch_sizes = sorted(set(r.config.batch_size for r in successful_results))
    seq_lens = sorted(set(r.config.seq_len for r in successful_results))
    strategy_params = sorted(
        set(r.config.num_persistent_layers if strategy == "zero3.5" else r.config.forward_reduce_bucket_size
            for r in successful_results))

    if len(strategy_params) > 1 and (len(batch_sizes) > 1 or len(seq_lens) > 1):
        # Create heatmaps for each strategy parameter value
        fig, axes = plt.subplots(len(strategy_params), 2, figsize=(12, 5 * len(strategy_params)))
        if len(strategy_params) == 1:
            axes = [axes]

        for idx, param_val in enumerate(strategy_params):
            param_results = [
                r for r in successful_results
                if (r.config.num_persistent_layers if strategy == "zero3.5" else r.config.forward_reduce_bucket_size
                    ) == param_val
            ]

            # Create matrices for heatmap
            throughput_matrix = np.zeros((len(batch_sizes), len(seq_lens)))
            memory_matrix = np.zeros((len(batch_sizes), len(seq_lens)))

            for r in param_results:
                i = batch_sizes.index(r.config.batch_size)
                j = seq_lens.index(r.config.seq_len)
                throughput_matrix[i, j] = r.tokens_per_sec
                memory_matrix[i, j] = r.peak_gpu_mem_mb

            # Throughput heatmap
            ax_t = axes[idx][0] if len(strategy_params) > 1 else axes[0]
            im_t = ax_t.imshow(throughput_matrix, cmap='YlGn', aspect='auto')
            ax_t.set_xticks(range(len(seq_lens)))
            ax_t.set_xticklabels(seq_lens)
            ax_t.set_yticks(range(len(batch_sizes)))
            ax_t.set_yticklabels(batch_sizes)
            ax_t.set_xlabel('Sequence Length')
            ax_t.set_ylabel('Batch Size')
            ax_t.set_title(f'Throughput (tokens/sec)\n{param_name}={param_val}')
            plt.colorbar(im_t, ax=ax_t)

            # Add text annotations
            for i in range(len(batch_sizes)):
                for j in range(len(seq_lens)):
                    if throughput_matrix[i, j] > 0:
                        ax_t.text(j, i, f'{throughput_matrix[i, j]:.0f}', ha='center', va='center', fontsize=8)

            # Memory heatmap
            ax_m = axes[idx][1] if len(strategy_params) > 1 else axes[1]
            im_m = ax_m.imshow(memory_matrix, cmap='YlOrRd', aspect='auto')
            ax_m.set_xticks(range(len(seq_lens)))
            ax_m.set_xticklabels(seq_lens)
            ax_m.set_yticks(range(len(batch_sizes)))
            ax_m.set_yticklabels(batch_sizes)
            ax_m.set_xlabel('Sequence Length')
            ax_m.set_ylabel('Batch Size')
            ax_m.set_title(f'Peak GPU Memory (MB)\n{param_name}={param_val}')
            plt.colorbar(im_m, ax=ax_m)

            # Add text annotations
            for i in range(len(batch_sizes)):
                for j in range(len(seq_lens)):
                    if memory_matrix[i, j] > 0:
                        ax_m.text(j, i, f'{memory_matrix[i, j]:.0f}', ha='center', va='center', fontsize=8)

        plt.tight_layout()
        heatmap_path = output_dir / f'{strategy}_heatmaps.png'
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Heatmaps saved to {heatmap_path}")

    # Generate summary 3D plot if possible
    if len(strategy_params) > 1 and len(set(total_tokens)) > 1:
        try:
            fig = plt.figure(figsize=(14, 6))

            # 3D Throughput plot
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(total_tokens, strategy_param, throughput, c=throughput, cmap='viridis', s=50)
            ax1.set_xlabel('Total Tokens')
            ax1.set_ylabel(param_name)
            ax1.set_zlabel('Throughput (tokens/sec)')
            ax1.set_title('3D Throughput Analysis')

            # 3D Memory plot
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.scatter(total_tokens, strategy_param, memory, c=memory, cmap='plasma', s=50)
            ax2.set_xlabel('Total Tokens')
            ax2.set_ylabel(param_name)
            ax2.set_zlabel('Peak Memory (MB)')
            ax2.set_title('3D Memory Analysis')

            plt.tight_layout()
            plot_3d_path = output_dir / f'{strategy}_3d_plots.png'
            plt.savefig(plot_3d_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"3D plots saved to {plot_3d_path}")
        except Exception as e:
            print(f"Could not generate 3D plots: {e}")


def print_summary_table(results: list[ExperimentResult], strategy: str):
    """Print a formatted summary table of results"""
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY")
    print("=" * 100)

    if strategy == "zero3.5":
        param_header = "PersistLayers"
    else:
        param_header = "FwdReduceSize"

    header = f"{'Batch':>6} {'SeqLen':>7} {param_header:>13} {'Tokens':>10} {'Time(s)':>10} {'Tok/s':>12} {'Mem(MB)':>10} {'Status':>8}"
    print(header)
    print("-" * 100)

    for r in sorted(
            results,
            key=lambda x:
        (x.config.total_tokens, x.config.num_persistent_layers or x.config.forward_reduce_bucket_size or 0)):
        if strategy == "zero3.5":
            param_val = r.config.num_persistent_layers or 0
        else:
            param_val = r.config.forward_reduce_bucket_size or 0

        status = "✓" if r.success else "✗"
        if r.success:
            print(f"{r.config.batch_size:>6} {r.config.seq_len:>7} {param_val:>13} {r.config.total_tokens:>10} "
                  f"{r.avg_step_time:>10.4f} {r.tokens_per_sec:>12.1f} {r.peak_gpu_mem_mb:>10.1f} {status:>8}")
        else:
            print(f"{r.config.batch_size:>6} {r.config.seq_len:>7} {param_val:>13} {r.config.total_tokens:>10} "
                  f"{'N/A':>10} {'N/A':>12} {'N/A':>10} {status:>8}")

    print("=" * 100)

    # Print best configurations
    successful = [r for r in results if r.success]
    if successful:
        best_throughput = max(successful, key=lambda x: x.tokens_per_sec)
        best_memory = min(successful, key=lambda x: x.peak_gpu_mem_mb)

        print("\nBest Configurations:")
        print(f"  Highest Throughput: {best_throughput.tokens_per_sec:.1f} tokens/sec")
        print(
            f"    - batch={best_throughput.config.batch_size}, seq_len={best_throughput.config.seq_len}, "
            f"param={best_throughput.config.num_persistent_layers or best_throughput.config.forward_reduce_bucket_size}"
        )
        print(f"  Lowest Memory: {best_memory.peak_gpu_mem_mb:.1f} MB")
        print(f"    - batch={best_memory.config.batch_size}, seq_len={best_memory.config.seq_len}, "
              f"param={best_memory.config.num_persistent_layers or best_memory.config.forward_reduce_bucket_size}")


def parse_strategy_params(s: str) -> list[int | tuple[int, int]]:
    """Parse strategy parameters from string.

    Examples:
    - "1" -> [1]
    - "1,2,3" -> [1, 2, 3]
    - 1,2,3 -> [1, 2, 3]
    - "(1,2)" -> [(1,2)]
    - "(1,2),(3,4)" -> [(1,2), (3,4)]
    """
    result = []
    # Use regex to properly split on commas outside parentheses
    parts = re.split(r',\s*(?![^()]*\))', s.strip())

    for part in parts:
        part = part.strip()
        if part.startswith('(') and part.endswith(')'):
            # Parse tuple
            inner = part[1:-1]
            nums = [int(x.strip()) for x in inner.split(',')]
            result.append(tuple(nums))
        else:
            # Parse int
            result.append(int(part))

    return result


def format_strategy_params(params):
    # params: list of ints or tuples
    def fmt(p):
        if isinstance(p, tuple):
            return '_'.join(map(str, p))
        else:
            return str(p)

    return '-'.join(fmt(p) for p in params) if params else 'none'


def main():
    parser = argparse.ArgumentParser(description="ZeRO Experiment")
    parser.add_argument("--model", type=str, default="examples/Qwen3-4B", help="HF model id or path")
    parser.add_argument("--zero", type=float, choices=[2, 3, 3.5, 4], help="ZeRO optimization strategy")
    parser.add_argument("--accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", nargs="+", type=int, default=[4], help="Batch sizes to explore")
    parser.add_argument("--seq_len", nargs="+", type=int, default=[4096], help="Sequence lengths to explore")
    parser.add_argument("--strategy_param",
                        type=parse_strategy_params,
                        default=[],
                        help="Strategy-specific parameters: comma-separated ints or tuples. "
                        "For ZeRO-3.5: num_persistent_layers. "
                        "For ZeRO-4: forward_reduce_bucket_size. "
                        "Examples: 1,2,3 or \"(1,2),(3,4)\"")
    parser.add_argument("--steps", type=int, default=16, help="Number of benchmark steps")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup steps")
    parser.add_argument("--output_dir",
                        type=str,
                        default="examples/results",
                        help="Directory to save results and plots")
    parser.add_argument("--profile_dir",
                        type=str,
                        default="examples/profile",
                        help="Directory to save profiling results")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--dse", action='store_true', help="run DSE or single experiment")
    parser.add_argument("--checkpointing", action='store_true', help="use activation checkpointing or not")
    parser.add_argument("--memory_snapshot", action='store_true', help="Whether to save memory snapshot")
    parser.add_argument("--profile", action="store_true", help="Whether to use torch profiler")
    parser.add_argument("--log-level",
                        type=str,
                        default="info",
                        choices=["debug", "info", "warning", "error", "critical"])
    args = parser.parse_args()

    set_log_level_from_string(args.log_level)

    world_size = get_accelerator().device_count() if get_accelerator().is_available() else 1
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    if args.profile or args.memory_snapshot:
        os.makedirs(args.profile_dir, exist_ok=True)

    if args.dse:
        # Generate multiple experiment configurations for parameter sweep
        configs = generate_experiment_configs(strategy=args.zero,
                                              world_size=world_size,
                                              num_layers=num_layers,
                                              batch_sizes=args.batch_size,
                                              seq_lens=args.seq_len,
                                              strategy_params=args.strategy_param,
                                              gradient_accumulation_steps=args.accumulation_steps,
                                              checkpointing=args.checkpointing)
    else:
        # Generate a single experiment configuration using the first values from lists
        single_config = ExperimentConfig(
            strategy=args.zero,
            batch_size=args.batch_size[0],
            seq_len=args.seq_len[0],
            world_size=world_size,
            num_layers=num_layers,
            gradient_accumulation_steps=args.accumulation_steps,
            num_persistent_layers=args.strategy_param[0] if args.strategy_param and args.zero == 3.5 else None,
            forward_reduce_bucket_size=args.strategy_param[0] if args.strategy_param and args.zero == 4 else None,
            checkpointing=args.checkpointing)
        configs = [single_config]

    if args.local_rank == 0 or args.local_rank == -1:
        print(f"\n{'='*60}")
        print(f"ZeRO Experiment")
        print(f"{'='*60}")
        print(f"Strategy: ZeRO-{args.zero}")
        print(f"Model: {args.model}")
        print(f"Batch sizes: {args.batch_size}")
        print(f"Sequence lengths: {args.seq_len}")
        print(f"Strategy parameters: {args.strategy_param}")
        print(f"Total experiments: {len(configs)}")
        print(f"{'='*60}\n")

    # Run experiments
    results = []
    for i, exp_config in enumerate(configs):
        if args.local_rank == 0 or args.local_rank == -1:
            print(f"\n[{i+1}/{len(configs)}] Running experiment...")

        result = run_single_experiment(model_name=args.model,
                                       exp_config=exp_config,
                                       steps=args.steps,
                                       warmup=args.warmup,
                                       local_rank=args.local_rank,
                                       memory_snapshot=args.memory_snapshot,
                                       profile=args.profile,
                                       profile_dir=args.profile_dir)
        results.append(result)

    # Only save and plot on rank 0
    if args.local_rank == 0 or args.local_rank == -1:
        # Save results to CSV
        timestamp = int(time.time())
        csv_path = f"{args.output_dir}/ZeRO-{args.zero}_batch{'-'.join(map(str, args.batch_size))}_seq{'-'.join(map(str, args.seq_len))}_param{format_strategy_params(args.strategy_param)}_results_{timestamp}.csv"
        save_results(results, csv_path)

        if args.dse:
            # Print summary table
            print_summary_table(results, args.zero)

            # Generate plots
            plot_results(results, args.output_dir, args.zero)

        print(f"\n{'='*60}")
        print("Experiment completed!")
        print(f"Results saved to: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
