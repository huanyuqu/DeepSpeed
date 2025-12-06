# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import argparse
import gc
import json
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from liger_kernel.transformers import apply_liger_kernel_to_qwen3

apply_liger_kernel_to_qwen3()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def benchmark_config(model_name,
                     config_path,
                     level,
                     batch_size=2,
                     seq_len=128,
                     steps=30,
                     warmup=5,
                     local_rank=None,
                     memory_snapshot=False,
                     profile=False):
    if local_rank == 0:
        print(f"\n=== Benchmarking {config_path} ===")

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, config=config, trust_remote_code=True)

    # Record memory history
    if memory_snapshot:
        torch.cuda.memory._record_memory_history(max_entries=100000)  #ignore-cuda

    deepspeed.init_distributed()
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    if config_dict.get("zero_optimization", {}).get("stage") == 3:
        with deepspeed.zero.Init():
            model = AutoModelForCausalLM.from_config(config=config, trust_remote_code=True, dtype=torch.float16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, dtype=torch.float16)

    if local_rank == 0:
        print("Initialization completed")

    model.gradient_checkpointing_enable()
    if local_rank == 0:
        print("Recomputation enabled")

    # simple synthetic dataset
    texts = ["Hello world"] * (batch_size * (steps + warmup))
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # initialize deepspeed
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=3e-7)
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(model=model,
                                                                optimizer=optimizer,
                                                                model_parameters=model.parameters(),
                                                                config=config_path)

    model_engine.train()
    device = model_engine.device

    # prepare iterator
    it = iter(dataloader)

    # warmup
    for i in range(warmup):
        batch_input_ids, batch_attention_mask, batch_labels = next(it)
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model_engine(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        model_engine.backward(loss)
        model_engine.step()

    # reset CUDA peak memory counters if applicable
    get_accelerator().reset_peak_memory_stats(device)

    # Reset DeepSpeed internal step counters to align gradient accumulation
    if hasattr(model_engine, "global_steps"):
        model_engine.global_steps = 0
    if hasattr(model_engine, "micro_steps"):
        model_engine.micro_steps = 0
    # Try to reset optimizer state if supported
    if hasattr(model_engine.optimizer, "zero_grad"):
        model_engine.optimizer.zero_grad(set_to_none=True)

    times = []
    total_tokens = 0

    if profile:
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
        ],
                                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
                                    on_trace_ready=lambda prof: prof.export_chrome_trace(
                                        f'examples/trace_zero{level}_rank{local_rank}_{int(time.time())}.json'),
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
                total_tokens += batch_input_ids.numel()  # batch_size * seq_len

                # if (i + 1) % 10 == 0 or i == 0:
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
            total_tokens += batch_input_ids.numel()  # batch_size * seq_len

            # if (i + 1) % 10 == 0 or i == 0:
            if local_rank == 0:
                print(f"  step {i+1}/{steps}, loss={loss.item():.4f}, step_time={step_time:.4f}s")

    avg_time = sum(times) / len(times)
    throughput = total_tokens / sum(times)  # tokens / sec
    peak_mem = None
    peak_mem = get_accelerator().max_memory_allocated(device) / (1024**2)  # MB

    if local_rank == 0:
        print(
            f"Result for {config_path}: avg_step_time={avg_time:.4f}s, tokens/sec={throughput:.1f}, peak_gpu_mem(MB)={peak_mem}"
        )

    # cleanup
    del model_engine, optimizer_engine, model, tokenizer, dataset, dataloader
    gc.collect()
    get_accelerator().empty_cache()
    if local_rank == 0:
        if memory_snapshot:
            torch.cuda.memory._dump_snapshot(  #ignore-cuda
                f"examples/memory_snapshot_zero{level}_{int(time.time())}.pickle")  #ignore-cuda

    # Ensure all ranks reach this point before destroying the process group
    if deepspeed.comm.is_initialized():
        deepspeed.comm.barrier()
        deepspeed.comm.destroy_process_group()

    return {
        "config": config_path,
        "avg_step_time": avg_time,
        "tokens_per_sec": throughput,
        "peak_gpu_mem_mb": peak_mem
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="examples/Qwen3-4B", help="HF model id or path")
    parser.add_argument("--configs", nargs="+", default=["examples/zero3.json"])
    parser.add_argument("--offload", action="store_true", help="Whether to use offload configs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser.add_argument("--memory_snapshot", action="store_true", help="Whether to save memory snapshot")
    parser.add_argument("--profile", action="store_true", help="Whether to use torch profiler")
    args = parser.parse_args()

    if args.offload:
        args.configs = [cfg.replace('.json', '_offload.json') for cfg in args.configs]

    results = []
    for cfg in args.configs:
        res = benchmark_config(model_name=args.model,
                               config_path=cfg,
                               level=3,
                               batch_size=args.batch_size,
                               seq_len=args.seq_len,
                               steps=args.steps,
                               warmup=args.warmup,
                               local_rank=args.local_rank,
                               memory_snapshot=args.memory_snapshot,
                               profile=args.profile)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
