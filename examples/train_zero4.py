# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import time
import argparse
import gc

import torch
from torch.utils.data import DataLoader, TensorDataset
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer


def benchmark_config(model_name, config_path, batch_size=2, seq_len=128, steps=30, warmup=5):
    print(f"\n=== Benchmarking {config_path} ===")
    deepspeed.init_distributed()

    # load model/tokenizer on CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 trust_remote_code=True,
                                                 device_map="cpu",
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)

    # simple synthetic dataset
    texts = ["Hello world"] * (batch_size * (steps + warmup))
    enc = tokenizer(texts, padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # initialize deepspeed
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(model=model,
                                                                model_parameters=model.parameters(),
                                                                config=config_path)

    device = model_engine.device
    use_cuda = device.type == "cuda"

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
    if use_cuda:
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
    for i in range(steps):
        batch_input_ids, batch_attention_mask, batch_labels = next(it)
        batch_input_ids = batch_input_ids.to(device)
        batch_attention_mask = batch_attention_mask.to(device)
        batch_labels = batch_labels.to(device)

        if use_cuda:
            get_accelerator().synchronize(device)
        t0 = time.perf_counter()

        outputs = model_engine(input_ids=batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        model_engine.backward(loss)
        model_engine.step()

        if use_cuda:
            get_accelerator().synchronize(device)
        t1 = time.perf_counter()
        step_time = t1 - t0
        times.append(step_time)
        total_tokens += batch_input_ids.numel()  # batch_size * seq_len

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  step {i+1}/{steps}, loss={loss.item():.4f}, step_time={step_time:.4f}s")

    avg_time = sum(times) / len(times)
    throughput = total_tokens / sum(times)  # tokens / sec
    peak_mem = None
    if use_cuda:
        peak_mem = get_accelerator().max_memory_allocated(device) / (1024**2)  # MB

    print(
        f"Result for {config_path}: avg_step_time={avg_time:.4f}s, tokens/sec={throughput:.1f}, peak_gpu_mem(MB)={peak_mem}"
    )

    # cleanup
    del model_engine, optimizer_engine, model, tokenizer, dataset, dataloader
    gc.collect()
    if use_cuda:
        get_accelerator().empty_cache()
    return {
        "config": config_path,
        "avg_step_time": avg_time,
        "tokens_per_sec": throughput,
        "peak_gpu_mem_mb": peak_mem
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="examples/Qwen3-0.6B", help="HF model id or path")
    parser.add_argument("--configs", nargs="+", default=["examples/zero2.json"])
    parser.add_argument("--offload", action="store_true", help="Whether to use offload configs")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    args = parser.parse_args()

    if args.offload:
        args.configs = [cfg.replace('.json', '_offload.json') for cfg in args.configs]

    results = []
    for cfg in args.configs:
        res = benchmark_config(model_name=args.model,
                               config_path=cfg,
                               batch_size=args.batch_size,
                               seq_len=args.seq_len,
                               steps=args.steps,
                               warmup=args.warmup)
        results.append(res)

    print("\n=== Summary ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
