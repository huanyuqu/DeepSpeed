#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example: Training with ZeRO-4 strategy

ZeRO-4 is an optimization of ZeRO-3:
- ZeRO-3 (standard): Release parameters after backward, re-gather before forward
- ZeRO-4 (this implementation): Retain parameters after backward, reuse directly in forward, release after forward

This avoids redundant all-gather operations in subsequent forward passes.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import deepspeed


# Simple model example
class SimpleModel(nn.Module):

    def __init__(self, hidden_size=768, num_layers=12):
        super().__init__()
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size))
            for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.layers(x)
        x = self.final(x)
        return x


def main():
    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Model, optimizer, data
    model = SimpleModel(hidden_size=768, num_layers=12)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create dummy data
    dataset = TensorDataset(torch.randn(1000, 768), torch.randint(0, 10, (1000, )))
    dataloader = DataLoader(dataset, batch_size=4)

    # Initialize model and optimizer with DeepSpeed
    # Key: partition_params_backward=false in config_file enables ZeRO-4
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="zero4_config.json"  # Use the config file created above
    )

    # Training loop
    num_epochs = 2
    for epoch in range(num_epochs):
        for step, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(model_engine.device)
            targets = targets.to(model_engine.device)

            # Forward pass
            outputs = model_engine(inputs)
            loss = nn.CrossEntropyLoss()(outputs, targets)

            # Backward pass
            model_engine.backward(loss)
            model_engine.step()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    print("Training completed!")


if __name__ == "__main__":
    main()
