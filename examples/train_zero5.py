#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Example: Training with Forward Reduce Strategy

Forward Reduce Strategy:
During gradient accumulation, some gradient reduce-scatter operations
are deferred to the forward pass of the next micro-batch.
This utilizes idle communication bandwidth during forward pass to overlap communication and computation.
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

    # Use DeepSpeed to initialize model and optimizer
    # Key: forward_reduce is enabled in config_file
    model_engine, optimizer_engine, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config="zero5_config.json"  # Use the new config file
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
