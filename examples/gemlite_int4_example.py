#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: Using GemLite kernels for Int4 weight-only quantization

This example demonstrates how to use the GemLite kernel backend for int4 quantization.
GemLite provides:
- Automatic kernel selection based on batch size
- Support for 1, 2, 4, 8-bit weights
- Groupwise quantization with configurable group sizes (≥16)
- Triton-based kernels optimized for different batch sizes

Requirements:
- CUDA-capable GPU
- GemLite package (included in the repo at /gemlite/)
"""

import torch

from torchao.quantization import Int4WeightOnlyConfig, quantize_


def main():
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available. GemLite requires a CUDA device.")
        return

    device = "cuda"
    dtype = torch.bfloat16

    print("=" * 80)
    print("GemLite Int4 Quantization Example")
    print("=" * 80)

    # Create a simple model
    print("\n1. Creating a simple linear layer...")
    linear = torch.nn.Linear(512, 1024, bias=True, dtype=dtype, device=device)
    print(
        f"   Original model size: {linear.weight.numel() * linear.weight.element_size() / 1024:.2f} KB"
    )

    # Create GemLite quantization config
    print("\n2. Creating GemLite quantization config...")
    config = Int4WeightOnlyConfig(
        group_size=128,  # Group size for quantization (must be ≥16, divisible by 32)
        int4_packing_format="gemlite",  # Use GemLite kernels
    )
    print(
        f"   Config: group_size={config.group_size}, format={config.int4_packing_format}"
    )

    # Quantize the model
    print("\n3. Quantizing the model with GemLite...")
    quantize_(linear, config)

    # Calculate quantized model size
    qdata_size = linear.weight.qdata.numel() * linear.weight.qdata.element_size()
    scale_size = linear.weight.scale.numel() * linear.weight.scale.element_size()
    zero_size = (
        linear.weight.zero_point.numel() * linear.weight.zero_point.element_size()
    )
    total_size = (qdata_size + scale_size + zero_size) / 1024

    print(f"   Quantized model size: {total_size:.2f} KB")
    print(f"   Compression ratio: {(512 * 1024 * 2 / 1024) / total_size:.2f}x")

    # Test with different batch sizes
    print("\n4. Testing with different batch sizes (GemLite auto-selects kernels)...")
    batch_sizes = [1, 4, 16, 64, 128]

    for batch_size in batch_sizes:
        input_tensor = torch.randn(batch_size, 512, dtype=dtype, device=device)
        output = linear(input_tensor)

        # Determine which kernel would be selected
        if batch_size == 1:
            kernel_type = "GEMV_REVSPLITK (optimized for single batch)"
        elif batch_size <= 64:
            kernel_type = "GEMM_SPLITK (optimized for small batches)"
        else:
            kernel_type = "GEMM (optimized for large batches)"

        print(
            f"   Batch size {batch_size:3d}: output shape {tuple(output.shape)}, kernel: {kernel_type}"
        )

    # Compare accuracy
    print("\n5. Accuracy comparison...")
    linear_fp = torch.nn.Linear(512, 1024, bias=True, dtype=dtype, device=device)
    linear_fp.weight.data.copy_(linear.weight.dequantize())
    if linear.bias is not None:
        linear_fp.bias.data.copy_(linear.bias)

    test_input = torch.randn(8, 512, dtype=dtype, device=device)

    with torch.no_grad():
        output_fp = linear_fp(test_input)
        output_q = linear(test_input)

    mse = torch.mean((output_fp - output_q) ** 2).item()
    max_error = torch.max(torch.abs(output_fp - output_q)).item()

    print(f"   Mean Squared Error: {mse:.6f}")
    print(f"   Max Absolute Error: {max_error:.6f}")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
