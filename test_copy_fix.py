# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3
"""Test script to verify the copy_ fix for regular tensors into quantized tensors."""

import torch

from torchao.quantization.quantize_.workflows.int4.int4_tile_packed_to_4d_tensor import (
    Int4TilePackedTo4dTensor,
)


def test_copy_regular_to_quantized():
    """Test copying a regular tensor into a quantized tensor."""
    print("Testing copy_ fix for Int4TilePackedTo4dTensor...")

    # Create a regular high-precision tensor
    hp_tensor = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    print(f"Created HP tensor: shape={hp_tensor.shape}, dtype={hp_tensor.dtype}")

    # Create a quantized version
    block_size = [1, 128]
    quantized_tensor = Int4TilePackedTo4dTensor.from_hp(hp_tensor, block_size)
    print(f"Created quantized tensor: {type(quantized_tensor).__name__}")

    # Create another regular tensor to copy in (simulates vLLM loading weights)
    new_hp_tensor = torch.randn(256, 128, dtype=torch.bfloat16, device="cuda")
    print(f"Created new HP tensor to copy: shape={new_hp_tensor.shape}")

    # This should work now with our fix!
    try:
        quantized_tensor.copy_(new_hp_tensor)
        print("✓ SUCCESS: copy_ worked! Regular tensor was quantized and copied.")
        return True
    except Exception as e:
        print(f"✗ FAILED: copy_ raised {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    if torch.cuda.is_available():
        success = test_copy_regular_to_quantized()
        exit(0 if success else 1)
    else:
        print("CUDA not available, skipping test")
        exit(0)
