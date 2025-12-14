# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

from torchao.quantization import Int4WeightOnlyConfig, quantize_
from torchao.quantization.utils import compute_error
from torchao.testing.utils import TorchAOIntegrationTestCase
from torchao.utils import torch_version_at_least


@unittest.skipIf(not torch_version_at_least("2.8.0"), "Need pytorch 2.8+")
@unittest.skipIf(not torch.cuda.is_available(), "Need CUDA available")
class TestInt4GemliteTensor(TorchAOIntegrationTestCase):
    def setUp(self):
        self.config = Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format="gemlite",
        )
        self.GPU_DEVICES = ["cuda"] if torch.cuda.is_available() else []

    @parametrize("group_size", [64, 128])
    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_linear(self, group_size, dtype):
        """Test basic linear layer quantization and inference."""
        device = "cuda"
        config = Int4WeightOnlyConfig(
            group_size=group_size,
            int4_packing_format="gemlite",
        )

        # Create a simple linear layer
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)

        # Get original output
        original = linear(input)

        # Quantize using GemLite
        quantize_(linear, config)

        # Get quantized output
        quantized = linear(input)

        # Check that quantization error is reasonable (SQNR > 20 dB)
        self.assertTrue(compute_error(original, quantized) > 20)

    @parametrize("batch_size", [1, 8, 128])
    def test_batch_size_dispatch(self, batch_size):
        """Test that GemLite correctly dispatches kernels for different batch sizes."""
        dtype = torch.bfloat16
        device = "cuda"

        # Create linear layer and quantize
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, self.config)

        # Test with different batch sizes
        input = torch.randn(batch_size, 128, dtype=dtype, device=device)
        output = linear(input)

        # Check output shape is correct
        self.assertEqual(output.shape, (batch_size, 256))
        self.assertEqual(output.dtype, dtype)

    def test_divisibility_constraints(self):
        """Test that GemLite enforces divisibility constraints."""
        dtype = torch.bfloat16
        device = "cuda"

        # K (in_features) must be divisible by 32
        with self.assertRaises(ValueError):
            linear = torch.nn.Linear(100, 256, dtype=dtype, device=device)
            quantize_(linear, self.config)

    def test_group_size_constraints(self):
        """Test that GemLite enforces minimum group size."""
        dtype = torch.bfloat16
        device = "cuda"

        # Group size must be >= 16
        config = Int4WeightOnlyConfig(
            group_size=8,  # Too small
            int4_packing_format="gemlite",
        )

        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)

        with self.assertRaises(ValueError):
            quantize_(linear, config)

    def test_with_bias(self):
        """Test that bias is handled correctly."""
        dtype = torch.bfloat16
        device = "cuda"

        # Create linear layer with bias
        input = torch.randn(1, 128, dtype=dtype, device=device)
        linear = torch.nn.Linear(128, 256, bias=True, dtype=dtype, device=device)

        # Get original output
        original = linear(input)

        # Quantize
        quantize_(linear, self.config)

        # Get quantized output
        quantized = linear(input)

        # Check that bias is preserved
        self.assertTrue(compute_error(original, quantized) > 20)

    def test_multiple_forward_passes(self):
        """Test that multiple forward passes produce consistent results."""
        dtype = torch.bfloat16
        device = "cuda"

        # Create and quantize linear layer
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, self.config)

        # Create input
        input = torch.randn(4, 128, dtype=dtype, device=device)

        # Run multiple forward passes
        output1 = linear(input)
        output2 = linear(input)

        # Results should be identical
        self.assertTrue(torch.allclose(output1, output2))

    def test_contiguity(self):
        """Test that GemLite requires contiguous tensors."""
        dtype = torch.bfloat16
        device = "cuda"

        # Create and quantize linear layer
        linear = torch.nn.Linear(128, 256, dtype=dtype, device=device)
        quantize_(linear, self.config)

        # Create contiguous input
        input = torch.randn(4, 128, dtype=dtype, device=device)

        # Should work fine with contiguous input
        output = linear(input)
        self.assertEqual(output.shape, (4, 256))

    @parametrize("in_features,out_features", [(128, 256), (256, 512), (512, 1024)])
    def test_various_shapes(self, in_features, out_features):
        """Test GemLite with various tensor shapes."""
        dtype = torch.bfloat16
        device = "cuda"

        # Create linear layer
        linear = torch.nn.Linear(in_features, out_features, dtype=dtype, device=device)

        # Quantize
        quantize_(linear, self.config)

        # Test inference
        input = torch.randn(2, in_features, dtype=dtype, device=device)
        output = linear(input)

        self.assertEqual(output.shape, (2, out_features))
        self.assertEqual(output.dtype, dtype)


instantiate_parametrized_tests(TestInt4GemliteTensor)


if __name__ == "__main__":
    run_tests()
