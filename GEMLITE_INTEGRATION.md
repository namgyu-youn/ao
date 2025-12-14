# GemLite Kernel Support for Int4 Quantization

This document describes the integration of GemLite Triton kernels into TorchAO's int4 quantization workflow.

## Overview

GemLite is a high-performance Triton kernel library for low-bit (1, 2, 4, 8-bit) matrix multiplication. This integration adds GemLite as a new packing format option for int4 weight-only quantization in TorchAO.

## Implementation Summary

### Files Added

1. **`torchao/quantization/quantize_/workflows/int4/int4_gemlite_tensor.py`**
   - New `Int4GemliteTensor` tensor subclass following the v2 architecture
   - Implements quantization using GemLite's symmetric quantization scheme
   - Forward pass uses `gemlite.core.forward_functional` with automatic kernel selection
   - Supports groupwise quantization with configurable group sizes

2. **`test/quantization/quantize_/workflows/int4/test_int4_gemlite_tensor.py`**
   - Comprehensive unit tests for `Int4GemliteTensor`
   - Tests for various batch sizes, dtypes, and tensor shapes
   - Validation of GemLite constraints (group size ≥ 16, divisibility requirements)

3. **`examples/gemlite_int4_example.py`**
   - Usage example demonstrating GemLite integration
   - Shows automatic kernel selection for different batch sizes
   - Includes accuracy comparison and compression ratio analysis

### Files Modified

1. **`torchao/quantization/quantize_/workflows/int4/int4_packing_format.py`**
   - Added `GEMLITE = "gemlite"` to `Int4PackingFormat` enum

2. **`torchao/quantization/quant_api.py`**
   - Added import for `Int4GemliteTensor`
   - Added case in `_int4_weight_only_quantize_tensor()` to handle GEMLITE format

3. **`torchao/quantization/quantize_/workflows/__init__.py`**
   - Added `Int4GemliteTensor` to imports and exports

## Features

### Automatic Kernel Selection

GemLite automatically selects the optimal kernel based on batch size:

- **Batch size = 1**: `GEMV_REVSPLITK` (optimized for single-batch inference)
- **Batch size 2-64**: `GEMM_SPLITK` (optimized for small batch decoding)
- **Batch size > 64**: `GEMM` (optimized for compute-bound scenarios)

### Supported Configurations

- **Weight precision**: 4-bit (with support for 1, 2, 8-bit in GemLite core)
- **Activation dtype**: FP16, BF16, FP32
- **Quantization granularity**: Groupwise (group_size ≥ 16, divisible by 32)
- **Quantization scheme**: Symmetric with shift (asymmetric-like)

### Requirements

- **Hardware**: CUDA-capable GPU (compute capability ≥ 7.0)
- **Software**: Triton compiler
- **Constraints**:
  - `in_features` must be divisible by 32
  - `group_size` must be ≥ 16
  - `in_features` must be divisible by `group_size`

## Usage

```python
import torch
from torchao.quantization import Int4WeightOnlyConfig, quantize_

# Create a linear layer
linear = torch.nn.Linear(512, 1024, dtype=torch.bfloat16, device="cuda")

# Configure GemLite quantization
config = Int4WeightOnlyConfig(
    group_size=128,
    int4_packing_format="gemlite",
)

# Quantize the model
quantize_(linear, config)

# Use the quantized model
input = torch.randn(8, 512, dtype=torch.bfloat16, device="cuda")
output = linear(input)
```

## Implementation Details

### Quantization Process

The `Int4GemliteTensor.from_hp()` method implements the following quantization pipeline:

1. **Validation**: Check divisibility and group size constraints
2. **Groupwise quantization**: Reshape weights to (N, K/group_size, group_size)
3. **Compute scales**: Per-group max absolute value / 7.0
4. **Quantize**: Symmetric quantization to [-7, 7], then shift to [0, 15]
5. **Pack**: Use GemLite's bitpacking utilities to pack 4-bit values
6. **Store metadata**: Save quantization parameters for forward pass

### Forward Pass

The linear operation is implemented via `forward_functional`:

```python
result = forward_functional(
    x,                      # Input tensor
    bias,                   # Bias (optional)
    [W_q, scales, zeros],  # Tensor args (packed weights, scales, zero points)
    metadata,               # Meta args (quantization config)
    matmul_type=-1,        # -1 = automatic kernel selection
)
```

### Metadata Format

The `gemlite_metadata` list contains:
- `scaled_activations`: Whether to use dynamic activation quantization (False for weight-only)
- `W_nbits`: 4
- `group_size`: Quantization group size
- `unpack_mask`: 15 (2^4 - 1)
- `elements_per_sample`: Packing density
- `input_dtype`, `output_dtype`, `acc_dtype`, `meta_dtype`: Data type information
- `channel_scale_mode`: 0 (weight-only quantization)
- `W_group_mode`: 3 (asymmetric quantization: (Wq - zeros) * scales)
- `data_contiguous`: 1 (ensure contiguous memory layout)

## Comparison with Other Formats

| Feature | PLAIN | PRESHUFFLED | TILE_PACKED_TO_4D | **GEMLITE** |
|---------|-------|-------------|-------------------|-------------|
| Kernel | FBGEMM | FBGEMM (H100) | Tinygemm | Triton (custom) |
| Bit widths | 4 | 4 | 4 | **1, 2, 4, 8** |
| Group sizes | Any | Any | Any | **≥16, div by 32** |
| Activation types | BF16 | BF16, FP8 | BF16 | **FP16, BF16, FP8, INT8** |
| Batch optimization | No | No | Limited | **Yes (auto)** |
| Portability | CUDA | H100+ | SM 8.0+ | **CUDA 7.0+** |

### Advantages of GemLite

1. **Automatic kernel selection**: Optimal performance across different batch sizes
2. **Broad hardware support**: Works on GPUs with compute capability ≥ 7.0
3. **Extensibility**: Easy to add support for other bit widths (1, 2, 8-bit)
4. **Future-ready**: Built-in support for dynamic quantization and MXFP formats

## Testing

Run the tests with:

```bash
pytest test/quantization/quantize_/workflows/int4/test_int4_gemlite_tensor.py -v
```

Run the example:

```bash
python examples/gemlite_int4_example.py
```

## Future Extensions

1. **Extended bit widths**: Enable 1, 2, 8-bit weight quantization
2. **Dynamic activation quantization**: Add A8W4 support
3. **MXFP formats**: Integrate MXFP4/MXFP8 for Blackwell GPUs
4. **Config caching**: Integrate GemLite's kernel config caching for faster startup
5. **Kernel tuning**: Expose kernel selection parameters for advanced users

## References

- GemLite repository: `/gemlite/`
- TorchAO int4 workflows: `torchao/quantization/quantize_/workflows/int4/`
- Original int4 tensor implementation: `int4_tensor.py`
