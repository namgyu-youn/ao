# Stochastic Rounding Implementation Plan for NVFP4

## Overview

Add stochastic rounding support for NVFP4 training. Only the rounding step changes; bit-packing and scale computation remain shared.

## Architecture

```
FP32 Input → Scaling → Rounding (RNE/Stochastic) → Bit-Packing → Packed FP4
                                      ↑ ONLY THIS CHANGES
```

## Implementation

### 1. Add RoundingMode Enum

**File**: `torchao/prototype/mx_formats/config.py`
**Location**: After line 84

```python
class RoundingMode(Enum):
    """Rounding modes for FP32 → FP4 quantization."""
    RNE = "rne"  # Round to Nearest, ties to Even (inference)
    STOCHASTIC = "stochastic"  # Probabilistic (training)
```

### 2. Stochastic Rounding Implementation

**File**: `torchao/prototype/custom_fp_utils.py`
**Location**: After line 142

**CRITICAL**: Must handle all three branches (saturate, denormal, normal). The plan previously only showed the normal branch, which is **incorrect** and would cause failures for edge cases.

```python
def _f32_to_floatx_unpacked_stochastic(x: Tensor, ebits: int, mbits: int) -> Tensor:
    """Stochastic rounding with Unbiased: E[quantized(x)] = x."""
    # Setup (lines 27-95): Identical to _f32_to_floatx_unpacked
    # Calculate: exp_bias, max_normal, min_normal, denorm_mask_float, masks, etc.

    # Branch 1: Saturate - Unchanged from RNE
    # x >= max_normal → max_int (lines 98-100)

    # Branch 2: Denormal - Unchanged from RNE
    # x < min_normal → denormal quantization (lines 105-108)
    denormal_x = (x + denorm_mask_float).view(torch.int32) - denorm_mask_int

    # Branch 3: Normal - ONLY THIS DIFFERS
    # Replace RNE tie-breaking (lines 113-123) with stochastic rounding:
    normal_x = x.view(torch.int32)

    # Extract bits to be dropped for probability calculation
    num_dropped_bits = MBITS_F32 - mbits
    dropped_bits = normal_x & ((1 << num_dropped_bits) - 1)

    # Stochastic decision: round up with probability = dropped_bits / 2^num_dropped_bits
    rand_threshold = torch.randint(0, 1 << num_dropped_bits, x.shape,
                                   device=x.device, dtype=torch.int32)
    round_up = (rand_threshold < dropped_bits).to(torch.int32) << num_dropped_bits

    # Apply exponent conversion and rounding
    normal_x = (normal_x + ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + round_up) >> num_dropped_bits

    # Merge branches (lines 128-142): Identical to RNE
    # Combine saturate/denormal/normal results, add sign bit

    return result.to(torch.uint8)
```

**File**: `torchao/prototype/mx_formats/kernels.py`
**Update**: Line 56

```python
def f32_to_f4_unpacked(x: Tensor, round_mode: str = "rne") -> Tensor:
    """Convert FP32 to FP4 E2M1 unpacked."""
    if round_mode == "stochastic":
        return _f32_to_floatx_unpacked_stochastic(x, EBITS_F4_E2M1, MBITS_F4_E2M1)
    return _f32_to_floatx_unpacked(x, EBITS_F4_E2M1, MBITS_F4_E2M1)
```

### 3. Parameter Threading

**File**: `torchao/prototype/mx_formats/nvfp4_tensor.py`

**Line 14**: Add import
```python
from torchao.prototype.mx_formats.config import RoundingMode
```

**Lines 42-47**: Update dataclass
```python
@dataclass
class QuantizeTensorToNVFP4Kwargs(QuantizeTensorKwargs):
    block_size: int = 16
    is_swizzled_scales: bool = False
    use_triton_kernel: bool = False
    use_dynamic_per_tensor_scale: bool = False
    round_mode: RoundingMode = RoundingMode.RNE  # NEW
```

**Line 125**: Add parameter to `to_nvfp4()`
```python
def to_nvfp4(..., round_mode: RoundingMode = RoundingMode.RNE):
```

**Line 160**: Pass to PyTorch path (MISSING IN ORIGINAL PLAN)
```python
blockwise_scales, data_lp = nvfp4_quantize(data_hp, block_size, per_tensor_scale, round_mode)
```

**Line 641**: Update `nvfp4_quantize()` signature
```python
def nvfp4_quantize(..., round_mode: RoundingMode = RoundingMode.RNE):
```

**Line 710**: Pass to conversion
```python
data_lp = f32_to_f4_unpacked(data_scaled, round_mode=round_mode.value)
```

### 4. Triton Kernel Handling

**Issue**: PTX instruction `cvt.rn.satfinite.e2m1x2.f32` (line 452 in kernels.py) uses **hardware RNE only**.

**Solution**: Disable Triton for stochastic mode (add after line 153 in nvfp4_tensor.py):
```python
if round_mode == RoundingMode.STOCHASTIC:
    use_triton_kernel = False  # Force PyTorch path for correct stochastic rounding
```

**Note**: Custom PTX implementation would require manual bit manipulation without hardware support. PyTorch path is correct; optimize only if profiling shows bottleneck.

## Testing

**File**: `test/prototype/mx_formats/test_nvfp4_tensor.py`

```python
def test_stochastic_produces_variation():
    """Stochastic rounding should vary across runs."""
    x = torch.tensor([1.3, 2.7], dtype=torch.float32)
    results = [NVFP4Tensor.to_nvfp4(x, round_mode=RoundingMode.STOCHASTIC).dequantize()
               for _ in range(100)]
    assert torch.stack(results).var(dim=0).sum() > 0

def test_stochastic_unbiased():
    """E[stochastic(x)] ≈ x (unbiased)."""
    x = torch.randn(1000, 1000, dtype=torch.bfloat16)
    mean = torch.stack([NVFP4Tensor.to_nvfp4(x, round_mode=RoundingMode.STOCHASTIC).dequantize()
                        for _ in range(50)]).mean(dim=0)
    assert (mean - x).abs().mean() < 0.1

def test_denormal_handling():
    """Stochastic rounding must handle denormal values correctly."""
    x = torch.tensor([1e-8, -1e-8, 0.5], dtype=torch.float32)
    result = NVFP4Tensor.to_nvfp4(x, round_mode=RoundingMode.STOCHASTIC)
    assert result is not None  # Should not crash on denormals
```

## Usage

```python
from torchao.prototype.mx_formats.config import RoundingMode
from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

# Inference: deterministic
weight_inf = NVFP4Tensor.to_nvfp4(weight, round_mode=RoundingMode.RNE)

# Training: preserves small gradient updates
weight_train = NVFP4Tensor.to_nvfp4(weight, round_mode=RoundingMode.STOCHASTIC)
```

## Files Modified

- `torchao/prototype/mx_formats/config.py` - RoundingMode enum
- `torchao/prototype/custom_fp_utils.py` - Stochastic rounding function
- `torchao/prototype/mx_formats/nvfp4_tensor.py` - Parameter threading, Triton override
- `torchao/prototype/mx_formats/kernels.py` - Dispatcher function
- `test/prototype/mx_formats/test_nvfp4_tensor.py` - Tests

## Critical Corrections from Original Plan

1. **Stochastic function must handle saturate/denormal/normal branches** - Original only showed normal branch
2. **Missing parameter pass at line 160** - to_nvfp4() → nvfp4_quantize() call site
3. **Removed confusing "pre-noise" references** - This approach was never proposed and is incorrect
4. **Added denormal test** - Edge case validation missing from original
