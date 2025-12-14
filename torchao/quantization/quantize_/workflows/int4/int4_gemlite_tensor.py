# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from torchao.utils import TorchAOBaseTensor

__all__ = [
    "Int4GemliteTensor",
]

aten = torch.ops.aten


from gemlite.bitpack import pack_weights_over_cols_torch, pack_weights_over_cols_triton
from gemlite.core import TORCH_TO_DTYPE, DType, forward_functional


class Int4GemliteTensor(TorchAOBaseTensor):
    """
    int4 quantization with GemLite kernel support

    Uses GemLite's Triton kernels with automatic kernel selection based on batch size.
    Supports groupwise quantization with configurable group sizes.

    Tensor Data Attributes:
        qdata: packed int4 weight, 2D tensor (K, N) where last dimension is packed
        scale: (K/group_size, N) for groupwise quantization, dtype is the same as the original Tensor dtype
        zero_point: (K/group_size, N) for groupwise quantization, dtype is the same as the original Tensor dtype

    Non-Tensor Data Attributes:
        block_size: the block size for quantization, representing the granularity (1, group_size)
        shape: the shape of the original Tensor (N, K)
        gemlite_metadata: list containing GemLite-specific metadata for kernel dispatch
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape", "gemlite_metadata"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        gemlite_metadata: list[int],
    ):
        kwargs = {}
        kwargs["device"] = qdata.device
        kwargs["dtype"] = scale.dtype
        kwargs["requires_grad"] = False
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        gemlite_metadata: list[int],
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.gemlite_metadata = gemlite_metadata

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}, gemlite_kernel=True"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
    ):
        """
        Create Int4GemliteTensor from high-precision weight tensor.

        Args:
            w: Weight tensor of shape (N, K)
            block_size: Quantization block size, e.g. [1, 128] for groupwise with group_size=128

        Returns:
            Int4GemliteTensor with packed weights and GemLite metadata
        """

        assert len(block_size) == w.ndim, (
            f"Expecting the length of block_size to be equal to the dimension of the weight, "
            f"got {block_size=} and {w.ndim=}"
        )

        assert all(x == 1 for x in block_size[:-1]) and block_size[-1] != 1, (
            "Only groupwise quantization is supported right now"
        )

        group_size = block_size[-1]
        original_shape = w.shape
        N, K = original_shape

        # Check divisibility requirements
        if K % 32 != 0:
            raise ValueError(
                f"GemLite requires in_features (K={K}) to be divisible by 32"
            )

        if K % group_size != 0:
            raise ValueError(
                f"in_features (K={K}) must be divisible by group_size ({group_size})"
            )

        # Minimum group size for GemLite
        if group_size < 16:
            raise ValueError("GemLite requires group_size >= 16")

        # Quantize using symmetric quantization (similar to GemLite's approach)
        # W_q shape: (N, K), scale shape: (N, K/group_size), zero_point shape: (N, K/group_size)
        W_nbits = 4
        device = w.device
        dtype = w.dtype

        # Reshape for groupwise quantization: (N, K/group_size, group_size)
        w_grouped = w.reshape(N, K // group_size, group_size)

        # Compute scales: max absolute value per group, shape (N, K/group_size)
        max_val = 7.0  # For 4-bit symmetric: range is -7 to 7
        scales = w_grouped.abs().amax(dim=-1, keepdim=False) / max_val
        scales = scales.clamp(min=1e-6)

        # Quantize to int4 range [-7, 7], then shift to uint4 range [0, 15]
        # Expand scales for broadcasting: (N, K/group_size, 1)
        scales_expanded = scales.unsqueeze(-1)
        w_quantized = (w_grouped / scales_expanded).clamp(-max_val, max_val).round()

        # Shift to unsigned range [0, 15]
        zero_point_value = 7.0
        w_uint = (w_quantized + zero_point_value).to(torch.uint8)

        # Flatten back to (N, K)
        w_uint = w_uint.reshape(N, K)

        # Pack weights using GemLite's packing utilities
        # GemLite expects weights transposed as (K, N) for packing
        _pack_weights = (
            pack_weights_over_cols_triton
            if device.type == "cuda"
            else pack_weights_over_cols_torch
        )

        # Pack over cols (transpose, pack, transpose back)
        W_packed, elements_per_sample = _pack_weights(
            w_uint.view(original_shape),
            W_nbits=W_nbits,
            packing_bitwidth=32,  # Default packing bitwidth
            transpose=True,  # Transpose to (K, N) internally
        )

        # Compute zero points (symmetric with shift)
        zero_points = torch.full_like(scales, zero_point_value, dtype=dtype)

        # Convert to GemLite's expected format
        # GemLite expects scale and zero_point to be transposed: (K/group_size, N)
        scales = scales.t().contiguous().to(dtype)
        zero_points = zero_points.t().contiguous().to(dtype)

        # Create GemLite metadata
        # Format: [scaled_activations, W_nbits, group_size, unpack_mask, elements_per_sample,
        #          input_dtype, output_dtype, acc_dtype, meta_dtype,
        #          channel_scale_mode, W_group_mode, data_contiguous]

        gemlite_dtype = TORCH_TO_DTYPE[dtype]
        gemlite_metadata = [
            0,  # scaled_activations = False (weight-only quantization)
            W_nbits,  # 4
            group_size,
            15,  # unpack_mask = 2^4 - 1
            elements_per_sample,
            gemlite_dtype.value,  # input_dtype
            gemlite_dtype.value,  # output_dtype
            DType.FP32.value,  # acc_dtype
            gemlite_dtype.value,  # meta_dtype
            0,  # channel_scale_mode
            3,  # W_group_mode = 3 (asymmetric: (Wq - zeros) * scales)
            1,  # data_contiguous
        ]

        return Int4GemliteTensor(
            qdata=W_packed,
            scale=scales,
            zero_point=zero_points,
            block_size=block_size,
            shape=original_shape,
            gemlite_metadata=gemlite_metadata,
        )


implements = Int4GemliteTensor.implements
implements_torch_function = Int4GemliteTensor.implements_torch_function


@implements([aten.linear.default])
@implements_torch_function([torch.nn.functional.linear])
def _(func, types, args, kwargs):
    """
    Implement linear operation using GemLite's forward_functional.
    """
    input_tensor, weight_tensor, bias = (
        args[0],
        args[1],
        args[2] if len(args) > 2 else None,
    )
    assert isinstance(weight_tensor, Int4GemliteTensor)

    # Ensure tensors are contiguous
    assert weight_tensor.qdata.is_contiguous(), "Expected qdata to be contiguous"
    assert weight_tensor.scale.is_contiguous(), "Expected scale to be contiguous"
    assert weight_tensor.zero_point.is_contiguous(), (
        "Expected zero_point to be contiguous"
    )

    # Prepare tensor args for GemLite: [W_q, scales, zeros]
    tensor_args = [
        weight_tensor.qdata,
        weight_tensor.scale,
        weight_tensor.zero_point,
    ]

    # Use stored metadata
    meta_args = weight_tensor.gemlite_metadata

    # Call GemLite's forward_functional with automatic kernel selection
    result = forward_functional(
        input_tensor,
        bias,
        tensor_args,
        meta_args,
        matmul_type=-1,  # -1 = automatic kernel selection
    )

    return result


# Basic operator support - these maintain the tensor structure without unpacking
@implements(aten.detach.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int4GemliteTensor(
            args[0].qdata.detach(),
            args[0].scale.detach(),
            args[0].zero_point.detach(),
            args[0].block_size,
            args[0].shape,
            args[0].gemlite_metadata,
        ),
    )


@implements(aten.clone.default)
def _(func, types, args, kwargs):
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        Int4GemliteTensor(
            args[0].qdata.clone(),
            args[0].scale.clone(),
            args[0].zero_point.clone(),
            args[0].block_size.copy(),
            args[0].shape,
            args[0].gemlite_metadata.copy(),
        ),
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    """Transpose is not supported for GemLite packed tensors."""
    raise NotImplementedError(
        "Transpose operation is not supported for Int4GemliteTensor. "
        "GemLite kernels expect a specific memory layout."
    )


Int4GemliteTensor.__module__ = "torchao.quantization"

# Allow a model with Int4GemliteTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4GemliteTensor])
