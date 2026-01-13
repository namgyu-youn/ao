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


import gemlite
from gemlite.core import forward_functional


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
        gemlite_kwargs: dict containing GemLite-specific metadata for kernel dispatch
            - meta_args: complete GemLite kernel metadata list (required for forward pass)
            - in_features: input dimension (K)
            - out_features: output dimension (N)
            - packing_bitwidth: bit packing width (8, 16, or 32)
    """

    tensor_data_names = ["qdata", "scale", "zero_point"]
    tensor_attribute_names = ["block_size", "shape", "gemlite_kwargs"]

    def __new__(
        cls,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        block_size: list[int],
        shape: torch.Size,
        gemlite_kwargs: dict,
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
        gemlite_kwargs: dict,
    ):
        super().__init__()
        self.qdata = qdata
        self.scale = scale
        self.zero_point = zero_point
        self.block_size = block_size
        self.gemlite_kwargs = gemlite_kwargs

    def _quantization_type(self):
        return f"shape={self.shape}, block_size={self.block_size}, device={self.device}, gemlite_kernel=True"

    @classmethod
    def from_hp(
        cls,
        w: torch.Tensor,
        block_size: list[int],
        packing_bitwidth: int = 32,
    ):
        """
        Create Int4GemliteTensor from high-precision weight tensor using GemLite helper.

        Args:
            w: Weight tensor of shape (N, K)
            block_size: Quantization block size, e.g. [1, 128] for groupwise with group_size=128
            packing_bitwidth: Bit packing width (8, 16, or 32), default 32

        Returns:
            Int4GemliteTensor with packed weights and GemLite metadata
        """

        # Extract dimensions from input tensor
        N, K = w.shape
        group_size = block_size[1]

        # Validate GemLite constraints
        if group_size < 16:
            raise ValueError(f"GemLite requires group_size >= 16, got {group_size}")
        if K % 32 != 0:
            raise ValueError(
                f"GemLite requires in_features (K={K}) to be divisible by 32"
            )
        if K % group_size != 0:
            raise ValueError(
                f"in_features (K={K}) must be divisible by group_size ({group_size})"
            )

        # Reshape for groupwise: (N, K/group_size, group_size)
        w_grouped = w.reshape(N, K // group_size, group_size)

        # Compute scales: max absolute value per group
        max_val = 7.0  # For 4-bit symmetric: range is -7 to 7
        scales = w_grouped.abs().amax(dim=-1, keepdim=False) / max_val
        scales = scales.clamp(min=1e-6)

        # Quantize to int4 range [-7, 7], then shift to uint4 range [0, 15]
        scales_expanded = scales.unsqueeze(-1)
        w_quantized = (w_grouped / scales_expanded).clamp(-max_val, max_val).round()

        # Shift to unsigned range [0, 15]
        zero_point_value = 7.0
        int_data = (w_quantized + zero_point_value).to(torch.uint8).reshape(N, K)

        # Compute zero points (all same value for symmetric quantization)
        zero_point = torch.full_like(scales, zero_point_value, dtype=w.dtype)

        # Use GemLite helper to pack weights and generate metadata
        processor = gemlite.helper.A16Wn(
            device=w.device, packing_bitwidth=packing_bitwidth
        )

        # GemLite helper handles packing and metadata generation
        # Note: Parameters are positional: int_data, scale, zero_point, bit_width, group_size, bias
        gemlite_linear = processor.from_weights(
            int_data, scales, zero_point, 4, group_size, bias=None
        )

        # Extract metadata and packed tensors from GemLite
        meta_args = gemlite_linear.get_meta_args()
        packed_weight, scale, zero_point = gemlite_linear.get_tensor_args()

        # Create minimal metadata dict
        gemlite_kwargs = {
            "meta_args": meta_args,
            "in_features": K,
            "out_features": N,
            "packing_bitwidth": packing_bitwidth,
        }

        return cls(
            qdata=packed_weight,
            scale=scale,
            zero_point=zero_point,
            block_size=block_size,
            shape=w.shape,
            gemlite_kwargs=gemlite_kwargs,
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

    # Use stored metadata from dict
    meta_args = weight_tensor.gemlite_kwargs["meta_args"]

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
            args[0].gemlite_kwargs,
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
            args[0].gemlite_kwargs.copy(),
        ),
    )


@implements(aten.t.default)
def _(func, types, args, kwargs):
    """
    Transpose is not applied to the packed data because GemLite kernels
    expect weights in a specific layout. The functional linear op may
    decompose into transpose + addmm, but we use GemLite's matmul kernel
    which expects the weight as-is.
    """
    return return_and_correct_aliasing(
        func,
        args,
        kwargs,
        args[0],  # Return self without modification
    )


Int4GemliteTensor.__module__ = "torchao.quantization"

# Allow a model with Int4GemliteTensor weights to be loaded with `weights_only=True`
torch.serialization.add_safe_globals([Int4GemliteTensor])
