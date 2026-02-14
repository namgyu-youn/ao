# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional

import torch
import torch.nn.functional as F

from torchao.core.config import AOBaseConfig
from torchao.quantization.transform_module import (
    _QUANTIZE_CONFIG_HANDLER,
)
from torchao.utils import DummyModule


@torch.no_grad()
def get_act_scale(x):
    return x.abs().view(-1, x.shape[-1]).mean(0)


class AWQObserver(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        base_config: AOBaseConfig,
        scale_search_space_size: int = 20,
    ):
        """
        A custom observer for Activation aware Weight Quantization (AWQ)
        Note: this only applies to weight only quantization: https://github.com/pytorch/ao/issues/2388#issuecomment-3062863647

        Args:
            weight (torch.Tensor: The weight tensor to be observed.
            bias (Optional[torch.Tensor]): The bias tensor to be observed.
            config (AOBaseConfig): the configuration for quantize_, that we'll use to apply awq on top of
            scale_search_space_size (int): search space size for searching the best scale for weight and input activation
        """
        super().__init__()
        self.base_config = base_config
        self.weight = weight
        self.bias = bias
        self.inputs = []
        self.scale_options = scale_search_space_size
        self.device = self.weight.device
        if self.bias is not None:
            self.bias.to(self.device)

    @torch.no_grad()
    def forward(self, input: torch.Tensor, output: torch.Tensor):
        self.inputs.append(input.to("cpu"))

    def _eval_scales(self, acc, x_max, ratios):
        """Evaluate quantization loss for a batch of scale ratios.

        Args:
            acc: Concatenated calibration inputs
            x_max: Per-channel activation scales
            ratios: Tensor of ratio values in [0, 1] to evaluate

        Returns:
            losses: List of MSE losses for each ratio
            scales_list: List of corresponding scale tensors
        """
        losses, scales_list = [], []
        config_handler = _QUANTIZE_CONFIG_HANDLER[type(self.base_config)]
        for r in ratios:
            s_raw = x_max.pow(r).clamp(min=1e-4)
            s = (s_raw / (s_raw.max() * s_raw.min()).sqrt()).view(-1)
            s = s.to(self.weight.dtype)
            quant_mod = config_handler(DummyModule(self.weight * s), self.base_config)
            w = quant_mod.weight
            orig_out = F.linear(acc, self.weight, self.bias)
            q_out = F.linear(acc / s, w, self.bias)
            loss = (orig_out - q_out).pow(2).mean().item()
            losses.append(loss)
            scales_list.append(s)
        return losses, scales_list

    def calculate_qparams(self):
        """Calculate optimal per-channel scales using adaptive grid
        search.

        Uses a two-phase coarse-to-fine strategy:
        1. Coarse search: Sample ~25% of budget across [0, 1] to find
           promising region
        2. Fine search: Refine with remaining budget around best coarse
           candidate

        This reduces evaluations while maintaining quality compared to
        uniform grid search.
        """
        assert self.inputs is not None, (
            "calibrate observer first by running model on exemplar data"
        )
        for i in range(len(self.inputs)):
            self.inputs[i] = self.inputs[i].to(self.device)
        if self.bias is not None:
            self.bias = self.bias.to(self.device)

        acc = torch.cat(self.inputs, dim=-2)
        x_max = get_act_scale(acc)

        # Coarse search across full range
        n_coarse = max(int(self.scale_options * 0.25), 3)
        coarse_r = torch.linspace(0, 1, n_coarse, device=self.device)
        c_loss, c_scales = self._eval_scales(acc, x_max, coarse_r)

        # Fine search around best coarse result
        best_i = c_loss.index(min(c_loss))
        if best_i == 0:
            fine_range = (0.0, coarse_r[1].item())
        elif best_i == len(coarse_r) - 1:
            fine_range = (coarse_r[-2].item(), 1.0)
        else:
            fine_range = (
                coarse_r[best_i - 1].item(),
                coarse_r[best_i + 1].item(),
            )

        fine_r = torch.linspace(
            *fine_range, self.scale_options - n_coarse, device=self.device
        )
        f_loss, f_scales = self._eval_scales(acc, x_max, fine_r)

        # Return best scale from both phases
        all_losses = c_loss + f_loss
        all_scales = c_scales + f_scales
        return all_scales[all_losses.index(min(all_losses))].detach()


class AWQObservedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        act_obs: torch.nn.Module,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.act_obs = act_obs

    def forward(self, input: torch.Tensor):
        output = F.linear(input, self.weight, self.bias)
        self.act_obs(input, output)
        return output

    @classmethod
    def from_float(cls, float_linear: torch.nn.Linear, act_obs: AWQObserver):
        observed_linear = cls(
            float_linear.in_features,
            float_linear.out_features,
            act_obs,
            False,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
        )
        observed_linear.weight = float_linear.weight
        observed_linear.bias = float_linear.bias
        return observed_linear
