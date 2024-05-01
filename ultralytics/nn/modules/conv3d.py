# -*- coding: utf-8 -*-
"""
ultralytics/nn/modules/conv3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import torch
from torch import nn

from ultralytics.nn.modules.conv import autopad


__all__ = ('Conv',)


class Conv(nn.Module):
    """
    Standard 3D convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation).
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given arguments including activation.

        Kwargs:
            c1 <int>: ch_in
            c2 <int>: ch_out
            k  <int>: kernel size
            s  <int>: stride
            p  <int | None>: padding
            g  <int>: groups
            d  <int>: dilation
            act <bool>: activation
        """
        super().__init__()
        self.conv = nn.Conv3d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm3d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply convolution, batch normalization and activation to input tensor.

        x <torch.Tensor>: input with shape <batch, channels, depth, height, width>
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform transposed convolution of 2D data.

        x <torch.Tensor>: input with shape <batch, channels, depth, height, width>
        """
        return self.act(self.conv(x))
