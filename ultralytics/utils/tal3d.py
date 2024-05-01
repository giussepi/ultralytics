# -*- coding: utf-8 -*-
"""
ultralytics/utils/tal3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from typing import List, Tuple

import torch

from ultralytics.utils.checks import check_version


TORCH_1_10 = check_version(torch.__version__, '1.10.0')


__all__ = [
    'make_anchors',
]


def make_anchors(
        feats: List[torch.Tensor], strides: torch.Tensor, grid_cell_offset: float = 0.5
) -> Tuple[torch.Tensor]:
    """
    Generate anchors from features <b, c, h, w, d>.

    Kwargs:
        feats <List[torch.Tensor]>: list of 5D tensors with shape <b, c, h, w, d>
        strides     <torch.Tensor>:
        grid_cell_offset   <float>: Default 0.5
    """
    # TODO: I may need to handle a different stride for the depth
    assert isinstance(feats, list), type(feats)
    assert len(feats) > 0
    assert len(feats) == len(strides)
    assert isinstance(grid_cell_offset, float), type(grid_cell_offset)

    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device

    for i, stride in enumerate(strides):
        _, _, h, w, d = feats[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sz = torch.arange(end=d, device=device, dtype=dtype) + grid_cell_offset  # shift z
        sz, sy, sx = torch.meshgrid(sz, sy, sx, indexing='ij') if TORCH_1_10 else torch.meshgrid(sz, sy, sx)
        anchor_points.append(torch.stack((sx, sy, sz), -1).view(-1, 3))
        stride_tensor.append(torch.full((h * w * d, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)
