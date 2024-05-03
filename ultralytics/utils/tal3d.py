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
    'make_anchors', 'dist2bbox',
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


def dist2bbox(distance: torch.Tensor, anchor_points: torch.Tensor, xyzwhd: bool = True, dim: int = -1):
    """Transform distance(ltrb) to box(xyzwhd or xyzxyz)."""
    lt, rb = distance.chunk(2, dim)  # torch.Size([1, 3, 73]), torch.Size([1, 3, 73])
    x1y1z1 = anchor_points - lt
    x2y2z2 = anchor_points + rb

    if xyzwhd:
        c_xyz = (x1y1z1 + x2y2z2) / 2
        whd = x2y2z2 - x1y1z1

        return torch.cat((c_xyz, whd), dim)  # xyzwhd bbox torch.Size([1, 6, 73])

    return torch.cat((x1y1z1, x2y2z2), dim)  # xyxy bbox
