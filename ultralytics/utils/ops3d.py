# -*- coding: utf-8 -*-
"""
ultralytics/utils/ops3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import numpy as np
import torch


__all__ = [
    'ltdwhd2xyzwhd',
    'ltdwhd2xyzxyz',
    'xyzwhd2ltdwhd',
    'xyzwhd2xyzxyz',
    'xyzxyz2ltdwhd',
    'xyzxyz2xyzwhd'
]


def ltdwhd2xyzwhd(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    Convert nx6 boxes from [x1, y1, z1, w, h, d] to [x, y, z w, h, d] where xyz1=top-left, xyz=center.
    Note: 3D version of ltwh2xywh

    Args:
        x (torch.Tensor): the input tensor

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the xywh format.
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'{x.shape[-1]} != 6'

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] + x[..., 3] / 2  # center x
    y[..., 1] = x[..., 1] + x[..., 4] / 2  # center y
    y[..., 2] = x[..., 2] + x[..., 5] / 2  # center z

    return y


def ltdwhd2xyzxyz(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    It converts the bounding box from [x1, y1,z1, w, h, d] to [x1, y1, z1, x2, y2, z2]
    where xyz1=top-left, xyz2=bottom-right.
    Note: 3D version of ltwh2xyxy

    Args:
        x (np.ndarray | torch.Tensor): the input image

    Returns:
        y (np.ndarray | torch.Tensor): the xyxy coordinates of the bounding boxes.
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'{x.shape[-1]} != 6'

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 3] = x[..., 3] + x[..., 0]  # width
    y[..., 4] = x[..., 4] + x[..., 1]  # height
    y[..., 5] = x[..., 5] + x[..., 2]  # depth

    return y


def xyzwhd2ltdwhd(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    Convert the bounding box format from [x, y, z, w, h, d] to [x1, y1, z1, w, h, d],
    where x1, y1, z1 are the top-left coordinates.
    Note: 3D version of xywh2ltwh

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding box coordinates in the xyzwhd format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the ltdwhd format
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'{x.shape[-1]} != 6'

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 3] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 4] / 2  # top left y
    y[..., 2] = x[..., 2] - x[..., 5] / 2  # top left z

    return y


def xyzwhd2xyzxyz(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    Convert bounding box coordinates from (x, y, z, width, height, depth) format to
    (x1, y1, z1, x2, y2, z2) format where (x,y,z) is the centre, (x1, y1, z1) is the
    top-left corner and (x2, y2, z2) is the bottom-right corner.
    Note: 3D version of xywh2xyxy

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in
                                       (x, y, z, width, height, depth) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, z1, x2, y2, z2) format.
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'input shape last dimension expected 6 but input shape is {x.shape}'

    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    dw = x[..., 3] / 2  # half-width
    dh = x[..., 4] / 2  # half-height
    dd = x[..., 5] / 2  # half-depth
    y[..., 0] = x[..., 0] - dw  # top left x
    y[..., 1] = x[..., 1] - dh  # top left y
    y[..., 2] = x[..., 2] - dd  # top left z
    y[..., 3] = x[..., 0] + dw  # bottom right x
    y[..., 4] = x[..., 1] + dh  # bottom right y
    y[..., 5] = x[..., 2] + dd  # bottom right z

    return y


def xyzxyz2ltdwhd(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    Convert nx6 bounding boxes from [x1, y1, z1, x2, y2, z2] to [x1, y1, z1,  w, h, d],
    where xyz1=top-left, xyz2=bottom-right.
    Note: 3D version of xyxy2ltwh

    Args:
        x (np.ndarray | torch.Tensor): The input tensor with the bounding boxes coordinates
                                       in the xyzxyz format

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in the ltdwhd format.
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'input shape last dimension expected 6 but input shape is {x.shape}'

    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 3] = x[..., 3] - x[..., 0]  # width
    y[..., 4] = x[..., 4] - x[..., 1]  # height
    y[..., 5] = x[..., 5] - x[..., 2]  # height

    return y


def xyzxyz2xyzwhd(x: [torch.Tensor | np.ndarray]) -> torch.Tensor | np.ndarray:
    """
    Convert bounding box coordinates from (x1, y1, z1, x2, y2, z2) format to
    (x, y, z, width, height, depth) format where (x1, y1, z1) is the
    top-left corner, (x2, y2, z2) is the bottom-right corner and (x, y, z) is the centre
    Note: 3D version of xyxy2xywh

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    assert isinstance(x, (torch.Tensor, np.ndarray)), type(x)
    assert x.shape[-1] == 6, f'input shape last dimension expected 6 but input shape is {x.shape}'

    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    y[..., 0] = (x[..., 0] + x[..., 3]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 4]) / 2  # y center
    y[..., 2] = (x[..., 2] + x[..., 5]) / 2  # z center
    y[..., 3] = x[..., 3] - x[..., 0]  # width
    y[..., 4] = x[..., 4] - x[..., 1]  # height
    y[..., 5] = x[..., 5] - x[..., 2]  # depth

    return y
