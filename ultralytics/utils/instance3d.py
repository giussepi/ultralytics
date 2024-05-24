# -*- coding: utf-8 -*-
"""
instance3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import numpy as np
from numbers import Number
from typing import List

from .instance import _ntuple
from .ops3d import ltwhd2xyzwhd, ltwhd2xyzxyz, xyzwhd2ltwhd, xyzwhd2xyzxyz, xyzxyz2ltwhd, xyzxyz2xyzwhd


to_6tuple = _ntuple(6)

# `xyzxyz` means left top ini_depth and right bottom end_depth
# `xyzwhd` means center x, center y, center z and width, height, depth (YOLO format)
_formats = ['xyzxyz', 'xyzwhd', 'ltwhd']


__all__ = [
    'Bboxes3D',
]


class Bboxes3D:
    """
    A class for handling 3D bounding boxes.

    The class supports various bounding box formats like 'xyzxyz', 'xyzwhd', 'ltwhd'
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyzxyz', 'xyzwhd' or 'ltwhd').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes: np.ndarray, format: str = 'xyzxyz') -> None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        assert isinstance(bboxes, np.ndarray), type(bboxes)

        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2, bboxes.ndim
        assert bboxes.shape[1] == 6, bboxes.shape[1]
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format: str):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f'Invalid bounding box format: {format}, format must be one of {_formats}'
        if self.format == format:
            return

        if self.format == 'xyzxyz':
            func = xyzxyz2xyzwhd if format == 'xyzwhd' else xyzxyz2ltwhd
        elif self.format == 'xyzwhd':
            func = xyzwhd2xyzxyz if format == 'xyzxyz' else xyzwhd2ltwhd
        else:
            func = ltwhd2xyzxyz if format == 'xyzxyz' else ltwhd2xyzwhd
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        raise NotImplementedError('This is a 3D bbox, call volumnes() instead of areas()')

    def volumes(self):
        """Return box volumes."""
        self.convert('xyzxyz')
        return (self.bboxes[:, 3] - self.bboxes[:, 0]) * (self.bboxes[:, 4] - self.bboxes[:, 1]) \
            * (self.bboxes[:, 5] - self.bboxes[:, 2])

    # # def denormalize(self, w, h):
    # #    if not self.normalized:
    # #         return
    # #     assert (self.bboxes <= 1.0).all()
    # #     self.bboxes[:, 0::2] *= w
    # #     self.bboxes[:, 1::2] *= h
    # #     self.normalized = False
    # #
    # # def normalize(self, w, h):
    # #     if self.normalized:
    # #         return
    # #     assert (self.bboxes > 1.0).any()
    # #     self.bboxes[:, 0::2] /= w
    # #     self.bboxes[:, 1::2] /= h
    # #     self.normalized = True

    def mul(self, scale: tuple | list | int):
        """
        Args:
            scale (tuple | list | int): the scale for four coords.
        """
        if isinstance(scale, Number):
            scale = to_6tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 6
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]
        self.bboxes[:, 4] *= scale[4]
        self.bboxes[:, 5] *= scale[5]

    def add(self, offset: tuple | list | int):
        """
        Args:
            offset (tuple | list | int): the offset for four coords.
        """
        if isinstance(offset, Number):
            offset = to_6tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 6
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]
        self.bboxes[:, 4] += offset[4]
        self.bboxes[:, 5] += offset[5]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List['Bboxes3D'], axis=0) -> 'Bboxes':
        """
        Concatenate a list of Bboxes3D objects into a single Bboxes3D object.

        Args:
            boxes_list (List[Bboxes3D]): A list of Bboxes3D objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.

        Returns:
            Bboxes3D: A new Bboxes3D object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes3D objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes3D) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index: int | np.ndarray) -> 'Bboxes':
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes3D object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            # return Bboxes3D(self.bboxes[index].view(1, -1))
            # view does not work as intended on numpy, using reshape instead
            return self.__class__(self.bboxes[index].reshape(1, -1))

        b = self.bboxes[index]
        assert b.ndim == 2, f'Indexing on Bboxes3D with {index} failed to return a matrix!'

        return self.__class__(b)
