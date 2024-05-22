# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_tal3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import pytest
import torch

from ultralytics.utils.tal3d import make_anchors, dist2bbox


@pytest.mark.pearbioquicktests
class Testmake_anchors:

    def test_1(self):
        feats = [torch.randn(1, 65, 4, 4, 4), torch.randn(1, 65, 2, 2, 2), torch.randn(1, 65, 1, 1, 1)]
        strides = torch.tensor([8., 16., 32.])
        actual_anchors, actual_strides = make_anchors(feats, strides)
        expected_anchors_shape = torch.Size([4*4*4+2*2*2+1*1*1, 3])
        expected_strides_shape = torch.Size([4*4*4+2*2*2+1*1*1, 1])
        expected_strides = torch.cat([
            torch.full((4*4*4, 1), 8.),
            torch.full((2*2*2, 1), 16.),
            torch.full((1*1*1, 1), 32.)
        ])
        expected_anchors = torch.tensor([[0.5000, 0.5000, 0.5000],
                                         [1.5000, 0.5000, 0.5000],
                                         [2.5000, 0.5000, 0.5000],
                                         [3.5000, 0.5000, 0.5000],
                                         [0.5000, 1.5000, 0.5000],
                                         [1.5000, 1.5000, 0.5000],
                                         [2.5000, 1.5000, 0.5000],
                                         [3.5000, 1.5000, 0.5000],
                                         [0.5000, 2.5000, 0.5000],
                                         [1.5000, 2.5000, 0.5000],
                                         [2.5000, 2.5000, 0.5000],
                                         [3.5000, 2.5000, 0.5000],
                                         [0.5000, 3.5000, 0.5000],
                                         [1.5000, 3.5000, 0.5000],
                                         [2.5000, 3.5000, 0.5000],
                                         [3.5000, 3.5000, 0.5000],
                                         [0.5000, 0.5000, 1.5000],
                                         [1.5000, 0.5000, 1.5000],
                                         [2.5000, 0.5000, 1.5000],
                                         [3.5000, 0.5000, 1.5000],
                                         [0.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000],
                                         [2.5000, 1.5000, 1.5000],
                                         [3.5000, 1.5000, 1.5000],
                                         [0.5000, 2.5000, 1.5000],
                                         [1.5000, 2.5000, 1.5000],
                                         [2.5000, 2.5000, 1.5000],
                                         [3.5000, 2.5000, 1.5000],
                                         [0.5000, 3.5000, 1.5000],
                                         [1.5000, 3.5000, 1.5000],
                                         [2.5000, 3.5000, 1.5000],
                                         [3.5000, 3.5000, 1.5000],
                                         [0.5000, 0.5000, 2.5000],
                                         [1.5000, 0.5000, 2.5000],
                                         [2.5000, 0.5000, 2.5000],
                                         [3.5000, 0.5000, 2.5000],
                                         [0.5000, 1.5000, 2.5000],
                                         [1.5000, 1.5000, 2.5000],
                                         [2.5000, 1.5000, 2.5000],
                                         [3.5000, 1.5000, 2.5000],
                                         [0.5000, 2.5000, 2.5000],
                                         [1.5000, 2.5000, 2.5000],
                                         [2.5000, 2.5000, 2.5000],
                                         [3.5000, 2.5000, 2.5000],
                                         [0.5000, 3.5000, 2.5000],
                                         [1.5000, 3.5000, 2.5000],
                                         [2.5000, 3.5000, 2.5000],
                                         [3.5000, 3.5000, 2.5000],
                                         [0.5000, 0.5000, 3.5000],
                                         [1.5000, 0.5000, 3.5000],
                                         [2.5000, 0.5000, 3.5000],
                                         [3.5000, 0.5000, 3.5000],
                                         [0.5000, 1.5000, 3.5000],
                                         [1.5000, 1.5000, 3.5000],
                                         [2.5000, 1.5000, 3.5000],
                                         [3.5000, 1.5000, 3.5000],
                                         [0.5000, 2.5000, 3.5000],
                                         [1.5000, 2.5000, 3.5000],
                                         [2.5000, 2.5000, 3.5000],
                                         [3.5000, 2.5000, 3.5000],
                                         [0.5000, 3.5000, 3.5000],
                                         [1.5000, 3.5000, 3.5000],
                                         [2.5000, 3.5000, 3.5000],
                                         [3.5000, 3.5000, 3.5000],
                                         [0.5000, 0.5000, 0.5000],
                                         [1.5000, 0.5000, 0.5000],
                                         [0.5000, 1.5000, 0.5000],
                                         [1.5000, 1.5000, 0.5000],
                                         [0.5000, 0.5000, 1.5000],
                                         [1.5000, 0.5000, 1.5000],
                                         [0.5000, 1.5000, 1.5000],
                                         [1.5000, 1.5000, 1.5000],
                                         [0.5000, 0.5000, 0.5000]])

        assert actual_anchors.shape == expected_anchors_shape
        assert actual_strides.shape == expected_strides_shape
        assert torch.equal(actual_strides, expected_strides)
        assert torch.equal(actual_anchors, expected_anchors)


@pytest.mark.pearbioquicktests
class Testdist2bbox:

    @pytest.fixture
    def get_kwargs(self):
        d = {
            'distance': torch.ones(1, 6, 73),
            'anchor_points': torch.randn(1, 3, 73),
            'xyzwhd': True,
            'dim': 1
        }
        yield d
        del d

    def test_1(self, get_kwargs):
        # xyzwhd = True,
        kwargs = get_kwargs
        lt, rb = kwargs['distance'].chunk(2, kwargs['dim'])
        x1y1z1 = kwargs['anchor_points'] - lt
        x2y2z2 = kwargs['anchor_points'] + rb

        c_xyz = (x1y1z1 + x2y2z2) / 2
        whd = x2y2z2 - x1y1z1
        expected_value = torch.cat((c_xyz, whd), kwargs['dim'])
        actual_value = dist2bbox(**kwargs)

        assert torch.equal(actual_value, expected_value)

    def test_2(self, get_kwargs):
        # xyzwhd = False,
        kwargs = get_kwargs
        kwargs['xyzwhd'] = False
        lt, rb = kwargs['distance'].chunk(2, kwargs['dim'])
        x1y1z1 = kwargs['anchor_points'] - lt
        x2y2z2 = kwargs['anchor_points'] + rb
        expected_value = torch.cat((x1y1z1, x2y2z2), kwargs['dim'])
        actual_value = dist2bbox(**kwargs)

        assert torch.equal(actual_value, expected_value)
