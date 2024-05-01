# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_tal3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import torch

from ultralytics.utils.tal3d import make_anchors


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
