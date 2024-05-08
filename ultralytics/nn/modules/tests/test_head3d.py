# -*- coding: utf-8 -*-
"""
ultralytics/nn/modules/tests/test_head3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez

"""

import math

import pytest
import torch

from ultralytics.nn.modules.block3d import DFL
from ultralytics.nn.modules.head3d import Detect


class TestDetect:

    @pytest.fixture
    def get_instance(self):
        obj = Detect(nc=1, ch=[64, 128, 256])
        yield obj
        del obj

    def test_init(self, get_instance):
        m = get_instance
        assert m.dynamic is False
        assert m.export is False
        assert m.shape is None
        assert torch.equal(m.anchors, torch.empty(0))
        assert torch.equal(m.strides, torch.empty(0))
        assert m.num_coords == 6
        assert m.nc == 1
        assert m.nl == 3
        assert m.reg_max == 16
        assert m.no == 97
        assert torch.equal(m.stride, torch.zeros(3))
        assert isinstance(m.cv2, torch.nn.ModuleList), type(m.cv2)
        assert isinstance(m.cv3, torch.nn.ModuleList), type(m.cv2)
        assert isinstance(m.dfl, DFL), type(m.dfl)

    def test_bias_init_1(self, get_instance):
        # m.stride full of zeros
        m = get_instance
        with pytest.raises(ValueError):
            m.bias_init()

    def test_bias_init_2(self, get_instance):
        # m.stride full of zeros
        m = get_instance
        m.stride = torch.tensor([8., 16., 32.])
        m.bias_init()

        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            actual = a[-1].bias.data
            expected = torch.ones_like(a[-1].bias.data)
            assert torch.equal(actual, expected)
            actual = b[-1].bias.data[:m.nc]
            expected = torch.full_like(b[-1].bias.data[:m.nc], math.log(5 / m.nc / (640 / s) ** 2))
            assert torch.equal(actual, expected)

    def test_forward_1(self, get_instance):
        # training = True
        m = get_instance
        input_ = [
            torch.randn(1, 64, 30, 32, 32),
            torch.randn(1, 128, 10, 16, 16),
            torch.randn(1, 256, 4, 8, 8),
        ]

        actual_output = m(input_)
        assert [i.shape for i in actual_output] == [torch.Size([1, 97, 30, 32, 32]), torch.Size(
            [1, 97, 10, 16, 16]), torch.Size([1, 97, 4, 8, 8])]

    def test_forward_2(self):
        # training = True
        m = Detect(nc=1, ch=[66, 132, 264])
        input_ = [
            torch.randn(1, 66, 32, 32, 32),
            torch.randn(1, 132, 16, 16, 16),
            torch.randn(1, 264, 8, 8, 8),
        ]

        actual_output = m(input_)
        assert [i.shape for i in actual_output] == [torch.Size([1, 97, 32, 32, 32]), torch.Size(
            [1, 97, 16, 16, 16]), torch.Size([1, 97, 8, 8, 8])]

    def test_forward_3(self, get_instance):
        # training = False
        m = get_instance
        m.training = False
        input_ = [
            torch.randn(1, 64, 32, 32, 32),
            torch.randn(1, 128, 16, 16, 16),
            torch.randn(1, 256, 8, 8, 8),
        ]

        actual_output = m(input_)
        assert actual_output[0].shape == torch.Size([1, 7, 37376])
        assert len(actual_output[1]) == 3
        assert [i.shape for i in actual_output[1]] == [torch.Size([1, 97, 32, 32, 32]), torch.Size(
            [1, 97, 16, 16, 16]), torch.Size([1, 97, 8, 8, 8])]
