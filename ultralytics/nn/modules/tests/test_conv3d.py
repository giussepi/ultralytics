# -*- coding: utf-8 -*-
"""
ultralytics/nn/modules/tests/test_conv3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import pytest
import torch

from ultralytics.nn.modules.conv3d import Conv


class TestConv:

    @pytest.fixture
    def get_input(self):
        data = torch.rand(5, 2, 16, 640, 640)
        yield data
        del data

    def test_init(self):
        m = Conv(2, 4)
        assert len(m._modules) == 3
        assert isinstance(m._modules['conv'], torch.nn.Conv3d), type(m._modules['conv'])
        assert isinstance(m._modules['bn'], torch.nn.BatchNorm3d), type(m._modules['bn'])
        assert isinstance(m._modules['act'], torch.nn.SiLU), type(m._modules['act'])

    def test_forward(self, get_input):
        m = Conv(2, 4)
        output = m(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape

    def test_forward_fuse(self, get_input):
        m = Conv(2, 4)
        output = m.forward_fuse(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape
