# -*- coding: utf-8 -*-
"""
ultralytics/nn/modules/tests/test_block3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from abc import ABC, abstractmethod

import pytest
import torch

from ultralytics.nn.modules.block3d import SPPF, C2f, Bottleneck, DFL
from ultralytics.nn.modules.conv3d import Conv


class MixinTest(ABC):

    @pytest.fixture
    def get_input(self):
        data = torch.rand(5, 2, 16, 640, 640)
        yield data
        del data

    @abstractmethod
    def get_layer_instance(self):
        # layer = my_layer(args, **kwargs)
        # yield layer
        # del layer
        raise NotImplementedError


class TestSPPF(MixinTest):

    @pytest.fixture
    def get_layer_instance(self):
        layer = SPPF(2, 4)
        yield layer
        del layer

    def test_init(self, get_layer_instance):
        m = get_layer_instance
        assert len(m._modules) == 3
        assert isinstance(m._modules['cv1'], Conv), type(m._modules['cv1'])
        assert isinstance(m._modules['cv2'], Conv), type(m._modules['cv2'])
        assert isinstance(m._modules['m'], torch.nn.MaxPool3d), type(m._modules['m'])

    def test_forward(self, get_layer_instance, get_input):
        m = get_layer_instance
        output = m(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape


class TestBottleneck(MixinTest):

    @pytest.fixture
    def get_layer_instance(self):
        layer = Bottleneck(2, 4)
        yield layer
        del layer

    def test_init(self, get_layer_instance):
        m = get_layer_instance
        assert len(m._modules) == 2
        assert isinstance(m._modules['cv1'], Conv), type(m._modules['cv1'])
        assert isinstance(m._modules['cv2'], Conv), type(m._modules['cv2'])
        assert isinstance(m.add, bool), type(m.add)
        assert m.add is False

    def test_forward_1(self, get_layer_instance, get_input):
        # self.add = False
        m = get_layer_instance
        output = m(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape

    def test_forward_2(self, get_input):
        # self.add = True
        m = Bottleneck(2, 2)

        assert m.add is True

        output = m(get_input)
        expected_shape = [5, 2, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape


class TestC2f(MixinTest):

    @pytest.fixture
    def get_layer_instance(self):
        layer = C2f(2, 4)
        yield layer
        del layer

    def test_init(self, get_layer_instance):
        m = get_layer_instance
        assert len(m._modules) == 3
        assert isinstance(m._modules['cv1'], Conv), type(m._modules['cv1'])
        assert isinstance(m._modules['cv2'], Conv), type(m._modules['cv2'])
        assert isinstance(m._modules['m'], torch.nn.ModuleList), type(m._modules['m'])
        assert hasattr(m, 'c')
        assert isinstance(m.c, int)

    def test_forward(self, get_layer_instance, get_input):
        m = get_layer_instance
        output = m(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape

    def test_forward_split(self, get_layer_instance, get_input):
        m = get_layer_instance
        output = m.forward_split(get_input)
        expected_shape = [5, 4, 16, 640, 640]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape


class TestDFL(MixinTest):

    @pytest.fixture
    def get_layer_instance(self):
        layer = DFL(16)
        yield layer
        del layer

    def test_init_1(self, get_layer_instance):
        m = get_layer_instance
        assert len(m._modules) == 1
        assert isinstance(m._modules['conv'], torch.nn.Conv2d), type(m._modules['conv'])
        assert hasattr(m, 'conv_ch')
        assert hasattr(m, 'o_ch')
        assert m.conv_ch == 16
        assert m.o_ch == 6

    def test_forward_1(self, get_layer_instance,):
        m = get_layer_instance
        input_ = torch.rand(1, 96, 73)
        output = m(input_)
        expected_shape = [1, 6, 73]
        actual_shape = list(output.shape)

        assert actual_shape == expected_shape

    def test_forward_2(self, get_layer_instance,):
        # wrong input channels
        m = get_layer_instance
        input_ = torch.rand(1, 64, 73)
        with pytest.raises(AssertionError):
            m(input_)
