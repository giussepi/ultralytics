# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_torch_utils.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from collections import OrderedDict

import pytest
from torch import nn

from ultralytics.utils.torch_utils import initialize_weights


@pytest.mark.pearbioquicktests
class Testinitialize_weights:

    @pytest.fixture
    def get_model_2D(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 20, 5)),
            ('bn1', nn.BatchNorm2d(20, eps=1e-05, momentum=0.1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(20, 64, 5)),
            ('bn2', nn.BatchNorm2d(64, eps=1e-05, momentum=0.1)),
            ('relu2', nn.ReLU(inplace=False))
        ]))
        yield model
        del model

    @pytest.fixture
    def get_model_3D(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(1, 20, 5)),
            ('bn1', nn.BatchNorm3d(20, eps=1e-05, momentum=0.1)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv3d(20, 64, 5)),
            ('bn2', nn.BatchNorm3d(64, eps=1e-05, momentum=0.1)),
            ('relu2', nn.ReLU(inplace=False))
        ]))
        yield model
        del model

    def test_2D(self, get_model_2D):
        model = get_model_2D
        conv1_str = str(model.conv1)
        conv2_str = str(model.conv2)

        initialize_weights(model)

        assert str(model.conv1).strip() == conv1_str.strip()
        assert str(model.conv2).strip() == conv2_str.strip()
        assert model.bn1.eps == 1e-3
        assert model.bn1.momentum == .03
        assert model.bn2.eps == 1e-3
        assert model.bn2.momentum == .03
        assert model.relu1.inplace is True
        assert model.relu2.inplace is True

    def test_3D(self, get_model_3D):
        model = get_model_3D
        conv1_str = str(model.conv1)
        conv2_str = str(model.conv2)

        initialize_weights(model)

        assert str(model.conv1).strip() == conv1_str.strip()
        assert str(model.conv2).strip() == conv2_str.strip()
        assert model.bn1.eps == 1e-3
        assert model.bn1.momentum == .03
        assert model.bn2.eps == 1e-3
        assert model.bn2.momentum == .03
        assert model.relu1.inplace is True
        assert model.relu2.inplace is True
