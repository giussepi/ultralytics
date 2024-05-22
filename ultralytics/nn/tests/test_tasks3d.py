# -*- coding: utf-8 -*-
"""
ultralytics/nn/tests/test_tasks3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from collections import namedtuple
from copy import deepcopy

import torch
import pytest

from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF
from ultralytics.nn.modules.head import Detect, Segment, Pose
from ultralytics.nn.tasks import DetectionModel as ODetectionModel, yaml_model_load, \
    parse_model as parse_model_2D
from ultralytics.nn.tasks3d import DetectionModel, parse_model, Modules
from ultralytics.utils.loss3d import v8DetectionLoss


# class TestODetectionModel:

#     def test_init(self):
#         # dmodel = ODetectionModel(cfg='yolov8n.yaml', ch=3, nc=None, verbose=True)
#         dmodel = ODetectionModel(cfg='ultralytics/cfg/models/v8/custom_yolov8n.yaml', ch=3, nc=None, verbose=True)

@pytest.mark.pearbioquicktests
class Testparse_model:

    def test_1(self):
        # verifying the model is constructed properly when using the 2D modules
        Modules2D = Modules(Conv, SPPF, C2f, Concat, Detect, Segment, Pose)
        ch = 1
        yaml = yaml_model_load('ultralytics/cfg/models/v8/custom_yolov8n.yaml')
        actual_model, actual_save = parse_model(deepcopy(yaml), ch, modules=Modules2D)
        expected_model, expected_save = parse_model_2D(deepcopy(yaml), ch)

        assert str(actual_model) == str(expected_model)
        assert actual_save == expected_save

    def test_2(self):
        # NOTE: This test may need to be modified if the current model built is not correct
        expected_model = """Sequential(
  (0): Conv(
    (conv): Conv3d(1, 16, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (1): Conv(
    (conv): Conv3d(16, 32, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (2): C2f(
    (cv1): Conv(
      (conv): Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(48, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (3): Conv(
    (conv): Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (4): C2f(
    (cv1): Conv(
      (conv): Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0-1): 2 x Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (5): Conv(
    (conv): Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (6): C2f(
    (cv1): Conv(
      (conv): Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0-1): 2 x Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (7): Conv(
    (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (8): C2f(
    (cv1): Conv(
      (conv): Conv3d(256, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(384, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (9): SPPF(
    (cv1): Conv(
      (conv): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(512, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): MaxPool3d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
  )
  (10): Upsample(scale_factor=2.0, mode='nearest')
  (11): Concat()
  (12): C2f(
    (cv1): Conv(
      (conv): Conv3d(384, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(192, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (13): Upsample(scale_factor=2.0, mode='nearest')
  (14): Concat()
  (15): C2f(
    (cv1): Conv(
      (conv): Conv3d(192, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(96, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (16): Conv(
    (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (17): Concat()
  (18): C2f(
    (cv1): Conv(
      (conv): Conv3d(192, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(192, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (19): Conv(
    (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=False)
    (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (act): SiLU()
  )
  (20): Concat()
  (21): C2f(
    (cv1): Conv(
      (conv): Conv3d(384, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (cv2): Conv(
      (conv): Conv3d(384, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
      (bn): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act): SiLU()
    )
    (m): ModuleList(
      (0): Bottleneck(
        (cv1): Conv(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (cv2): Conv(
          (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
      )
    )
  )
  (22): Detect(
    (cv2): ModuleList(
      (0): Sequential(
        (0): Conv(
          (conv): Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(96, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (1): Sequential(
        (0): Conv(
          (conv): Conv3d(128, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(96, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (2): Sequential(
        (0): Conv(
          (conv): Conv3d(256, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(96, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(96, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
    )
    (cv3): ModuleList(
      (0): Sequential(
        (0): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (1): Sequential(
        (0): Conv(
          (conv): Conv3d(128, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
      (2): Sequential(
        (0): Conv(
          (conv): Conv3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (1): Conv(
          (conv): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
          (bn): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (act): SiLU()
        )
        (2): Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
      )
    )
    (dfl): DFL(
      (conv): Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1), bias=False)
    )
  )
)
        """
        ch = 1
        yaml = yaml_model_load('ultralytics/cfg/models/v8/custom_yolov8n.yaml')
        actual_model, actual_save = parse_model(yaml, ch)

        assert str(actual_model).strip() == expected_model.strip()


@pytest.mark.pearbioquicktests
class TestDetectionModel:

    @pytest.fixture
    def get_obj(self):
        dmodel = DetectionModel(cfg='ultralytics/cfg/models/v8/custom_yolov8n.yaml', ch=3, nc=None, verbose=True)
        yield dmodel
        del dmodel

    def test_init(self, get_obj):
        dmodel = get_obj
        assert isinstance(dmodel.model, torch.nn.Sequential), type(dmodel)
        assert isinstance(dmodel.save, list), type(dmodel.save)
        assert dmodel.inplace is True
        assert dmodel.model[-1].inplace is True
        assert torch.equal(dmodel.model[-1].stride, torch.tensor([8., 16., 32.]))

    def test_predict_augment(self, get_obj):
        dmodel = get_obj
        with pytest.raises(NotImplementedError):
            dmodel._predict_augment(1)

    def test_descale_pred(self, get_obj):
        dmodel = get_obj
        with pytest.raises(NotImplementedError):
            dmodel._descale_pred(1, 1, 1, 1)

    def test_clip_augmented(self, get_obj):
        dmodel = get_obj
        with pytest.raises(NotImplementedError):
            dmodel._clip_augmented(1)

    def test_init_criterion(self, get_obj):
        dmodel = get_obj
        with pytest.raises(NotImplementedError):
            dmodel.init_criterion()
        # TODO: once v8DetectionLoss remove the previous lines
        #       and uncomment the following ones
        # obj = dmodel.init_criterion()
        # assert isinstance(obj, v8DetectionLoss), type(obj)
