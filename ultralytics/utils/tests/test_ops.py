# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_ops.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""
from abc import ABC

import numpy as np
import pytest

from ultralytics.utils.ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


class MixinTest(ABC):
    @pytest.fixture
    def get_bboxes(self):
        bboxes = np.array(
            [[0.010937, 0.31641, 0.00625, 0.0046875], [0.60469, 0.85469, 0.00625, 0.00625]],
            dtype=np.float32
        )
        yield bboxes
        del bboxes


@pytest.mark.pearbioquicktests
class Testltwh2xywh(MixinTest):

    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.014062, 0.31875375, 0.00625, 0.0046875], [0.607815, 0.857815, 0.00625, 0.00625]],
            dtype=np.float32
        )
        actual_bboxes = ltwh2xywh(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testltwh2xyxy(MixinTest):

    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.010937, 0.31641, 0.017187, 0.3210975], [0.60469, 0.85469, 0.61094, 0.86094]],
            dtype=np.float32
        )
        actual_bboxes = ltwh2xyxy(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxywh2ltwh(MixinTest):

    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.007812, 0.31406625, 0.00625, 0.0046875], [0.601565, 0.851565, 0.00625, 0.00625]],
            dtype=np.float32
        )
        actual_bboxes = xywh2ltwh(bboxes)

        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxywh2xyxy(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.007812, 0.31406625, 0.014062, 0.31875375], [0.601565, 0.851565, 0.607815, 0.857815]],
            dtype=np.float32
        )
        actual_bboxes = xywh2xyxy(bboxes)

        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyxy2ltwh(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.010937, 0.31641, -0.004687, -0.3117225], [0.60469, 0.85469, -0.59844, -0.84844]],
            dtype=np.float32
        )
        actual_bboxes = xyxy2ltwh(bboxes)

        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyxy2xywh(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [[0.0085935, 0.16054875, -0.004687, -0.3117225], [0.30547, 0.43047, -0.59844, -0.84844]],
            dtype=np.float32
        )
        actual_bboxes = xyxy2xywh(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)
