# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_ops3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""
from abc import ABC

import numpy as np
import pytest

from ultralytics.utils.ops3d import ltdwhd2xyzwhd, ltdwhd2xyzxyz, xyzwhd2ltdwhd, xyzwhd2xyzxyz, \
    xyzxyz2ltdwhd, xyzxyz2xyzwhd


class MixinTest(ABC):
    @pytest.fixture
    def get_bboxes(self):
        bboxes = np.array(
            [
                [0.010937, 0.31641, 0.1023, 0.00625, 0.0046875, 0.00432],
                [0.60469, 0.85469, 0.23403, 0.00625, 0.00625, 0.00555]
            ],
            dtype=np.float32
        )
        yield bboxes
        del bboxes


@pytest.mark.pearbioquicktests
class Testltdwhd2xyzwhd(MixinTest):

    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.014062, 0.31875375, 0.10446, 0.00625, 0.0046875, 0.00432],
                [0.607815, 0.857815, 0.236805, 0.00625, 0.00625, 0.00555]
            ],
            dtype=np.float32
        )
        actual_bboxes = ltdwhd2xyzwhd(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testltdwhd2xyzxyz(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.010937, 0.31641, 0.1023, 0.017187, 0.3210975, 0.10662],
                [0.60469, 0.85469, 0.23403, 0.61094, 0.86094, 0.23958]
            ],
            dtype=np.float32
        )
        actual_bboxes = ltdwhd2xyzxyz(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyzwhd2ltdwhd(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.007812, 0.31406625, 0.10014, 0.00625, 0.0046875, 0.00432],
                [0.601565, 0.851565, 0.231255, 0.00625, 0.00625, 0.00555]
            ],
            dtype=np.float32
        )
        actual_bboxes = xyzwhd2ltdwhd(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyzwhd2xyzxyz(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.007812, 0.31406625, 0.10014, 0.014062, 0.31875375, 0.10446],
                [0.601565, 0.851565, 0.231255, 0.607815, 0.857815, 0.236805]
            ],
            dtype=np.float32
        )
        actual_bboxes = xyzwhd2xyzxyz(bboxes)

        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyzxyz2ltdwhd(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.010937, 0.31641, 0.1023, -0.004687, -0.3117225, -0.09798],
                [0.60469, 0.85469, 0.23403, -0.59844, -0.84844, -0.22848]
            ],
            dtype=np.float32
        )
        actual_bboxes = xyzxyz2ltdwhd(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)


@pytest.mark.pearbioquicktests
class Testxyzxyz2xyzwhd(MixinTest):
    def test_1(self, get_bboxes):
        bboxes = get_bboxes
        expected_bboxes = np.array(
            [
                [0.0085935, 0.16054875, 0.05331, -0.004687, -0.3117225, -0.09798],
                [0.30547, 0.43047, 0.11979, -0.59844, -0.84844, -0.22848]
            ],
            dtype=np.float32
        )
        actual_bboxes = xyzxyz2xyzwhd(bboxes)
        assert np.allclose(actual_bboxes, expected_bboxes, 1e-5, 1e-5)
