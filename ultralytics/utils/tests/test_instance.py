# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_instance.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from copy import deepcopy

import mock
import numpy as np
import pytest

from ultralytics.utils.instance import Bboxes


@pytest.mark.pearbioquicktests
class TestBboxes:

    @pytest.fixture
    def get_instance(self):
        bboxes = np.array(
            [[0.010937, 0.31641, 0.00625, 0.0046875], [0.60469, 0.85469, 0.00625, 0.00625]],
            dtype=np.float32
        )
        format_ = 'xywh'
        obj = Bboxes(
            bboxes=deepcopy(bboxes),
            format=format_
        )
        yield obj, bboxes, format_
        del obj, bboxes, format_

    def test_init(self, get_instance):
        obj, bboxes, format_ = get_instance
        assert np.array_equal(obj.bboxes, bboxes)
        assert obj.format == format_, f'{obj.format} != {format_}'

    def test_convert_0(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.convert(obj.format)
        assert obj.format == format_, f'{obj.format} != {format_}'
        assert np.array_equal(bboxes, obj.bboxes)

    @mock.patch('ultralytics.utils.instance.xywh2xyxy')
    def test_convert_1(self, mocked_xywh2xyxy, get_instance):
        obj, bboxes, format_ = get_instance
        new_format = 'xyxy'
        obj.convert(new_format)
        mocked_xywh2xyxy.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance.xywh2ltwh')
    def test_convert_2(self, mocked_xywh2ltwh, get_instance):
        obj, bboxes, format_ = get_instance
        new_format = 'ltwh'
        obj.convert(new_format)
        mocked_xywh2ltwh.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance.xyxy2xywh')
    def test_convert_3(self, mocked_xyxy2xywh, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyxy'
        new_format = 'xywh'
        obj.convert(new_format)
        mocked_xyxy2xywh.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance.xyxy2ltwh')
    def test_convert_4(self, mocked_xyxy2ltwh, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyxy'
        new_format = 'ltwh'
        obj.convert(new_format)
        mocked_xyxy2ltwh.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance.ltwh2xyxy')
    def test_convert_5(self, mocked_ltwh2xyxy, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'ltwh'
        new_format = 'xyxy'
        obj.convert(new_format)
        mocked_ltwh2xyxy.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance.ltwh2xywh')
    def test_convert_6(self, mocked_ltwh2xywh, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'ltwh'
        new_format = 'xywh'
        obj.convert(new_format)
        mocked_ltwh2xywh.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    def test_areas(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyxy'
        expected_areas = np.array([0.001461043, 0.507740434])
        actual_areas = obj.areas()

        assert np.allclose(actual_areas, expected_areas, 1e-5, 1e-5)

    def test_mul_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.mul(3)
        actual_output = obj.bboxes
        expected_output = bboxes * 3

        assert np.array_equal(actual_output, expected_output)

    def test_mul_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.mul([1, 2, 3, 4])
        actual_output = obj.bboxes
        expected_output = bboxes * np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

        assert np.allclose(actual_output, expected_output, 1e-5, 1e-5)

    def test_add_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.add(3)
        actual_output = obj.bboxes
        expected_output = bboxes + 3

        assert np.array_equal(actual_output, expected_output)

    def test_add_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.add([1, 2, 3, 4])
        actual_output = obj.bboxes
        expected_output = bboxes + np.array([[1, 2, 3, 4], [1, 2, 3, 4]])

        assert np.allclose(actual_output, expected_output, 1e-5, 1e-5)

    def test_len(self, get_instance):
        obj, bboxes, format_ = get_instance
        assert len(obj) == len(bboxes)

    def test_concatenate(self, get_instance):
        obj, bboxes, format_ = get_instance
        bboxes_2 = np.array(
            [[0.01233, 0.343, 0.0232, 0.023], [0.593, 0.997, 0.023, 0.0988]],
            dtype=np.float32
        )
        obj2 = Bboxes(
            bboxes=deepcopy(bboxes_2),
            format=obj.format
        )
        expected_output = np.concatenate([obj.bboxes, bboxes_2], axis=0)
        actual_output = Bboxes.concatenate([obj, obj2], axis=0).bboxes
        assert np.array_equal(actual_output, expected_output)

    def test_getitem_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        expected_output = np.array(
            [[0.010937, 0.31641, 0.00625, 0.0046875]],
            dtype=np.float32
        )
        actual_output = obj[0].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        expected_output = np.array(
            [[0.60469, 0.85469, 0.00625, 0.00625]],
            dtype=np.float32
        )
        actual_output = obj[1].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_3(self, get_instance):
        obj, bboxes, format_ = get_instance

        expected_output = np.array(
            [[0.60469, 0.85469, 0.00625, 0.00625]],
            dtype=np.float32
        )
        actual_output = obj[np.array([1])].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_4(self, get_instance):
        obj, bboxes, format_ = get_instance

        expected_output = deepcopy(bboxes)
        actual_output = obj[np.array([0, 1])].bboxes

        assert np.array_equal(actual_output, expected_output)
