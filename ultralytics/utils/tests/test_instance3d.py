# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_instance3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from copy import deepcopy

import mock
import numpy as np
import pytest

from ultralytics.utils.instance3d import Bboxes3D


@pytest.mark.pearbioquicktests
class TestBboxes:

    @pytest.fixture
    def get_instance(self):
        bboxes = np.array(
            [
                [0.010937, 0.31641, 0.1023, 0.00625, 0.0046875, 0.00432],
                [0.60469, 0.85469, 0.23403, 0.00625, 0.00625, 0.00555]
            ],
            dtype=np.float32
        )
        format_ = 'xyzwhd'
        obj = Bboxes3D(
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

    @mock.patch('ultralytics.utils.instance3d.xyzwhd2xyzxyz')
    def test_convert_1(self, mocked_xyzwhd2xyzxyz, get_instance):
        obj, bboxes, format_ = get_instance
        new_format = 'xyzxyz'
        obj.convert(new_format)
        mocked_xyzwhd2xyzxyz.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance3d.xyzwhd2ltwhd')
    def test_convert_2(self, mocked_xyzwhd2ltwhd, get_instance):
        obj, bboxes, format_ = get_instance
        new_format = 'ltwhd'
        obj.convert(new_format)
        mocked_xyzwhd2ltwhd.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance3d.xyzxyz2xyzwhd')
    def test_convert_3(self, mocked_xyzxyz2xyzwhd, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyzxyz'
        new_format = 'xyzwhd'
        obj.convert(new_format)
        mocked_xyzxyz2xyzwhd.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance3d.xyzxyz2ltwhd')
    def test_convert_4(self, mocked_xyzxyz2ltwhd, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyzxyz'
        new_format = 'ltwhd'
        obj.convert(new_format)
        mocked_xyzxyz2ltwhd.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance3d.ltwhd2xyzxyz')
    def test_convert_5(self, mocked_ltwhd2xyzxyz, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'ltwhd'
        new_format = 'xyzxyz'
        obj.convert(new_format)
        mocked_ltwhd2xyzxyz.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    @mock.patch('ultralytics.utils.instance3d.ltwhd2xyzwhd')
    def test_convert_6(self, mocked_ltwhd2xyzwhd, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'ltwhd'
        new_format = 'xyzwhd'
        obj.convert(new_format)
        mocked_ltwhd2xyzwhd.assert_called_once()
        assert obj.format == new_format, f'{obj.format} != {new_format}'

    def test_areas(self, get_instance):
        obj, bboxes, format_ = get_instance
        with pytest.raises(NotImplementedError):
            obj.areas()

    def test_volumes(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.format = 'xyzxyz'

        # [0.010937, 0.31641, 0.1023, 0.00625, 0.0046875, 0.00432],
        # [0.60469, 0.85469, 0.23403, 0.00625, 0.00625, 0.00555]

        expected_areas = np.array([-0.000143153, -0.116008534])
        actual_areas = obj.volumes()

        assert np.allclose(actual_areas, expected_areas, 1e-5, 1e-5)

    def test_mul_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.mul(3)
        actual_output = obj.bboxes
        expected_output = bboxes * 3

        assert np.array_equal(actual_output, expected_output)

    def test_mul_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.mul([1, 2, 3, 4, 5, 6])
        actual_output = obj.bboxes
        expected_output = bboxes * np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

        assert np.allclose(actual_output, expected_output, 1e-5, 1e-5)

    def test_add_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.add(3)
        actual_output = obj.bboxes
        expected_output = bboxes + 3

        assert np.array_equal(actual_output, expected_output)

    def test_add_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        obj.add([1, 2, 3, 4, 5, 6])
        actual_output = obj.bboxes
        expected_output = bboxes + np.array([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]])

        assert np.allclose(actual_output, expected_output, 1e-5, 1e-5)

    def test_len(self, get_instance):
        obj, bboxes, format_ = get_instance
        assert len(obj) == len(bboxes)

    def test_concatenate(self, get_instance):
        obj, bboxes, format_ = get_instance
        bboxes_2 = np.array(
            [[0.01233, 0.343, 0.0232, 0.023, 0.123, 0.0023], [0.593, 0.997, 0.023, 0.0988, 0.232, 0.8778]],
            dtype=np.float32
        )
        obj2 = Bboxes3D(
            bboxes=deepcopy(bboxes_2),
            format=obj.format
        )
        expected_output = np.concatenate([obj.bboxes, bboxes_2], axis=0)
        actual_output = Bboxes3D.concatenate([obj, obj2], axis=0).bboxes
        np.array_equal(actual_output, expected_output)

    def test_getitem_1(self, get_instance):
        obj, bboxes, format_ = get_instance
        expected_output = np.array(
            [[0.010937, 0.31641, 0.1023, 0.00625, 0.0046875, 0.00432]],
            dtype=np.float32
        )
        actual_output = obj[0].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_2(self, get_instance):
        obj, bboxes, format_ = get_instance
        expected_output = np.array(
            [[0.60469, 0.85469, 0.23403, 0.00625, 0.00625, 0.00555]],
            dtype=np.float32
        )
        actual_output = obj[1].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_3(self, get_instance):
        obj, bboxes, format_ = get_instance

        expected_output = np.array(
            [[0.60469, 0.85469, 0.23403, 0.00625, 0.00625, 0.00555]],
            dtype=np.float32
        )
        actual_output = obj[np.array([1])].bboxes

        assert np.array_equal(actual_output, expected_output)

    def test_getitem_4(self, get_instance):
        obj, bboxes, format_ = get_instance

        expected_output = deepcopy(bboxes)
        actual_output = obj[np.array([0, 1])].bboxes

        assert np.array_equal(actual_output, expected_output)
