# -*- coding: utf-8 -*-
"""
ultralytics/data/tests/test_augment.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import numpy as np
import pytest

from ultralytics.data.augment import LetterBox


class TestLetterBox:

    @pytest.fixture
    def get_instance(self):
        obj = LetterBox()
        yield obj
        del obj

    @pytest.fixture
    def get_img(self):
        img = np.load(
            'ultralytics/data/tests/tiny-Pear-Tree-DB-5/train/Spots 1/Stitiching All Channels _000_Region1_ChannelViolet,SYTOXGreen,Orange,Far Red_Seq0001_C020022_day_0_[ims1_2023-11-06T17-24-17.631]/Stitiching All Channels _000_Region1_ChannelViolet,SYTOXGreen,Orange,Far Red_Seq0001_C020022_day_0_[ims1_2023-11-06T17-24-17.631]_crop_X02432_Y01920_Z00013.npy')
        yield img
        del img

    def test_init(self, get_instance):
        obj = get_instance
        assert obj.new_shape == (640, 640), f'{obj.new_shape} != (640, 640)'
        assert obj.auto is False
        assert obj.scaleFill is False
        assert obj.scaleup is True
        assert obj.center is True
        assert obj.stride == 32, f'{obj.stride} != 32'

    def test_call(self, get_img):
        new_shape = (800, 800)
        obj = LetterBox(new_shape)
        img = obj(image=get_img)

        assert isinstance(img, np.ndarray), type(img)
        assert img.shape == new_shape, f'{img.shape} != {new_shape}'
