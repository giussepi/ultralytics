# -*- coding: utf-8 -*-
"""
ultralytics/models/yolo/tests/test_constants.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import pytest

from ultralytics.models.yolo.constants import YOLOTasks


@pytest.mark.pearbioquicktests
class TestYOLOTasks:

    def test_validate_1(self):
        for task in YOLOTasks.MEMBERS:
            assert YOLOTasks.validate(task)

    def test_validate_2(self):
        assert not YOLOTasks.validate('other task')
