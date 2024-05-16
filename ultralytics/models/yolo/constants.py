# -*- coding: utf-8 -*-
"""
ultralytics/models/yolo/constants.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

__all__ = [
    'YOLOTasks',
]


class YOLOTasks:
    CLASSIFY = 'classify'
    DETECT = 'detect'
    SEGMENT = 'segment'
    POSE = 'pose'

    MEMBERS = (CLASSIFY, DETECT, SEGMENT, POSE)

    @classmethod
    def validate(cls, task: str):
        return task in cls.MEMBERS
