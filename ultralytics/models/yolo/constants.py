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
    DETECT3D = 'detect3d'
    SEGMENT = 'segment'
    POSE = 'pose'

    MEMBERS = (CLASSIFY, DETECT, DETECT3D, SEGMENT, POSE)

    @classmethod
    def validate(cls, task: str):
        return task in cls.MEMBERS
