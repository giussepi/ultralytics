# -*- coding: utf-8 -*-
"""
ultralytics/models/yolo3d/constants.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

__all__ = [
    'YOLOTasks',
]


class YOLOTasks:
    DETECT3D = 'detect3d'

    MEMBERS = (DETECT3D, )

    @classmethod
    def validate(cls, task: str):
        return task in cls.MEMBERS
