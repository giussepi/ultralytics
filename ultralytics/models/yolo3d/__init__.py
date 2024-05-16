# -*- coding: utf-8 -*-
"""
ultralytics/models/yolo3d/__init__.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

from ultralytics.models.yolo3d import detect
from .constants import YOLOTasks


__all__ = [
    'detect',
    'YOLOTasks',
]
