# -*- coding: utf-8 -*-
# ultralytics/models/yolo/__init__.py

# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment
from .constants import YOLOTasks
from .model import YOLO


__all__ = [
    'classify',
    'segment',
    'detect',
    'pose',
    'YOLO',
    'YOLOTasks'
]
