# -*- coding: utf-8 -*-
"""
ultralytics/models/yolo/model.py

Ultralytics YOLO 🚀, AGPL-3.0 license

Copyright 2024 (C) Pear Bio Ltd
Modified by: Giussepi Lopez
"""

from ultralytics.engine.model import Model
from ultralytics.models import yolo, yolo3d  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, PoseModel, SegmentationModel
from ultralytics.nn.tasks3d import DetectionModel as DetectionModel3D


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            yolo.YOLOTasks.CLASSIFY: {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            yolo.YOLOTasks.DETECT: {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            yolo3d.YOLOTasks.DETECT3D: {
                'model': DetectionModel3D,
                'trainer': yolo3d.detect.DetectionTrainer,
                'validator': yolo3d.detect.DetectionValidator,
                'predictor': yolo3d.detect.DetectionPredictor, },
            yolo.YOLOTasks.SEGMENT: {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            yolo.YOLOTasks.POSE: {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }
