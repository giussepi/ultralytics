# -*- coding: utf-8 -*-
"""
ultralytics/utils/loss3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""


__all__ = [
    'v8DetectionLoss',
]


class v8DetectionLoss:
    """Criterion class for computing training losses."""
    # TODO: implement it!
    # TODO: ultralytics/nn/tests/test_tasks3d.py -> TestDetectionModel.test_init_criterion needs
    #       to be updated after implementing this function

    def __init__(self, model):  # model must be de-paralleled
        """
        Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function.
        """
        raise NotImplementedError

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        raise NotImplementedError

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        raise NotImplementedError

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        raise NotImplementedError
