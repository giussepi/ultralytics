# -*- coding: utf-8 -*-
"""
ultralytics/nn/modules/head3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import math

import torch
from torch import nn

from ultralytics.nn.modules.block3d import DFL
from ultralytics.nn.modules.conv3d import Conv
from ultralytics.utils.tal3d import dist2bbox, make_anchors


__all__ = [
    'Detect',
]


class Detect(nn.Module):
    """
    YOLOv8 3D Detect head for detection models.
    """
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    num_coords = 6  # num coord to be produced like x1y1z1 x2y2z2

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * self.num_coords  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2 = max((16, ch[0] // self.num_coords, self.reg_max * self.num_coords))  # channels
        c3 = max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv3d(c2, self.reg_max * self.num_coords, 1))
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv3d(c3, self.nc, 1))
            for x in ch
        )
        self.dfl = DFL(self.reg_max, o_ch=self.num_coords) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCDHW

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        if self.training:
            return x

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * self.num_coords]
            cls = x_cat[:, self.reg_max * self.num_coords:]
        else:
            box, cls = x_cat.split((self.reg_max * self.num_coords, self.nc), 1)

        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xyzwhd=True, dim=1) * self.strides

        if self.export and self.format in ('tflite', 'edgetpu'):
            # Normalize xyzwhd with image size to mitigate quantization error of TFLite integer models as done in YOLOv5:
            # https://github.com/ultralytics/yolov5/blob/0c8de3fca4a702f8ff5c435e67f378d1fce70243/models/tf.py#L307-L309
            # See this PR for details: https://github.com/ultralytics/ultralytics/pull/1695
            img_d = shape[2] * self.stride[0]
            img_h = shape[3] * self.stride[0]
            img_w = shape[4] * self.stride[0]
            img_size = torch.tensor([img_w, img_h, img_d, img_w, img_h, img_d], device=dbox.device)\
                            .reshape(1, self.num_coords, 1)
            dbox /= img_size

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
