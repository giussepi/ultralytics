# -*- coding: utf-8 -*-
"""
ultralytics/nn/tasks3d.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import contextlib
from collections import namedtuple
from copy import deepcopy
from typing import Optional, List

import torch
from tabulate import tabulate
from torch import nn

from ultralytics.nn.modules.conv import Concat
from ultralytics.nn.modules.conv3d import Conv
from ultralytics.nn.modules.block3d import C2f, SPPF
from ultralytics.nn.modules.head3d import Detect, Segment, Pose
from ultralytics.nn.tasks import BaseModel as BaseModel2D, yaml_model_load
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.loss3d import v8DetectionLoss
from ultralytics.utils.torch_utils import initialize_weights, make_divisible


__all__ = [
    'Modules',
    'parse_model',
    'BaseModel3D',
    'DetectionModel',
]

# modules to be employed by the parse_model
Modules = namedtuple(
    'Modules',
    ['Conv', 'SPPF', 'C2f', 'Concat', 'Detect', 'Segment', 'Pose'],
    defaults=[Conv, SPPF, C2f, Concat, Detect, Segment, Pose]
)


def parse_model(
        d: dict, ch: int, /, *, verbose: bool = True, modules: Optional[namedtuple] = Modules()
) -> List:
    """
    Parse a YOLO model.yaml dictionary into a PyTorch model.

    Kwargs:
        d <dict>: model dict
        ch <int>: input channels
        verbose <bool>: Whether or print logging messages
        modules <namedtuple>: namedtuple contaning the modules to be used to build the NN

    Returns:
        model<nn.Sequential>, save<list>

    """
    assert isinstance(d, dict), type(d)
    assert isinstance(ch, int), type(ch)
    assert ch > 0, f'{ch} must be larger than 0'
    assert isinstance(verbose, bool), type(verbose)
    assert isinstance(modules, Modules), type(modules)

    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'activation', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))

    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        # FIXME: should it be depth, width, height? review it!!!
        depth, width, max_channels = scales[scale]

    if act:
        modules.Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    if verbose:
        tabulate_data = [['enum_idx', 'from_idx', 'depth_gain', 'num_params', 'layer', 'args']]

    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        # m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else getattr(modules, m)  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (modules.Conv, modules.SPPF, modules.C2f):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (modules.C2f, ):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is modules.Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (modules.Detect, modules.Segment, modules.Pose):
            args.append([ch[x] for x in f])
            if m in (modules.Segment, modules.Pose):
                raise NotImplementedError
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            tabulate_data.append([i, f, n_, f'{m.np:10.0f}',  f'{t:<45}', f'{str(args):<30}'])
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    if verbose:
        LOGGER.info(tabulate(tabulate_data, headers='firstrow'))

    return nn.Sequential(*layers), sorted(save)


class BaseModel3D(BaseModel2D):

    def _apply(self, fn):
        """
        Applies a function to all the tensors in the model that are not parameters or registered buffers.

        Args:
            fn (function): the function to apply to the model

        Returns:
            (BaseModel): An updated BaseModel object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self


class DetectionModel(BaseModel3D):
    """YOLOv8 3D detection model."""

    def __init__(self,
                 cfg: [str | dict] = 'yolov8n.yaml',
                 ch: int = 3,
                 nc: [int | None] = None,
                 verbose: bool = True
                 ):
        """
        Initialize the YOLOv8 detection model with the given config and parameters.

        kwargs:
            cfg      <str|dict>: yaml model to load
                            Default 'yolov8n.yaml'.
            ch       <int>: input channels.
                            Default 3
            nc       <int>: number of classes.
                            Default None
            verbose <bool>: Whether ot nor not print extra messages
                            Default True
        """
        super().__init__()

        self.yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override YAML value

        self.model, self.save = parse_model(deepcopy(self.yaml), ch, verbose=verbose)  # model, savelist
        self.names = {i: f'{i}' for i in range(self.yaml['nc'])}  # default names dict
        self.inplace = self.yaml.get('inplace', True)

        # Build strides
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment, Pose)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            def forward(x): return self.forward(x)[0] if isinstance(m, (Segment, Pose)) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s, s))])  # forward
            self.stride = m.stride
            m.bias_init()  # only run once
        else:
            self.stride = torch.Tensor([32])  # default stride for i.e. RTDETR

        # Init weights, biases
        initialize_weights(self)
        if verbose:
            self.info()
            LOGGER.info('')

    def _predict_augment(self, x):
        """Perform augmentations on input image x and return augmented inference and train outputs."""
        raise NotImplementedError

    @staticmethod
    def _descale_pred(p, flips, scale, img_size, dim=1):
        """De-scale predictions following augmented inference (inverse operation)."""
        raise NotImplementedError

    def _clip_augmented(self, y):
        """Clip YOLO augmented inference tails."""
        raise NotImplementedError

    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return v8DetectionLoss(self)
