# -*- coding: utf-8 -*-
"""
ultralytics/data/tests/test_dataset.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from ultralytics.data.augment import Compose, Instances
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import IterableSimpleNamespace


TEST_2D_DB_PATH = 'ultralytics/data/tests/tiny-Pear-Tree-DB-5'
TEST_2D_DB_YAML = 'ultralytics/data/tests/peartree.yaml'
YOLO_2D_CFG = 'ultralytics/data/tests/custom_config.yaml'
YOLO_3D_CFG = 'ultralytics/data/tests/custom_config_3d.yaml'
LABEL_2D_PATH = 'ultralytics/data/tests/label.pkl'
BATCH_2D_PATH = 'ultralytics/data/tests/batch2d.pkl'


@pytest.mark.pearbioquicktests
class TestYOLODataset:

    @pytest.fixture
    def get_initial_config(self):
        data = {'path': Path(TEST_2D_DB_PATH),
                'train': os.path.join(TEST_2D_DB_PATH, 'train'),
                'val':  os.path.join(TEST_2D_DB_PATH, 'val'),
                'test':  os.path.join(TEST_2D_DB_PATH, 'test'),
                'names': {0: 'colourdetection'},
                'yaml_file': TEST_2D_DB_YAML,
                'nc': 1
                }
        use_segments = False
        use_keypoints = False
        args = ()
        kwargs = {
            'img_path': os.path.join(TEST_2D_DB_PATH, 'train'),
            'imgsz': 640,
            'batch_size': 8,
            'augment': True,
            'hyp': IterableSimpleNamespace(
                task='detect',
                mode='train',
                model=None,
                data=TEST_2D_DB_YAML,
                epochs=1,
                patience=50,
                batch=8,
                imgsz=640,
                save=True,
                save_period=-1,
                cache=False,
                device=0,
                workers=4,
                project='pear-tree-detection',
                name='exp32',
                exist_ok=False,
                pretrained=True,
                optimizer='Adam',
                verbose=True,
                seed=20,
                deterministic=True,
                single_cls=False,
                rect=False,
                cos_lr=False,
                close_mosaic=10,
                resume=False,
                amp=False,
                fraction=1.0,
                profile=False,
                freeze='None',
                overlap_mask=True,
                mask_ratio=4,
                dropout=0.0,
                val=True,
                split='val',
                save_json=False,
                save_hybrid=False,
                conf=0.001,
                conf_pred=0.001,
                iou=0.7,
                max_det=10000,
                half=False,
                dnn=False,
                plots=True,
                source=None,
                vid_stride=1,
                stream_buffer=False,
                visualize=False,
                augment=False,
                agnostic_nms=False,
                classes=None,
                retina_masks=False,
                show=False,
                save_frames=False,
                save_txt=False,
                save_conf=False,
                save_crop=False,
                show_labels=True,
                show_conf=True,
                show_boxes=True,
                line_width=None,
                format='torchscript',
                keras=False,
                optimize=False,
                int8=False,
                dynamic=False,
                simplify=False,
                opset=None,
                workspace=4,
                nms=False,
                lr0=0.001,
                lrf=0.001,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=3.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                pose=12.0,
                kobj=1.0,
                label_smoothing=0.0,
                nbs=64,
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=0.0,
                translate=0.1,
                scale=0.5,
                shear=0.0,
                perspective=0.0,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                mixup=0.0,
                copy_paste=0.0,
                cfg=YOLO_2D_CFG,
                tracker='botsort.yaml',
                save_dir=os.path.join('pear-tree-detection', 'exp32'),
            ),
            'rect': False,
            'cache': None,
            'single_cls': False,
            'stride': 32,
            'pad': 0.0,
            'prefix': '\x1b[34m\x1b[1mtrain: \x1b[0m',
            'classes': None,
            'fraction': 1.0
        }
        yield data, use_segments, use_keypoints, args, kwargs
        del data, use_segments, use_keypoints, args, kwargs

    @pytest.fixture
    def get_cache_labels_data(self):
        cache_saving_path = Path(
            'ultralytics/data/tests/tiny-Pear-Tree-DB-5/train/Spots 1/Stitiching All Channels _000_Region1_ChannelViolet,SYTOXGreen,Orange,Far Red_Seq0001_C020022_day_0_[ims1_2023-11-06T17-24-17.cache')
        expected_label_0 = {
            'im_file': 'ultralytics/data/tests/tiny-Pear-Tree-DB-5/train/Spots 1/Stitiching All Channels _000_Region1_ChannelViolet,SYTOXGreen,Orange,Far Red_Seq0001_C020022_day_0_[ims1_2023-11-06T17-24-17.631]/Stitiching All Channels _000_Region1_ChannelViolet,SYTOXGreen,Orange,Far Red_Seq0001_C020022_day_0_[ims1_2023-11-06T17-24-17.631]_crop_X02432_Y01920_Z00013.npy',
            'shape': (640, 640),
            'cls': np.array([[0], [0]], dtype=np.float32),
            'bboxes': np.array(
                [[0.010937, 0.31641, 0.00625, 0.0046875], [0.60469, 0.85469, 0.00625, 0.00625]],
                dtype=np.float32
            ),
            'segments': [],
            'keypoints': None,
            'normalized': True,
            'bbox_format': 'xywh'
        }
        yield cache_saving_path, expected_label_0
        del cache_saving_path, expected_label_0

    @pytest.fixture
    def get_obj(self, get_initial_config):
        data, use_segments, use_keypoints, args, kwargs = get_initial_config
        obj = YOLODataset(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)
        yield obj
        del obj

    def test_init(self, get_obj, get_initial_config):
        obj = get_obj
        data, use_segments, use_keypoints, args, kwargs = get_initial_config
        assert obj.use_segments == use_segments
        assert obj.use_keypoints == use_keypoints
        assert obj.data == data

    def test_cache_labels(self, get_obj, get_cache_labels_data):
        obj = get_obj
        cache_saving_path, expected_label_0 = get_cache_labels_data

        if os.path.isfile(cache_saving_path):
            os.remove(cache_saving_path)

        x = obj.cache_labels(cache_saving_path)

        for key in ['labels', 'hash', 'results', 'msgs']:
            assert key in x.keys()

        assert len(x['results']) == 5
        assert len(x['labels']) == 48
        assert len(x['hash']) == 64
        assert len(x['msgs']) == 0

        assert expected_label_0.keys() == x['labels'][0].keys()
        for expected_val, actual_val in zip(expected_label_0.values(), x['labels'][0].values()):
            if isinstance(expected_val, np.ndarray):
                assert np.allclose(expected_val, actual_val, 1e-5, 1e-5)
            else:
                assert expected_val == actual_val

        assert os.path.isfile(cache_saving_path)
        cache_saving_path.unlink()
        assert not os.path.isfile(cache_saving_path)

    def test_get_labels(self, get_obj, get_cache_labels_data):
        obj = get_obj
        labels = obj.get_labels()
        cache_saving_path, expected_label_0 = get_cache_labels_data

        assert len(labels) == 48, len(labels)
        assert expected_label_0.keys() == labels[0].keys()
        for expected_val, actual_val in zip(expected_label_0.values(), labels[0].values()):
            if isinstance(expected_val, np.ndarray):
                assert np.allclose(expected_val, actual_val, 1e-5, 1e-5)
            else:
                assert expected_val == actual_val

        assert os.path.isfile(cache_saving_path)
        cache_saving_path.unlink()
        assert not os.path.isfile(cache_saving_path)

    def test_build_transforms(self, get_obj, get_initial_config):
        obj = get_obj
        data, use_segments, use_keypoints, args, kwargs = get_initial_config
        actual_transforms = obj.build_transforms(kwargs['hyp'])
        assert isinstance(actual_transforms, Compose), type(actual_transforms)

    def test_close_mosaic(self, get_obj, get_initial_config):
        obj = get_obj
        data, use_segments, use_keypoints, args, kwargs = get_initial_config
        obj.close_mosaic(kwargs['hyp'])
        assert isinstance(obj.transforms, Compose), type(obj.transforms)

    def test_update_labels_info(self, get_obj):
        obj = get_obj
        with open(LABEL_2D_PATH, 'rb') as f:
            label = pickle.load(f)
            removed_keys = ('bboxes', 'segments', 'keypoints', 'bbox_format', 'normalized',)

            assert isinstance(label, dict), type(label)
            assert set(removed_keys).issubset(set(label.keys()))
            assert 'instances' not in label.keys()

            label = obj.update_labels_info(label)
            assert isinstance(label, dict), type(label)
            assert isinstance(label['instances'], Instances), type(label['instances'])
            assert set(removed_keys).isdisjoint(set(label.keys()))

    def test_collate_fn(self, get_obj):
        obj = get_obj
        with open(BATCH_2D_PATH, 'rb') as f:
            batch = pickle.load(f)

            new_batch = obj.collate_fn(batch)
            expected_new_batch_ori_shape = ((640, 640), (640, 640), (640, 640), (640, 640),
                                            (640, 640), (640, 640), (640, 640), (640, 640))
            expected_new_batch_resized_shape = expected_new_batch_ori_shape
            expected_new_batch_ratio_pad = (
                ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0)),
                ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0)), ((1.0, 1.0), (0, 0))
            )
            expected_new_batch_img_size = torch.Size([8, 1, 640, 640])
            expected_new_batch_cls = torch.tensor([[0.]])
            expected_new_batch_bboxes = torch.tensor([[0.8773, 0.7930, 0.0109, 0.0109]])
            expected_new_batch_idx = torch.tensor([0.])
            assert isinstance(new_batch, dict), type(new_batch)
            assert isinstance(new_batch['im_file'], tuple), type(new_batch['im_file'])
            assert len(new_batch['im_file']) == 8, len(new_batch['im_file'])
            assert new_batch["ori_shape"] == expected_new_batch_ori_shape, \
                f'{new_batch["ori_shape"]} != {expected_new_batch_ori_shape}'
            assert new_batch["resized_shape"] == expected_new_batch_resized_shape, \
                f'{new_batch["resized_shape"]} != {expected_new_batch_resized_shape}'
            assert new_batch["ratio_pad"] == expected_new_batch_ratio_pad, \
                f'{new_batch["ratio_pad"]} != {expected_new_batch_ratio_pad}'
            assert new_batch["img"].size() == expected_new_batch_img_size, \
                f'{new_batch["img"]} != {expected_new_batch_img_size}'
            assert torch.equal(new_batch['cls'], expected_new_batch_cls)
            assert torch.allclose(new_batch['bboxes'], expected_new_batch_bboxes, 1e-4, 1e-4)
            assert torch.equal(new_batch['batch_idx'], expected_new_batch_idx)
