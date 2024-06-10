# -*- coding: utf-8 -*-
"""
ultralytics/utils/tests/test_loss.py

Copyright 2024 (C) Pear Bio Ltd
All rights reserved.
Original Author: Giussepi Lopez
"""

import pytest
import torch

from ultralytics.utils.loss import CenterLoss


@pytest.mark.pearbioquicktests
class TestCenterLoss:

    @pytest.fixture
    def get_data(self) -> tuple:
        fg_mask = torch.tensor([[True, True, False, False, True]], dtype=torch.bool)

        pred_bboxes = torch.tensor([[[36.6310,  5.9899, 38.4746,  9.5697],
                                     [10.0000, 10.0000, 20.0000, 20.0000],
                                     [35.8611, 20.5819, 36.9993, 23.3141],
                                     [73.8761, 24.0831, 75.2317, 25.5161],
                                     [27.6927, 43.2080, 29.2259, 45.3122]]])

        target_bboxes = torch.tensor([[[37.3750,  7.8750, 38.0000,  8.6250],
                                       [21.0000, 21.0000, 31.0000, 31.0000],
                                       [36.1250, 22.3750, 36.6250, 22.8750],
                                       [74.1250, 24.3750, 74.8750, 25.2500],
                                       [28.1250, 44.0000, 28.8750, 45.0000]]])

        yield pred_bboxes, target_bboxes, fg_mask
        del pred_bboxes, target_bboxes, fg_mask

    @pytest.fixture
    def get_obj(self, request):
        obj = CenterLoss(request.param)
        yield obj
        del obj

    @pytest.mark.parametrize('get_obj', [.0], indirect=True)
    def test_compare_centres_tolerance_0(self, get_obj, get_data):
        obj = get_obj
        pred_bboxes, target_bboxes, fg_mask = get_data
        predicted_outcome = obj.compare_centres(pred_bboxes, target_bboxes, fg_mask)

        pred_widths = torch.tensor([1.8436, 10, 1.5332])
        pred_heights = torch.tensor([3.5798, 10,  2.1042])

        target_widths = torch.tensor([0.625, 10, 0.75])
        target_heights = torch.tensor([0.75, 10, 1])

        pred_centers_x = torch.tensor([37.5528, 15, 28.4593])
        pred_centers_y = torch.tensor([7.7798, 15, 44.2601])

        target_centers_x = torch.tensor([37.6875, 26, 28.5])
        target_centers_y = torch.tensor([8.25, 26, 44.5])

        actual_centre_distance_x = abs(pred_centers_x - target_centers_x)
        actual_centre_distance_y = abs(pred_centers_y - target_centers_y)

        max_centre_dist_x = (target_widths + pred_widths) / 2
        max_centre_dist_y = (target_heights + pred_heights) / 2

        x_diff = (max_centre_dist_x - actual_centre_distance_x) / max_centre_dist_x
        y_diff = (max_centre_dist_y - actual_centre_distance_y) / max_centre_dist_y

        expected_outcome = (x_diff + y_diff) / 2
        expected_outcome[expected_outcome < 0] = 0

        torch.equal(
            (expected_outcome == 0),
            torch.tensor([False,  True, False], dtype=torch.bool)
        )
        assert torch.allclose(predicted_outcome, expected_outcome.unsqueeze(-1))

    @pytest.mark.parametrize('get_obj', [.3], indirect=True)
    def test_compare_centres_tolerance_dot_3(self, get_obj, get_data):
        obj = get_obj
        pred_bboxes, target_bboxes, fg_mask = get_data
        predicted_outcome = obj.compare_centres(pred_bboxes, target_bboxes, fg_mask)

        pred_widths = torch.tensor([1.8436, 10, 1.5332])
        pred_heights = torch.tensor([3.5798, 10,  2.1042])

        target_widths = torch.tensor([0.625, 10, 0.75])
        target_heights = torch.tensor([0.75, 10, 1])

        pred_centers_x = torch.tensor([37.5528, 15, 28.4593])
        pred_centers_y = torch.tensor([7.7798, 15, 44.2601])

        target_centers_x = torch.tensor([37.6875, 26, 28.5])
        target_centers_y = torch.tensor([8.25, 26, 44.5])

        actual_centre_distance_x = abs(pred_centers_x - target_centers_x) * (1 - .3)
        actual_centre_distance_y = abs(pred_centers_y - target_centers_y) * (1 - .3)

        max_centre_dist_x = (target_widths + pred_widths) / 2
        max_centre_dist_y = (target_heights + pred_heights) / 2

        x_diff = (max_centre_dist_x - actual_centre_distance_x) / max_centre_dist_x
        y_diff = (max_centre_dist_y - actual_centre_distance_y) / max_centre_dist_y

        expected_outcome = (x_diff + y_diff) / 2
        expected_outcome[expected_outcome < 0] = 0

        torch.equal(
            (expected_outcome == 0),
            torch.tensor([False,  False, False], dtype=torch.bool)
        )
        assert torch.allclose(predicted_outcome, expected_outcome.unsqueeze(-1))

    @pytest.mark.parametrize('get_obj', [.0], indirect=True)
    def test_compute_loss(self, get_obj, get_data):
        obj = get_obj
        pred_bboxes, target_bboxes, fg_mask = get_data
        predicted_outcome = obj.compute_loss(pred_bboxes, target_bboxes, fg_mask)

        pred_widths = torch.tensor([1.8436, 10, 1.5332])
        pred_heights = torch.tensor([3.5798, 10,  2.1042])

        target_widths = torch.tensor([0.625, 10, 0.75])
        target_heights = torch.tensor([0.75, 10, 1])

        pred_centers_x = torch.tensor([37.5528, 15, 28.4593])
        pred_centers_y = torch.tensor([7.7798, 15, 44.2601])

        target_centers_x = torch.tensor([37.6875, 26, 28.5])
        target_centers_y = torch.tensor([8.25, 26, 44.5])

        actual_centre_distance_x = abs(pred_centers_x - target_centers_x)
        actual_centre_distance_y = abs(pred_centers_y - target_centers_y)

        max_centre_dist_x = (target_widths + pred_widths) / 2
        max_centre_dist_y = (target_heights + pred_heights) / 2

        x_diff = (max_centre_dist_x - actual_centre_distance_x) / max_centre_dist_x
        y_diff = (max_centre_dist_y - actual_centre_distance_y) / max_centre_dist_y

        expected_outcome = (x_diff + y_diff) / 2
        expected_outcome[expected_outcome < 0] = 0
        expected_outcome = (1 - expected_outcome.unsqueeze(-1)).sum() / fg_mask.sum()

        assert expected_outcome.ndim == 0
        assert expected_outcome.numel() == 1
        assert torch.allclose(predicted_outcome, expected_outcome)

    @pytest.mark.parametrize('get_obj', [.3], indirect=True)
    def test_forward(self, get_obj, get_data):
        obj = get_obj
        pred_bboxes, target_bboxes, fg_mask = get_data
        torch.equal(
            obj.compute_loss(pred_bboxes, target_bboxes, fg_mask),
            obj(pred_bboxes, target_bboxes, fg_mask)
        )
