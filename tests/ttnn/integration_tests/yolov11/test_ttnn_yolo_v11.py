# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import sys
import torch.nn as nn
import torch
import ttnn
import logging
from ultralytics import YOLO
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.yolov11.tt.model_preprocessing import (
    create_yolov11_input_tensors,
    create_yolov11_model_parameters,
)
from models.experimental.yolov11.reference import yolov11
from models.experimental.yolov11.tt import ttnn_yolov11
from models.utility_functions import run_for_wormhole_b0


@pytest.mark.parametrize(
    "resolution",
    [
        # ([1, 3, 224, 224]),
        ([1, 3, 640, 640]),
    ],
)
@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "use_weights_from_ultralytics",
    [
        True,
        # False
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_yolov11(device, use_program_cache, reset_seeds, resolution, use_weights_from_ultralytics):
    weights = "yolo11n.pt"
    if use_weights_from_ultralytics:
        torch_model = YOLO(weights)
        state_dict = {k.replace("model.", "", 1): v for k, v in torch_model.state_dict().items()}

    torch_model = yolov11.YoloV11()
    # print("model is ", torch_model)
    torch_model.eval()
    if use_weights_from_ultralytics:
        torch_model.load_state_dict(state_dict)
    torch_input, ttnn_input = create_yolov11_input_tensors(
        device, batch=resolution[0], input_channels=resolution[1], input_height=resolution[2], input_width=resolution[3]
    )
    torch_output = torch_model(torch_input)
    parameters = create_yolov11_model_parameters(torch_model, torch_input, device=device)
    ttnn_model = ttnn_yolov11.YoloV11(device, parameters)
    ttnn_output = ttnn_model(ttnn_input)
    ttnn_output = ttnn.to_torch(ttnn_output)
    print("final shapesa re", torch_output.shape, ttnn_output.shape)
    # if resolution[2]==224:
    #     ttnn_output = ttnn_output.reshape(1, 7, 7, 256).permute(0, 3, 1, 2)
    # else:
    #     ttnn_output = ttnn_output.reshape(1, 20, 20, 256).permute(0, 3, 1, 2)
    # ttnn_output = ttnn_output.reshape(torch_output.shape)
    assert_with_pcc(torch_output, ttnn_output, 1.0)
