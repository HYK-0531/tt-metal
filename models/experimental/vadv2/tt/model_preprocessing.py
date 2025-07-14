# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.vadv2.reference.backbone import ResNet
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters["res_model"][prefix] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    bn = getattr(block, f"bn{conv_name[-1]}")
                    w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                    parameters["res_model"][prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

    return parameters


def create_vadv2_model_parameters(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters
