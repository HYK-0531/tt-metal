# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo import test_demo
from loguru import logger
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE, SDXL_DEFAULT_PROMPT

test_demo.__test__ = False

import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "num_images_per_device",
    ((1),),
)
def test_sdxl_stress(
    mesh_device,
    num_inference_steps,
    num_images_per_device,
    evaluation_range,
):
    prompts = (mesh_device.get_num_devices() * num_images_per_device) * [SDXL_DEFAULT_PROMPT]
    print("Prompts len is: ", len(prompts))
    print("Prompts is: ", prompts)

    os.environ["TT_MM_THROTTLE_PERF"] = "5"

    test_demo(
        mesh_device,
        True,  # do not save images
        prompts,
        num_inference_steps,
        True,  # vae on device
        evaluation_range,
    )

    logger.info(f"Success!")
