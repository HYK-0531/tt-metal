# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
from models.experimental.stable_diffusion_xl_base.demo.demo import test_demo
from models.experimental.stable_diffusion_xl_base.tests.test_sdxl_generate_images_and_prompts import sdxl_generate_images_and_prompts
from models.experimental.stable_diffusion_xl_base.utils.sdxl_calculate_save_clip_and_fid import sdxl_calculate_save_clip_and_fid
from models.experimental.stable_diffusion_xl_base.tests.test_common import SDXL_L1_SMALL_SIZE

# test_sdxl_generate_images_and_prompts.__test__ = False

@pytest.mark.parametrize("device_params", [{"l1_small_size": SDXL_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "num_inference_steps",
    ((50),),
)
@pytest.mark.parametrize(
    "vae_on_device",
    [
        (True),
        (False),
    ],
    ids=("device_vae", "host_vae"),
)
@pytest.mark.parametrize("captions_path", ["models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"])
@pytest.mark.parametrize("coco_statistics_path", ["models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"])
def test_accuracy_sdxl(
    mesh_device,
    is_ci_env,
    num_inference_steps,
    vae_on_device,
    captions_path,
    coco_statistics_path,
    evaluation_range,
):    
    images, prompts = sdxl_generate_images_and_prompts(
        mesh_device,
        is_ci_env,
        num_inference_steps,
        vae_on_device,
        captions_path,
        evaluation_range,
    )
    
    sdxl_calculate_save_clip_and_fid(
        vae_on_device,
        coco_statistics_path,
        evaluation_range,
        images,
        prompts
    )