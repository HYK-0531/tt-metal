# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
import os
from loguru import logger
import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score


import json


def sdxl_calculate_save_clip_and_fid(
    vae_on_device,
    coco_statistics_path,
    evaluation_range,
    images,
    prompts
):
    start_from, num_prompts = evaluation_range
    print("sdxl_calculate_save_clip_and_fid: start_from, num_prompts: ", start_from, num_prompts)  
    print("start_from, num_prompts: ", start_from, num_prompts)
    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if num_prompts >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, coco_statistics_path)
    else:
        logger.info("FID score is not calculated for less than 2 prompts.")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    data = {
        "model": "sdxl",  # For compatibility with current processes
        "metadata": {
            "device": "N150",
            "device_vae": vae_on_device,
            "start_from": start_from,
            "num_prompts": num_prompts,
            "model_name": "sdxl",
        },
        "benchmarks_summary": [
            {
                "device": "N150",
                "model": "sdxl",
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "fid_score": fid_score,
            }
        ],
    }

    out_root, file_name = "test_reports", "sdxl_test_results.json"
    os.makedirs(out_root, exist_ok=True)

    with open(f"{out_root}/{file_name}", "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Test results saved to {out_root}/{file_name}")