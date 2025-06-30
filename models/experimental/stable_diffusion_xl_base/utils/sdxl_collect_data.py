# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import csv
from PIL import Image

from models.experimental.stable_diffusion_xl_base.utils.sdxl_calculate_save_clip_and_fid import sdxl_calculate_save_clip_and_fid
from models.experimental.stable_diffusion_xl_base.utils.sdxl_metrics_graph import sdxl_metrix_graph


CAPTIONS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"
IMAGES_PATH, IMAGE_NAME_BASE = "output", "output"

def sdxl_collect_first_n_images(n_images):
    collected_images = []
    for index in range(n_images):
        current_filename_path = f"{IMAGES_PATH}/{IMAGE_NAME_BASE}{index+1}.png"
        img = Image.open(current_filename_path).convert("RGB")
        collected_images.append(img)
    return collected_images

def sdxl_collect_n_prompts_from_start(n_prompts):
    prompts = []
    with open(CAPTIONS_PATH, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for idx, row in enumerate(reader):
            if idx >= n_prompts:
                break
            prompts.append(row[2])
    return prompts

def sdxl_collect_results(
    vae_on_device,
    coco_statistics_path,
    n_prompts
    ):
    
    images = sdxl_collect_first_n_images(n_prompts)
    prompts = sdxl_collect_n_prompts_from_start(n_prompts)

    sdxl_metrix_graph(coco_statistics_path, images, prompts, 5)
    
    # sdxl_calculate_save_clip_and_fid(
    #     vae_on_device,
    #     coco_statistics_path,
    #     evaluation_range = (0, n_prompts),
    #     images = images,
    #     prompts = prompts   
    # )
    
if __name__ == "__main__":
    import sys
    sdxl_collect_results(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    
