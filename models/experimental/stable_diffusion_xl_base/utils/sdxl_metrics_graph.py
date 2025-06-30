# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import matplotlib.pyplot as plt
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder

import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score

def sdxl_metrix_graph(
    coco_statistics_path,
    images,
    prompts,
    period
):

    clip = CLIPEncoder()
    clip_scores = []
    x_axis = []
    num_prompts = len(prompts)
    print("len: ", len(images))

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())
    average_clip_score = sum(clip_scores) / len(clip_scores)
    deviation_clip_score = statistics.stdev(clip_scores)
    
    average_clip_iterative, fid_iterative = [], []
    for index in range(period-1, num_prompts, period):
        average_clip_iterative.append(sum(clip_scores[:index + 1]) / (index + 1))
        
        current_fid_score = calculate_fid_score(images[:index+1], coco_statistics_path)
        fid_iterative.append(current_fid_score)
        
        x_axis.append(index + 1)
    
    last_fid = fid_iterative[-1]
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Number of images generated')
    ax1.set_ylabel('Average CLIP score', color='tab:blue')
    ax1.plot(x_axis, average_clip_iterative, label='Avg CLIP', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('FID', color='tab:red')
    ax2.plot(x_axis, fid_iterative, label='FID', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    title = f"Avg CLIP: {average_clip_score:.2f}, σ: {deviation_clip_score:.2f}, Last FID: {last_fid:.2f}" if last_fid else \
            f"Avg CLIP: {average_clip_score:.2f}, σ: {deviation_clip_score:.2f}, FID: N/A"

    plt.title(title)
    fig.tight_layout()
    plt.grid()
    plt.savefig("metric_plot.png")
    # plt.show()
   
    
    