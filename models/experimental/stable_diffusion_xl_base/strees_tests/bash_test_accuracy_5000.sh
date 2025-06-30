#!/bin/bash

TOTAL_IMAGES=${1:-4}
BATCH_SIZE=${2:-2}
RESET_BOOL=${3:-1}
VAE_ON_DEVICE=${4:-"host_vae"}
COCO_STATISTICS_PATH=${5:-"models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"}

echo "Generate all images, with optional periodic reset"
for ((START=0; START<TOTAL_IMAGES; START+=BATCH_SIZE)); do

    if (( START + BATCH_SIZE > TOTAL_IMAGES )); then
        BATCH_CURRENT=$((TOTAL_IMAGES-START))
    else
        BATCH_CURRENT=$BATCH_SIZE
    fi
    echo "Running batch from $START to $((START + BATCH_CURRENT))"
           
    pytest models/experimental/stable_diffusion_xl_base/tests/test_sdxl_generate_images_and_prompts.py \
        --start-from=$START \
        --num-prompts=$BATCH_CURRENT \
        -k $VAE_ON_DEVICE || {
            echo "Batch failed at $START, exiting"
            exit 1
        }

    if (( RESET_BOOL == 1 && START + BATCH_SIZE < TOTAL_IMAGES )); then
        echo "RESTARTING"
        tt-smi -r
    fi
done

echo "Running post-processing and evaluation..."
python models/experimental/stable_diffusion_xl_base/utils/sdxl_collect_data.py "$VAE_ON_DEVICE" "$COCO_STATISTICS_PATH" "$TOTAL_IMAGES"



