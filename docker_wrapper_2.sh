#!/bin/bash

sudo docker exec \
  -e ARCH_NAME=wormhole_b0 \
  -e TT_METAL_HOME=/home/aho/tt-metal \
  -e PYTHONPATH=/home/aho/tt-metal \
  -e TT_METAL_ENV=dev \
  -e TT_MESH_ID=1 \
  -e TT_HOST_RANK=0 \
  test-container-aho orted "$@"
