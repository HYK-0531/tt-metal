# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class PBKey:
    shape: any = ()
    dtype: any = None
    memory_config: any = None


def create_buffer(mesh_device, pb_key):
    return ttnn.from_torch(
        torch.zeros(pb_key.shape),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=pb_key.dtype,
        memory_config=pb_key.memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def create_ag_persistent_output_buffers(mesh_device, model):
    # output buffer must match the config expected in the model

    persistent_buffers = {}

    shape = (1, 1, 128, 256)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [32, 320],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 152064)
    dtype = ttnn.bfloat8_b
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 256)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 4))}),
            [32, 128],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))}),
            [32, 160],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 256)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 128, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    return persistent_buffers


def create_rs_persistent_intermediate_buffers(mesh_device, model):
    # intermediate buffers can always be L1 (if we have space),
    # only the output buffers needs to match the config expected in the model

    # currently can't really have sharded intermediate buffers, as we don't know
    # the sharding cores, shard shape, etc to use for the intermediate tensor

    persistent_buffers = {}

    shape = (1, 1, 32, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.L1_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 128, 5120)
    dtype = ttnn.bfloat8_b
    memory_config = ttnn.L1_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 128, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.L1_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 5120)
    dtype = ttnn.bfloat8_b
    memory_config = ttnn.L1_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 5120)
    dtype = ttnn.bfloat16
    memory_config = ttnn.L1_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    return persistent_buffers


def create_rs_persistent_output_buffers(mesh_device, model):
    # output buffer must match the config expected in the model

    persistent_buffers = {}

    shape = (1, 1, 32, 640)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}),
            [32, 320],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 32, 640)
    dtype = ttnn.bfloat16
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(4, 3))}),
            [32, 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 128, 640)
    dtype = ttnn.bfloat8_b
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 128, 640)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 640)
    dtype = ttnn.bfloat8_b
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    shape = (1, 1, 256, 640)
    dtype = ttnn.bfloat16
    memory_config = ttnn.DRAM_MEMORY_CONFIG
    pb_key = PBKey(shape=shape, dtype=dtype, memory_config=memory_config)
    persistent_buffers[pb_key] = create_buffer(mesh_device=mesh_device, pb_key=pb_key)

    return persistent_buffers
