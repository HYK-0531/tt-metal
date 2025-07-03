# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger
import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def create_global_semaphores(mesh_device, num_devices, cores, initial_value, num_links):
    # create global semaphore handles
    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, cores, initial_value) for _ in range(2)]
    return ccl_semaphore_handles


def run_all_gather_impl(
    t3k_mesh_device,
    num_devices,
    ag_input_shape,
    dim,
    num_links,
    ag_input_dtype,
    layout,
    mem_config_input,
    mem_config_ag,
    ag_topology,
    num_iters=1,
    enable_trace=True,
    cluster_axis=None,
):
    torch.manual_seed(0)

    ##### Fabric setup #####
    compute_grid_size = t3k_mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice(
        [
            ccl_sub_device_crs,
        ]
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    sub_device_manager = t3k_mesh_device.create_sub_device_manager([worker_sub_device], 0)
    t3k_mesh_device.load_sub_device_manager(sub_device_manager)
    t3k_mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    # create global semaphore handles
    ccl_semaphore_handles = [
        create_global_semaphores(t3k_mesh_device, num_devices, ccl_sub_device_crs, 0, num_links)
        for _ in range(num_iters)
    ]

    ### Create persistent output buffers
    logger.info("Creating persistent buffers")
    # For all_gather, output is larger than input
    ag_output_shape = ag_input_shape[:]
    ag_output_shape[dim] *= num_devices
    persistent_output_buffers = [
        ttnn.from_torch(
            torch.zeros(ag_output_shape),
            device=t3k_mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ag_input_dtype,
            memory_config=mem_config_ag,
            mesh_mapper=ttnn.ReplicateTensorToMesh(t3k_mesh_device),
        )
        for _ in range(num_iters)
    ]

    logger.info("Done creating persistent buffers")

    ##### All gather input setup #####
    logger.info(f"All gather input shape: {ag_input_shape}")
    logger.info(f"All gather output shape: {ag_output_shape}")
    logger.info(f"All gather dim: {dim}")

    tt_input_tensor_mesh_list = []
    torch_input_tensor_list = []

    for i in range(num_iters):
        # For all_gather, create smaller per-device inputs that will be gathered
        input_tensors = []
        for device_idx in range(num_devices):
            ag_input_tensor = torch.rand(ag_input_shape).bfloat16()
            input_tensors.append(ag_input_tensor)

        torch_input_tensor_list.append(input_tensors)
        tt_input_tensors = []
        for j, t in enumerate(input_tensors):
            tt_input_tensors.append(ttnn.Tensor(t, ag_input_dtype).to(layout))
        input_tensor_mesh = ttnn.aggregate_as_tensor(tt_input_tensors).to(t3k_mesh_device, mem_config_input)

        tt_input_tensor_mesh_list.append(input_tensor_mesh)

    ##### Perform torch ops #####
    torch_all_gather_output_list = []
    for i in range(num_iters):
        # For all_gather, simply concatenate the inputs along the specified dimension
        ag_output = torch.cat(torch_input_tensor_list[i], dim=dim)
        torch_all_gather_output_list.append(ag_output)

    ##### Perform the TT ops #####
    tt_all_gather_output_list = []

    def run_op(i):
        tt_all_gather_output_tensor = ttnn.experimental.all_gather_async(
            tt_input_tensor_mesh_list[i],
            persistent_output_buffer=persistent_output_buffers[i],
            dim=dim,
            multi_device_global_semaphore=ccl_semaphore_handles[i],
            num_links=num_links,
            memory_config=mem_config_ag,
            topology=ag_topology,
            subdevice_id=worker_sub_device_id,
            cluster_axis=cluster_axis,
        )

        return tt_all_gather_output_tensor

    if enable_trace:
        # Compile the op
        for i in range(num_iters):
            tt_all_gather_output_tensor = run_op(i)
        logger.info(f"Done compiling Op")

        # Capture the trace
        trace_id = ttnn.begin_trace_capture(t3k_mesh_device, cq_id=0)
        for i in range(num_iters):
            tt_all_gather_output_tensor = run_op(i)
            tt_all_gather_output_list.append(tt_all_gather_output_tensor)
        ttnn.end_trace_capture(t3k_mesh_device, trace_id, cq_id=0)
        logger.info(f"Done capturing trace")

        # Execute trace
        ttnn.execute_trace(t3k_mesh_device, trace_id, cq_id=0, blocking=False)
        logger.info(f"Done executing trace")

        # Synchronize the devices
        ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
    else:
        for i in range(num_iters):
            tt_all_gather_output_tensor = run_op(i)
            tt_all_gather_output_list.append(tt_all_gather_output_tensor)

            logger.info(f"Waiting for op")
            ttnn.synchronize_device(t3k_mesh_device, sub_device_ids=sub_device_stall_group)
            logger.info(f"Done op")

            logger.info(f"Done iteration {i}")

    for i in range(num_iters):
        tt_ag_out_tensor = tt_all_gather_output_list[i]
        torch_ag_out_tensor = torch_all_gather_output_list[i]

        tt_ag_out_list = [ttnn.to_torch(tensor) for tensor in ttnn.get_device_tensors(tt_ag_out_tensor)]
        for tt_ag_out in tt_ag_out_list:
            eq, output = comp_pcc(tt_ag_out, torch_ag_out_tensor)
            logger.info(f"{output}, iteration {i}")
            assert eq, f"{i} FAILED ag: {output}"

    t3k_mesh_device.reset_sub_device_stall_group()
    t3k_mesh_device.clear_loaded_sub_device_manager()


@pytest.mark.parametrize("num_links", [1], ids=["1link"])
@pytest.mark.parametrize(
    "num_devices, ag_input_shape, dim, layout, ag_input_dtype",
    [
        (8, [8, 1, 512, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [4, 1, 1024, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [1, 1, 1024, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [1, 1, 352, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [2, 1, 2048, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
        (8, [1, 1, 4096, 640], 3, ttnn.TILE_LAYOUT, ttnn.bfloat16),  # use batching when fused
    ],
    ids=[
        "batch_8",
        "batch_4",
        "batch_1_sd35_spatial",
        "batch_1_sd35_prompt",
        "batch_2",
        "batch_1",
    ],
)
@pytest.mark.parametrize(
    "mem_config_input, mem_config_ag",
    [
        (
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        )
    ],
)
@pytest.mark.parametrize(
    "enable_trace, num_iters",
    [
        (True, 10),
        (False, 1),
    ],
    ids=["perf", "check"],
)
@pytest.mark.parametrize(
    "device_params, ag_topology",
    [
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 90112}, ttnn.Topology.Ring),
        ({"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 90112}, ttnn.Topology.Linear),
    ],
    indirect=["device_params"],
    ids=["fabric_ring", "fabric_linear"],
)
def test_all_gather_async(
    t3k_mesh_device,
    num_devices,
    num_links,
    ag_input_shape,
    dim,
    layout,
    ag_input_dtype,
    mem_config_input,
    mem_config_ag,
    enable_trace,
    num_iters,
    ag_topology,
):
    run_all_gather_impl(
        t3k_mesh_device,
        num_devices,
        ag_input_shape,
        dim,
        num_links,
        ag_input_dtype,
        layout,
        mem_config_input,
        mem_config_ag,
        ag_topology=ag_topology,
        enable_trace=enable_trace,
        num_iters=num_iters,
    )
