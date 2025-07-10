# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
from enum import Enum

import torch

import ttnn


class PBType(Enum):
    INTERMEDIARY = "intermediary"
    OUTPUT = "output"


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.ag_semaphores_idx = 0
        self.ag_semaphore_handles = [[], []]

        self.rs_semaphores_idx = 0
        self.rs_semaphore_handles = [[], []]

        self.output_buffer_specs = set()
        self.intermediary_buffer_specs = set()

        self.intermediary_buffers = {}
        self.output_buffers = {}

        self.initialize_persistent_buffers()

        for i in range(2):
            for _ in range(2):
                self.ag_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
            for _ in range(3):
                self.rs_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

        worker_sub_device = ttnn.SubDevice([self.sub_device_crs])
        sub_device_manager = self.mesh_device.create_sub_device_manager([worker_sub_device], 0)
        self.mesh_device.load_sub_device_manager(sub_device_manager)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def get_and_cycle_ag_semaphore_handles(self):
        current_idx = self.ag_semaphores_idx
        self.ag_semaphores_idx = (self.ag_semaphores_idx + 1) % 2
        return self.ag_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (self.rs_semaphores_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]

    def create_persistent_buffer(self, shape, mem_config, dtype, distributed=False):
        if distributed:
            shape[3] *= self.mesh_device.get_num_devices()
            cluster_shape = list(self.mesh_device.shape)
            # print("cluster_shape: ", cluster_shape)
            mesh_mapper = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, 3), mesh_shape=cluster_shape)
        else:
            mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
        return ttnn.from_torch(
            torch.zeros(shape),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=mem_config,
            mesh_mapper=mesh_mapper,
        )

    def spec_to_json(self, spec):
        shape, mem_config, dtype = spec

        mem_config_json = {
            "memory_layout": mem_config.memory_layout.name,
            "buffer_type": mem_config.buffer_type.name,
            "created_with_nd_shard_spec": mem_config.created_with_nd_shard_spec
            if hasattr(mem_config, "created_with_nd_shard_spec")
            else 0,
        }

        # Handle shard_spec if present
        if mem_config.shard_spec is not None:
            shard_spec = mem_config.shard_spec
            mem_config_json["shard_spec"] = {
                "grid": {
                    "start": {"x": 0, "y": 0},
                    "end": {
                        "x": shard_spec.grid.bounding_box().grid_size().x - 1,
                        "y": shard_spec.grid.bounding_box().grid_size().x - 1,
                    },
                },
                "shape": list(shard_spec.shape),
                "orientation": shard_spec.orientation.name,
                "mode": shard_spec.mode.name,
            }
        else:
            mem_config_json["shard_spec"] = None

        return {
            "shape": list(shape),
            "mem_config": mem_config_json,
            "dtype": dtype.name if hasattr(dtype, "name") else str(dtype),
        }

    def spec_from_json(self, obj):
        shape = tuple(obj["shape"])

        mem_layout = getattr(ttnn.TensorMemoryLayout, obj["mem_config"]["memory_layout"])
        buf_type = getattr(ttnn.BufferType, obj["mem_config"]["buffer_type"])
        created_with_nd = obj["mem_config"].get("created_with_nd_shard_spec", 0)

        shard_spec_json = obj["mem_config"].get("shard_spec")
        shard_spec = None

        if shard_spec_json is not None:
            grid_start = ttnn.CoreCoord(shard_spec_json["grid"]["start"]["x"], shard_spec_json["grid"]["start"]["y"])
            grid_end = ttnn.CoreCoord(shard_spec_json["grid"]["end"]["x"], shard_spec_json["grid"]["end"]["y"])
            grid_range = ttnn.CoreRange(grid_start, grid_end)
            grid = ttnn.CoreRangeSet({grid_range})
            orientation = getattr(ttnn.ShardOrientation, shard_spec_json["orientation"])
            mode = getattr(ttnn.ShardMode, shard_spec_json["mode"])
            shard_shape = tuple(shard_spec_json["shape"])

            # print(type(grid), type(shard_shape), type(orientation), type(mode))
            shard_spec = ttnn.ShardSpec(
                grid=grid,
                shard_shape=list(shard_shape),
                shard_orientation=orientation,
                shard_mode=mode,
            )

        mem_config = ttnn.MemoryConfig(
            memory_layout=mem_layout,
            buffer_type=buf_type,
            shard_spec=shard_spec,
            # created_with_nd_shard_spec=created_with_nd
        )

        dtype = getattr(ttnn.DataType, obj["dtype"])

        return (shape, mem_config, dtype)

    def save_specs_to_file(self, specs_set, filename):
        json_data = [self.spec_to_json(spec) for spec in specs_set]
        with open(filename, "w") as f:
            json.dump(json_data, f, indent=2)

    def load_specs_from_file(self, filename):
        with open(filename, "r") as f:
            return set(self.spec_from_json(spec) for spec in json.load(f))

    def initialize_persistent_buffers(self):
        # read intermediary buffer specs set from json file
        # self.intermediary_buffer_specs = self.load_specs_from_file(
        #     "models/demos/t3000/mixtral8x7b/tt/intermediary_PB_specs.json"
        # )
        # for shape, mem_config, dtype in self.intermediary_buffer_specs:
        #     self.get_or_add_persistent_buffer(shape, mem_config, dtype, PBType.INTERMEDIARY)
        # read output buffer specs set from json file
        self.output_buffer_specs = self.load_specs_from_file("models/demos/t3000/mixtral8x7b/tt/output_PB_specs.json")
        for shape, mem_config, dtype in self.output_buffer_specs:
            self.get_or_add_persistent_buffer(shape, mem_config, dtype, PBType.OUTPUT)

    def get_or_add_persistent_buffer(self, shape, mem_config, dtype, buffer_type: PBType, distributed=False):
        buffer_spec = (tuple(shape), mem_config, dtype)

        # if(buffer_type == PBType.INTERMEDIARY):
        # distributed = True  # Intermediary buffers are always distributed

        buffer_dict_name = f"{buffer_type.value}_buffers"
        specs_set_name = f"{buffer_type.value}_buffer_specs"

        buffer_dict = getattr(self, buffer_dict_name)
        specs_set = getattr(self, specs_set_name)

        # if buffer_spec in buffer_dict: assert buffer_dict[buffer_spec].is_allocated(), (
        #     f"Buffer {buffer_spec} already exists but is not allocated."
        # )

        if buffer_spec not in buffer_dict or not buffer_dict[buffer_spec].is_allocated():
            if buffer_spec not in specs_set:
                print(f"adding {buffer_type.value} buffer_spec: ", buffer_spec)
            buffer_dict[buffer_spec] = self.create_persistent_buffer(list(shape), mem_config, dtype, distributed)
            # Overwrite json file storing different buffer specs
            if buffer_spec not in specs_set:
                specs_set.add(buffer_spec)
                self.save_specs_to_file(
                    buffer_dict.keys(), f"models/demos/t3000/mixtral8x7b/tt/{buffer_type.value}_PB_specs.json"
                )
        # else:
        # print(f"{buffer_type.value} buffer_spec already exists: {buffer_spec}")
        return buffer_dict[buffer_spec]

    def close(self):
        self.mesh_device.reset_sub_device_stall_group()
