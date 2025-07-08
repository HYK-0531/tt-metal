// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/device/all_to_all_dispatch_device_operation.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric_edm_types.hpp>

namespace ttnn::operations::experimental::ccl {

// Reuse the detail functions from all_to_all_dispatch
// Reuse the detail functions from all_to_all_dispatch directly

struct AllGatherFabric2DDeviceOperation {
    struct operation_attributes_t {
        const uint32_t dim;
        const uint32_t cluster_axis;
        const std::vector<GlobalSemaphore> multi_device_global_semaphore;
        const uint32_t num_links;
        const std::optional<MemoryConfig> output_mem_config;
        const ttnn::ccl::Topology topology;
        const std::optional<tt::tt_metal::SubDeviceId> subdevice_id;
    };

    struct tensor_args_t {
        const Tensor input_tensor;
    };

    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct AllGatherFabric2DAdd {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id;
            tt::tt_metal::KernelHandle writer_kernel_id;
            CoreCoord core;
        };
        using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

        static cached_mesh_workload_t create_mesh_workload(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinateRangeSet& tensor_coords,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
            const operation_attributes_t& operation_attributes,
            const ttnn::MeshCoordinate& mesh_coordinate,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const ttnn::MeshCoordinateRangeSet& tensor_coords);

        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<AllGatherFabric2DAdd>;

    // Mandatory methods
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace ttnn::operations::experimental::ccl

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as
// ttnn::prim::all_gather_fabric_2d
constexpr auto all_gather_fabric_2d = ttnn::register_operation<
    "ttnn::prim::all_gather_fabric_2d",
    ttnn::operations::experimental::ccl::AllGatherFabric2DDeviceOperation>();
}  // namespace ttnn::prim
