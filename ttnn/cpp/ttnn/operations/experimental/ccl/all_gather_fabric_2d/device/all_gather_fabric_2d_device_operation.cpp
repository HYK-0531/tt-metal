// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "all_gather_fabric_2d_device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::ccl {

AllGatherFabric2DDeviceOperation::program_factory_t AllGatherFabric2DDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return AllGatherFabric2DAdd{};
}

void AllGatherFabric2DDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // Basic validation - can be expanded later
    auto input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
}

void AllGatherFabric2DDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {}

AllGatherFabric2DDeviceOperation::spec_return_value_t AllGatherFabric2DDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    const auto& input_spec = input_tensor.tensor_spec();
    auto input_shape = input_spec.logical_shape();

    // For now, return the same shape as input
    // This would be modified based on the actual all_gather logic
    return {TensorSpec(
        Shape(input_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), input_tensor.memory_config()))};
}

AllGatherFabric2DDeviceOperation::tensor_return_value_t AllGatherFabric2DDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

std::tuple<AllGatherFabric2DDeviceOperation::operation_attributes_t, AllGatherFabric2DDeviceOperation::tensor_args_t>
AllGatherFabric2DDeviceOperation::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    return {
        operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .multi_device_global_semaphore = multi_device_global_semaphore,
            .num_links = num_links,
            .output_mem_config = memory_config,
            .topology = topology,
            .subdevice_id = subdevice_id,
        },
        tensor_args_t{.input_tensor = input_tensor}};
}

}  // namespace ttnn::operations::experimental::ccl
