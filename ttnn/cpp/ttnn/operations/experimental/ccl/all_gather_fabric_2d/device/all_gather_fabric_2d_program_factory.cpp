// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_fabric_2d_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {
// Helper functions can be added here later
}  // namespace detail

ttnn::device_operation::CachedProgram<AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::shared_variables_t>
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_at_helper(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program{};
    return AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_at(
        operation_attributes, mesh_coordinate, tensor_args, tensor_return_value, program);
}

AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::cached_mesh_workload_t
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at_helper(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::shared_variables_t>
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    tt::tt_metal::Program& program) {
    return {
        std::move(program),
        create_at_program_processing(operation_attributes, mesh_coordinate, tensor_args, tensor_return_value, program)};
}

AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::shared_variables_t
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_at_program_processing(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    tt::tt_metal::Program& program) {
    // Return empty program as requested by the user
    // The actual implementation will be filled in later
    return shared_variables_t{};
}

void AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::override_runtime_arguments_per_program(
    const shared_variables_t& shared_variables,
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    AllGatherFabric2DDeviceOperation::tensor_return_value_t& tensor_return_value) {
    // Empty for now
}

void AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::override_runtime_arguments(
    cached_mesh_workload_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Empty for now
}

}  // namespace ttnn::operations::experimental::ccl
