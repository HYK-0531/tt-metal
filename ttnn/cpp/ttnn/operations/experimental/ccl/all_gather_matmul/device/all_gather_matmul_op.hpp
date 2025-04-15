// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

#include "ttnn/run_operation.hpp"

#include <optional>
#include <vector>
#include <algorithm>

/* Fusion includes */
#include "cpp/ttnn/operations/ccl/all_gather/device/all_gather_op.hpp"
#include "cpp/ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn {
namespace experimental {

struct AllGatherMatmul {
    struct operation_attributes_t {
        /* All Gather Params */
        const ttnn::AllGather all_gather_struct;

        /* Matmul Params */
        const operations::matmul::MatmulArgs matmul_struct;

        /* Fusion Params */
        const CoreCoord all_gather_core_grid_offset;
    };

    struct tensor_args_t {
        Tensor input_tensor;
        Tensor weight_tensor;
        Tensor all_gather_output;
        Tensor datacopy_output;
    };

    using spec_return_value_t = std::vector<TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    static void validate(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::ProgramDescriptor create_program(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const Tensor& weight_tensor,
        const uint32_t dim,
        const CoreCoord all_gather_core_grid_offset,
        const uint32_t num_links = 1,
        const std::optional<MemoryConfig>& memory_config_ag = std::nullopt,
        const std::optional<size_t> user_defined_num_workers = std::nullopt,
        const std::optional<size_t> user_defined_num_buffers_per_channel = std::nullopt,
        const std::optional<MemoryConfig>& memory_config_mm = std::nullopt,
        const bool transpose_a = false,
        const bool transpose_b = false,
        const std::optional<const DataType> dtype = std::nullopt,
        const std::optional<const operations::matmul::MatmulProgramConfig>& program_config = std::nullopt,
        const std::optional<const std::string>& activation = std::nullopt,
        const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
        const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt);
};

tt::tt_metal::ProgramDescriptor all_gather_matmul_multi_core_with_workers(

    /* General Params */
    const Tensor& input_tensor,
    Tensor& all_gather_output_tensor,
    Tensor& datacopy_output_tensor,
    const Tensor& weight_tensor,
    Tensor& matmul_output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    const std::optional<size_t> user_defined_num_workers,
    const std::optional<size_t> user_defined_num_buffers_per_channel,
    const std::optional<chip_id_t> receiver_device_id,
    const std::optional<chip_id_t> sender_device_id,
    ttnn::ccl::Topology topology,
    const CoreCoord core_grid_offset,

    /* Matmul Params */
    const std::optional<const Tensor> bias,
    bool bcast_batch,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    bool untilize_out);
}  // namespace experimental
}  // namespace ttnn

namespace ttnn::prim::experimental::ccl {
constexpr auto all_gather_matmul =
    ttnn::register_operation<"ttnn::prim::experimental::ccl::all_gather_matmul", ttnn::experimental::AllGatherMatmul>();
}  // namespace ttnn::prim::experimental::ccl
