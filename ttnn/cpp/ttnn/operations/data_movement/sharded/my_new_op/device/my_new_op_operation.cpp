// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/hal.hpp>

#include "my_new_op_operation.hpp"
#include "my_new_op_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void MyNewOpDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");

    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
    TT_FATAL(this->output_mem_config.buffer_type == BufferType::L1, "Error");
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            (*this->output_mem_config.shard_spec).shape[1] * input_tensor.element_size() % hal::get_l1_alignment() == 0,
            "Shard page size must currently have L1 aligned page size");
    }
    if (input_tensor.get_dtype() != this->output_dtype) {
        TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Error");
    }
}

std::vector<ttnn::TensorSpec> MyNewOpDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        input_tensor.get_logical_shape(),
        TensorLayout::fromPaddedShape(
            output_dtype,
            PageConfig(input_tensor.get_layout()),
            output_mem_config,
            input_tensor.get_logical_shape(),
            input_tensor.get_padded_shape()))};
}

operation::ProgramWithCallbacks MyNewOpDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return detail::my_new_op_multi_core(input_tensor, output_tensor, this->keep_l1_aligned);
}

}  // namespace ttnn::operations::data_movement
