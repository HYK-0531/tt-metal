// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_device_operation.hpp"

namespace ttnn::operations::examples {

ExampleMultipleReturnDeviceOperation::program_factory_t ExampleMultipleReturnDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return SingleCore{};
}

void ExampleMultipleReturnDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(attributes, tensor_args);
}

void ExampleMultipleReturnDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
}

ExampleMultipleReturnDeviceOperation::spec_return_value_t ExampleMultipleReturnDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    return TensorSpec(
        output.logical_shape(),
        tt::tt_metal::TensorLayout(
            output.dtype(), tt::tt_metal::PageConfig(output.layout()), MemoryConfig{}));
}

ExampleMultipleReturnDeviceOperation::tensor_return_value_t ExampleMultipleReturnDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

std::tuple<
    ExampleMultipleReturnDeviceOperation::operation_attributes_t,
    ExampleMultipleReturnDeviceOperation::tensor_args_t>
ExampleMultipleReturnDeviceOperation::invoke(const Tensor& input, const Tensor& other, const Tensor& output) {
    return {operation_attributes_t{true, 42}, tensor_args_t{input, other, output}};
}

}  // namespace ttnn::operations::examples
