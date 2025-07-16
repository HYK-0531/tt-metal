
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report_rank_crta_device_operation.hpp"

#include <cstdint>

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_bug_report_rank_crta {

void MorehBugReportRankCrtaOperation::validate_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {}

MorehBugReportRankCrtaOperation::program_factory_t MorehBugReportRankCrtaOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory();
}

void MorehBugReportRankCrtaOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

void MorehBugReportRankCrtaOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_tensors(operation_attributes, tensor_args);
};

MorehBugReportRankCrtaOperation::spec_return_value_t MorehBugReportRankCrtaOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_THROW("output tensor must have value");
}

MorehBugReportRankCrtaOperation::tensor_return_value_t MorehBugReportRankCrtaOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.output;
}

std::tuple<MorehBugReportRankCrtaOperation::operation_attributes_t, MorehBugReportRankCrtaOperation::tensor_args_t>
MorehBugReportRankCrtaOperation::invoke(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    return {
        MorehBugReportRankCrtaOperation::operation_attributes_t{
            init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4)},
        MorehBugReportRankCrtaOperation::tensor_args_t{input, other, output}};
}
}  // namespace ttnn::operations::moreh::moreh_bug_report_rank_crta
