
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report_rank_crta.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_bug_report_rank_crta/device/moreh_bug_report_rank_crta_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_bug_report_rank_crta {

Tensor MorehBugReportRankCrta::invoke(
    const Tensor& input,
    const Tensor& other,
    const Tensor& output,
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    return ttnn::prim::moreh_bug_report_rank_crta(input, other, output, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_bug_report_rank_crta
