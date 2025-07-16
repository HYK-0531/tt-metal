// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_bug_report_rank_crta {
struct MorehBugReportRankCrta {
    static Tensor invoke(
        const Tensor& input,
        const Tensor& other,
        const Tensor& output,
        std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config);
};
}  // namespace ttnn::operations::moreh::moreh_bug_report_rank_crta

namespace ttnn {
constexpr auto moreh_bug_report_rank_crta = ttnn::register_operation<
    "ttnn::moreh_bug_report_rank_crta",
    ttnn::operations::moreh::moreh_bug_report_rank_crta::MorehBugReportRankCrta>();
}
