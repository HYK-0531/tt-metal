// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_bug_report_rank_crta_pybind.hpp"

#include "moreh_bug_report_rank_crta.hpp"
#include "pybind11/cast.h"
#include "ttnn-pybind/decorators.hpp"
#include "ttnn/operations/moreh/moreh_bug_report_rank_crta/device/moreh_bug_report_rank_crta_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_bug_report_rank_crta {
void bind_moreh_bug_report_rank_crta_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_bug_report_rank_crta,
        "Moreh Bug Report Rank Crta Operation",
        ttnn::pybind_arguments_t{
            py::arg("input").noconvert(),
            py::arg("other").noconvert(),
            py::arg("output").noconvert(),
            py::kw_only(),
            py::arg("compute_kernel_config").noconvert() = std::nullopt});
}
}  // namespace ttnn::operations::moreh::moreh_bug_report_rank_crta
