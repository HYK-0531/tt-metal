// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <pybind11/pybind11.h>

namespace ttnn::operations::experimental::reinterleave {
void bind_reinterleave_operation(pybind11::module& module);
}  // namespace ttnn::operations::experimental::reinterleave
