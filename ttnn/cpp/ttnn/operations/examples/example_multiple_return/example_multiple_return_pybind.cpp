// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/examples/example_multiple_return/example_multiple_return_pybind.hpp"

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/examples/example_multiple_return/example_multiple_return.hpp"

namespace py = pybind11;

namespace ttnn::operations::examples {

void bind_example_multiple_return_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::prim::example_multiple_return,
        R"doc(example_multiple_return(input: ttnn.Tensor, other: ttnn.Tensor)doc",
        ttnn::pybind_arguments_t{
            py::arg("input"),
            py::arg("other"),
            py::arg("output"),
        });
}

}  // namespace ttnn::operations::examples
