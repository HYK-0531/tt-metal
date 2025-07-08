// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_fabric_2d_pybind.hpp"

#include <cstdint>
#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn-pybind/decorators.hpp"
#include "all_gather_fabric_2d.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::operations::experimental::ccl {

void py_bind_all_gather_fabric_2d(py::module& module) {
    auto doc =
        R"doc(all_gather_fabric_2d(input_tensor: ttnn.Tensor, dim: int, cluster_axis: int, multi_device_global_semaphore: Union[GlobalSemaphore, List[GlobalSemaphore]], num_links: int = 1, memory_config: Optional[MemoryConfig] = None, topology: ttnn.ccl.Topology = ttnn.ccl.Topology.Ring, subdevice_id: Optional[SubDeviceId] = None) -> ttnn.Tensor

            All-gather operation optimized for fabric 2D topology.
            This operation performs all-gather along a specified dimension within a 2D mesh of devices,
            operating on the specified cluster axis of the mesh.

            Args:
                input_tensor (ttnn.Tensor): the input tensor.
                dim (int): the dimension (tensor axis) to gather along for concatenation.
                cluster_axis (int): the axis of the 2D mesh of devices to operate on (0 for rows, 1 for columns).
                multi_device_global_semaphore (Union[GlobalSemaphore, List[GlobalSemaphore]]): the global semaphore for cross-device synchronization.
                num_links (int, optional): the number of links. Defaults to `1`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.
                topology (ttnn.ccl.Topology, optional): The topology configuration. Defaults to `ttnn.ccl.Topology.Ring`.
                subdevice_id (ttnn.SubDeviceId, optional): the subdevice id. Defaults to `None`.

           Returns:
               ttnn.Tensor: the output tensor.

            Example:

                >>> tensor = ttnn.experimental.all_gather_fabric_2d(
                                input_tensor,
                                dim=0,
                                cluster_axis=1,
                                multi_device_global_semaphore=semaphore,
                                num_links=1,
                                memory_config=memory_config,
                                topology=ttnn.ccl.Topology.Ring,
                                subdevice_id=subdevice_id)doc";

    using OperationType = decltype(ttnn::experimental::all_gather_fabric_2d);
    ttnn::bind_registered_operation(
        module,
        ttnn::experimental::all_gather_fabric_2d,
        doc,
        ttnn::pybind_overload_t{
            [](const OperationType& self,
               const ttnn::Tensor& input_tensor,
               int32_t dim,
               uint32_t cluster_axis,
               const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
               uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               ttnn::ccl::Topology topology,
               std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
                return self(
                    input_tensor,
                    dim,
                    cluster_axis,
                    multi_device_global_semaphore,
                    num_links,
                    memory_config,
                    topology,
                    subdevice_id);
            },
            py::arg("input_tensor").noconvert(),
            py::arg("dim"),
            py::arg("cluster_axis"),
            py::arg("multi_device_global_semaphore"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("topology") = ttnn::ccl::Topology::Ring,
            py::arg("subdevice_id") = std::nullopt});
}

}  // namespace ttnn::operations::experimental::ccl
