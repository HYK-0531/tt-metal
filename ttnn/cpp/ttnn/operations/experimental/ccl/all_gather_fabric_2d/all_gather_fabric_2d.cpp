// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "all_gather_fabric_2d.hpp"
#include "device/all_gather_fabric_2d_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::ccl {
namespace detail {}  // namespace detail

ttnn::Tensor ExecuteAllGatherFabric2D::invoke(
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    auto output_tensor = ttnn::Tensor(input_tensor);
    return output_tensor;
}

}  // namespace ttnn::operations::experimental::ccl
