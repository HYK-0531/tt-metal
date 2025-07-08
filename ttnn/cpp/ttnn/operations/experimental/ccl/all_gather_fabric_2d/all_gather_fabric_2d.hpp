// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

struct ExecuteAllGatherFabric2D {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        int32_t dim,
        uint32_t cluster_axis,
        const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
        uint32_t num_links,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring,
        std::optional<tt::tt_metal::SubDeviceId> subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::ccl

namespace experimental {
constexpr auto all_gather_fabric_2d = ttnn::register_operation<
    "ttnn::experimental::all_gather_fabric_2d",
    ttnn::operations::experimental::ccl::ExecuteAllGatherFabric2D>();
}  // namespace experimental

}  // namespace ttnn
