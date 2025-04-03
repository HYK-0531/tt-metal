// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <array>
#include <optional>
#include <variant>

#include "ttnn/common/queue_id.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
enum class DataType;
enum class ShardOrientation;
enum class TensorMemoryLayout;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations::data_movement {

struct InterleavedToShardedPartialOperation {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const std::variant<CoreCoord, CoreRangeSet>& grid,
        const std::array<uint32_t, 2>& shard_shape,
        int64_t& num_slices,
        int64_t& slice_index,
        tt::tt_metal::TensorMemoryLayout shard_scheme,
        tt::tt_metal::ShardOrientation shard_orientation,
        const std::optional<DataType>& data_type_arg);
};

}  // namespace operations::data_movement

constexpr auto interleaved_to_sharded_partial = ttnn::register_operation_with_auto_launch_op<
    "ttnn::interleaved_to_sharded_partial",
    ttnn::operations::data_movement::InterleavedToShardedPartialOperation>();
}  // namespace ttnn
