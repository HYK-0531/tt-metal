// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <tt-logger/tt-logger.hpp>
#include <array>
#include <cstdint>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape_base.hpp>
#include "impl/context/metal_context.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/pool/generic/generic_pools.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"
#include "umd/device/types/cluster_descriptor_types.h"

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn {
namespace operations {
namespace binary {
namespace test {

class Conv2dOpIfTest : public ttnn::TTNNFixtureWithDevice {};
TEST_F(Conv2dOpIfTest, Conv2d) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 50176, 3},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const auto weight_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 1568, 64},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const auto output_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 12544, 64},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const uint32_t in_channels = 3;
    const uint32_t out_channels = 64;
    const uint32_t batch_size = 1;
    const uint32_t input_height = 224;
    const uint32_t input_width = 224;
    const std::array<uint32_t, 2> kernel_size{7, 7};
    const std::array<uint32_t, 2> stride{2, 2};
    const std::array<uint32_t, 2> padding{3, 3};
    const std::array<uint32_t, 2> dilation{1, 1};
    const uint32_t groups = 1;

    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (board_type != BoardType::N300 && board_type != BoardType::N150) {
        GTEST_SKIP();
    }

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        auto query = ttnn::graph::query_op_constraints(
            ttnn::conv2d,
            device,
            input_spec,
            weight_spec,
            device,
            in_channels,
            out_channels,
            batch_size,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            output_spec.tensor_layout().get_memory_config());

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        // Ensure some real usage is reported
        EXPECT_GT(query.resource_usage.cb_peak_size_per_core, 10000);
        EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 10000);
        ASSERT_TRUE(query.output_tensor_spec.has_value());
        EXPECT_EQ(query.output_tensor_spec.value(), output_spec);
    }
}

class MaxPool2dOpIfTest : public ttnn::TTNNFixtureWithDevice {};
TEST_F(MaxPool2dOpIfTest, MaxPool2d) {
    const auto input_spec = ttnn::TensorSpec(
        ttnn::Shape{1, 1, 100352, 64},
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            ttnn::DRAM_MEMORY_CONFIG));
    const uint32_t in_channels = 64;
    const uint32_t batch_size = 8;
    const uint32_t input_height = 112;
    const uint32_t input_width = 112;
    const std::array<uint32_t, 2> kernel_size{3, 3};
    const std::array<uint32_t, 2> stride{2, 2};
    const std::array<uint32_t, 2> padding{1, 1};
    const std::array<uint32_t, 2> dilation{1, 1};

    const BoardType board_type = tt::tt_metal::MetalContext::instance().get_cluster().get_board_type(0);
    if (board_type != BoardType::N300 && board_type != BoardType::N150) {
        GTEST_SKIP();
    }

    // Run the test
    {
        tt::tt_metal::IDevice* device = device_;
        auto query = ttnn::graph::query_op_constraints(
            ttnn::max_pool2d,
            device,
            input_spec,
            batch_size,
            in_channels,
            input_height,
            input_width,
            kernel_size,
            stride,
            padding,
            dilation,
            false,         // ceil_mode
            std::nullopt,  // memory_config,
            TensorMemoryLayout::HEIGHT_SHARDED);

        EXPECT_EQ(query.status, ttnn::graph::ExecutionStatus::Success);
        // Ensure some real usage is reported
        EXPECT_GT(query.resource_usage.l1_output_buffer_per_core, 10000);
        EXPECT_GT(query.resource_usage.cb_peak_size_per_core, 10000);
        EXPECT_GT(query.resource_usage.l1_buffers_peak_per_core, 10000);
    }
}

}  // namespace test
}  // namespace binary
}  // namespace operations
}  // namespace ttnn
