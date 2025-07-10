// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "emitc.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/graph/graph_query_op_constraints.hpp"
#include "ttnn/graph/graph_trace_utils.hpp"

static std::tuple<::ttnn::Tensor, ::ttnn::Tensor, ::ttnn::Tensor> createTensors(ttnn::distributed::MeshDevice* dev) {
    ::ttnn::Tensor actTensor = ttnn::ones(
        ::ttnn::Shape({1, 1, 102400, 80}),
        ::ttnn::DataType::BFLOAT16,
        ::ttnn::Layout::TILE,
        *dev,
        ::ttnn::MemoryConfig{
            ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
            ::ttnn::BufferType::L1,
            ::tt::tt_metal::ShardSpec{
                ::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                    ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}},
                    ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{5, 7}}}},
                ::std::array<uint32_t, 2>{1664, 96},
                ::ttnn::types::ShardOrientation::ROW_MAJOR,
                ::ttnn::types::ShardMode::PHYSICAL}});

    ::ttnn::Tensor weights = ttnn::ones(
        ::ttnn::Shape({1, 1, 864, 160}),
        ::ttnn::DataType::BFLOAT16,
        ::ttnn::Layout::TILE,
        *dev,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});

    ::ttnn::Tensor bias = ttnn::ones(
        ::ttnn::Shape({1, 1, 1, 160}),
        ::ttnn::DataType::BFLOAT16,
        ::ttnn::Layout::TILE,
        *dev,
        ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED, ::ttnn::BufferType::DRAM});

    return std::make_tuple(actTensor, weights, bias);
}

/*
        %207 = "ttnn.conv2d"(%206, %34, %arg5, %199) <{batch_size = 1 : i32,
                        conv2d_config = #ttnn.conv2d_config<dtype = bf16, weights_dtype =
                            bf16, activation = "", deallocate_activation = false,
                            reallocate_halo_output = true, act_block_h_override = 0, act_block_w_div =
                            1, reshard_if_not_optimal = false, override_sharding_config = false,
                            transpose_shards = true, output_layout = tile, enable_act_double_buffer =
                            false, enable_weights_double_buffer = false, enable_split_reader = false,
                            enable_subblock_padding = false>, dilation = array<i32: 1, 1>, groups = 1 :
                            i32, in_channels = 80 : i32, input_height = 320 : i32, input_width = 320 :
                            i32, kernel_size = array<i32: 3, 3>, out_channels = 160 : i32, padding =
                            array<i32: 1, 1, 1, 1>, stride = array<i32: 2, 2>}> :
                    (tensor<1x1x102400x80xbf16, #ttnn_layout126>, tensor<1x1x720x160xbf16,
                            #ttnn_layout27>, tensor<1x1x1x160xbf16, #ttnn_layout15>, !ttnn.device)
                    -> tensor<1x1x25600x160xbf16, #ttnn_layout127>
*/

// Calls directly conv2d on device.
static ::ttnn::Tensor conv_test() {
    ttnn::distributed::MeshDevice* dev = ttnn::DeviceGetter::getInstance();

    auto [actTensor, weights, bias] = createTensors(dev);

    ::ttnn::Tensor output = ::std::get<0>(ttnn::conv2d(
        actTensor,
        weights,
        dev,
        80,
        160,
        1,
        320,
        320,
        ::std::array<uint32_t, 2>{3, 3},
        ::std::array<uint32_t, 2>{2, 2},
        ::std::array<uint32_t, 4>{1, 1, 1, 1},
        ::std::array<uint32_t, 2>{1, 1},
        1,
        tt::tt_metal::DataType::BFLOAT16,
        bias,
        ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16},
        ::std::nullopt,
        ::ttnn::MemoryConfig{
            ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
            ::ttnn::BufferType::L1,
            ::tt::tt_metal::ShardSpec{
                ::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                    ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}},
                    ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{5, 7}}}},
                ::std::array<uint32_t, 2>{416, 160},
                ::ttnn::types::ShardOrientation::ROW_MAJOR,
                ::ttnn::types::ShardMode::PHYSICAL}}));
    return output;
}

// Mimics query_op_constraints.
static void call_graph_capture_api() {
    nlohmann::json op_trace;
    ttnn::Tensor output;
    ttnn::distributed::MeshDevice* dev = ttnn::DeviceGetter::getInstance();
    {
        auto capture_outer = ::ttnn::graph::ScopedGraphCapture(::ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);

        auto [actTensor, weights, bias] = createTensors(dev);

        // inner graph capture is to capture the actual op graph trace
        try {
            auto capture_inner = ttnn::graph::ScopedGraphCapture(ttnn::graph::GraphProcessor::RunMode::NO_DISPATCH);
            ::ttnn::Tensor outputTensor = ::std::get<0>(ttnn::conv2d(
                actTensor,
                weights,
                dev,
                80,
                160,
                1,
                320,
                320,
                ::std::array<uint32_t, 2>{3, 3},
                ::std::array<uint32_t, 2>{2, 2},
                ::std::array<uint32_t, 4>{1, 1, 1, 1},
                ::std::array<uint32_t, 2>{1, 1},
                1,
                tt::tt_metal::DataType::BFLOAT16,
                bias,
                ::ttnn::operations::conv::conv2d::Conv2dConfig{.weights_dtype = ::ttnn::DataType::BFLOAT16},
                ::std::nullopt,
                ::ttnn::MemoryConfig{
                    ::ttnn::TensorMemoryLayout::HEIGHT_SHARDED,
                    ::ttnn::BufferType::L1,
                    ::tt::tt_metal::ShardSpec{
                        ::ttnn::CoreRangeSet{::std::set<::ttnn::CoreRange>{
                            ::ttnn::CoreRange{::ttnn::CoreCoord{0, 0}, ::ttnn::CoreCoord{7, 6}},
                            ::ttnn::CoreRange{::ttnn::CoreCoord{0, 7}, ::ttnn::CoreCoord{5, 7}}}},
                        ::std::array<uint32_t, 2>{416, 160},
                        ::ttnn::types::ShardOrientation::ROW_MAJOR,
                        ::ttnn::types::ShardMode::PHYSICAL}}));

            output = ttnn::graph::detail::extract_output_tensor(outputTensor);

            op_trace = capture_inner.end_graph_capture();

        }  // end of inner graph capture
        catch (const std::exception& e) {
            log_debug(tt::LogOp, "Error during graph capture: {}", e.what());
            return;
        }
    }

    log_info(tt::LogOp, "Graph capture API passed");

    auto interleaved_storage_cores = dev->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    size_t cb_peak_size_per_core = ttnn::graph::extract_circular_buffers_peak_size_per_core(op_trace);
    size_t l1_buffers_peak_per_core =
        ttnn::graph::extract_l1_buffer_allocation_peak_size_per_core(op_trace, interleaved_storage_cores);
    size_t l1_output_buffer_per_core =
        output.buffer()->is_dram()
            ? 0
            : ttnn::graph::extract_l1_output_buffer_allocation_size_per_core(output, interleaved_storage_cores);

    log_info(tt::LogOp, "cb_peak_size_per_core: {}", cb_peak_size_per_core);
    log_info(tt::LogOp, "l1_buffers_peak_per_core: {}", l1_buffers_peak_per_core);
    log_info(tt::LogOp, "l1_output_buffer_per_core: {}", l1_output_buffer_per_core);
}

TEST(EmitC, Conv2D) {
    call_graph_capture_api();
    conv_test();
}
