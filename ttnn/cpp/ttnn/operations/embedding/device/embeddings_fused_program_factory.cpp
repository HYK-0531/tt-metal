// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_fused_program_factory.hpp"
#include "embedding_common.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/embedding/device/embedding_device_operation.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include <tracy/Tracy.hpp>

using namespace tt;
using namespace tt::tt_metal;

namespace ttnn::operations::embedding::detail {

tt::tt_metal::operation::ProgramWithCallbacks embeddings_fused(
    const Tensor& a,
    const Tensor& weights,
    Tensor& output,
    EmbeddingsType embeddings_type,
    std::optional<uint32_t> pad_token) {
    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::Buffer* a_buffer = a.buffer();
    tt_metal::Buffer* weights_buffer = weights.buffer();
    tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    IDevice* device = a.device();
    auto dst_addr = output.buffer()->address();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool in0_is_dram = a.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool weights_is_dram = weights.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = output.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.padded_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t num_embeddings = weights.padded_shape()[-2];

    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_output_rows_per_batch = a.padded_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    // Note: num_blocks is just blocks along height
    uint32_t num_blocks = num_output_rows / TILE_HEIGHT;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch / TILE_HEIGHT;
    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2, num_tiles_per_block;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_cores = all_cores.num_cores();
        num_blocks_per_core_group_1 = shard_spec.shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_tiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
        num_tiles_per_block = weights.padded_shape()[-1] / TILE_WIDTH;
        row_major = false;
    }
    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.dtype());
    uint32_t weights_single_tile_size = tt_metal::detail::TileSize(weights_cb_data_format);
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    // Hardcoded limit to reduce L1 usage. Should be updated to be tuned based on overall L1 usage
    constexpr uint32_t max_double_buffer_tiles = 64;
    uint32_t buffering = num_tiles_per_block > max_double_buffer_tiles ? 1 : 2;

    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            buffering * num_tiles_per_block * weights_single_tile_size, {{src0_cb_index, weights_cb_data_format}})
            .set_page_size(src0_cb_index, weights_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(TILE_HEIGHT * input_element_size_bytes, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, TILE_HEIGHT * input_element_size_bytes);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t output_cb_index = CBIndex::c_2;
    uint32_t output_cb_size;
    if (output_sharded) {
        output_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        output_cb_size = buffering * num_tiles_per_block * output_single_tile_size;
    }
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(output_cb_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    if (output_sharded) {
        cb_output_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    constexpr uint32_t src2_cb_index = CBIndex::c_3;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt_metal::CircularBufferConfig cb_src2_config =
            tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        auto cb_src2 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }
    uint32_t weight_block_size;
    if (output_sharded) {
        weight_block_size = output.shard_spec().value().shape[1] * weights_element_size_bytes;
    } else {
        weight_block_size = weight_page_size;
    }

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;
    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)src0_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)in0_is_dram,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weights_is_dram,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)weight_block_size,
        (std::uint32_t)num_tiles_per_block,
        (std::uint32_t)input_block_size_bytes};

    std::map<std::string, std::string> embedding_defines = {
        {magic_enum::enum_name(embeddings_type).data(), "1"},
        {magic_enum::enum_name(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    if (num_blocks_per_core_group_1 > 0) {
        std::vector<uint32_t> compute_args_1 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_1),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_1 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp",
            core_group_1,
            tt_metal::ComputeConfig{.compile_args = compute_args_1});
    }

    if (num_blocks_per_core_group_2 > 0) {
        std::vector<uint32_t> compute_args_2 = {
            uint32_t(src0_cb_index),                // input embeddings_cb_index
            uint32_t(output_cb_index),              // output_cb_index
            uint32_t(num_blocks_per_core_group_2),  // per_core_block_cnt
            uint32_t(num_tiles_per_block)           // per_core_block_tile_cnt
        };
        auto tilize_kernel_id_2 = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.compile_args = compute_args_2});
    }
    KernelHandle writer_kernel_id = 0;
    // TODO: We can use the second risc to do more work in parallel
    if (!output_sharded) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)out_is_dram};

        // Tilized writer
        writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            all_cores,
            tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }

    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)0, (std::uint32_t)0};

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            reader_runtime_args[2] = input_offset / num_blocks_per_batch;
            reader_runtime_args[3] = input_offset % num_blocks_per_batch * input_block_size_bytes;
            reader_runtime_args[4] = weight_offset;
            reader_runtime_args[5] = local_num_blocks;
            tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        if (!output_sharded) {
            writer_runtime_args[1] = num_tiles_per_block * local_num_blocks;
            writer_runtime_args[2] = tile_offset;
            tile_offset += local_num_blocks * num_tiles_per_block;
            tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
            input_offset += local_num_blocks;
        } else {
            weight_offset += weight_block_size;
            if (weight_offset == weight_page_size) {
                weight_offset = 0;
                input_offset += local_num_blocks;
            }
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, cores, cb_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto output_buffer = output_tensors.at(0).buffer();
            auto output_buffer_address = output_buffer->address();
            auto input_buffer_address = input_tensors.at(0).buffer()->address();
            auto weights_buffer_address = input_tensors.at(1).buffer()->address();

            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);
            const bool output_sharded = is_sharded(output_buffer->buffer_layout());
            if (output_sharded) {
                UpdateDynamicCircularBufferAddress(program, cb_output, *output_buffer);
            }

            for (const auto& core : cores) {
                {
                    auto& runtime_args = reader_runtime_args[core.x][core.y];
                    runtime_args[0] = input_buffer_address;
                    runtime_args[1] = weights_buffer_address;
                }

                if (!output_sharded) {
                    auto& runtime_args = writer_runtime_args[core.x][core.y];
                    runtime_args[0] = output_buffer_address;
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::embedding::detail
