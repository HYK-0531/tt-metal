// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <fmt/base.h>
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <stdint.h>
#include <sys/types.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <bit>
#include <cctype>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::compute::tilize_perf {

struct TestConfig {
    bool fast_tilize = false;
    bool fp32_dest_acc_en = false;
    tt_metal::DataFormat input_data_format = tt_metal::DataFormat::Float16_b;
    tt_metal::DataFormat output_data_format = tt_metal::DataFormat::Float16_b;
    uint32_t num_tiles_c;
};

void run_single_core_tilize_program(tt_metal::IDevice* device, const TestConfig& test_config) {
    Program program = tt::tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t num_tiles = test_config.num_tiles_c;
    uint32_t input_single_tile_size =
        test_config.input_data_format == tt_metal::DataFormat::Float16_b ? 2 * 1024 : 4 * 1024;
    uint32_t output_single_tile_size = test_config.output_data_format == tt_metal::DataFormat::Float16_b ? 1024 * 2
                                       : test_config.output_data_format == tt_metal::DataFormat::Float32 ? 1024 * 4
                                                                                                         : 1088;
    uint32_t input_buffer_size = input_single_tile_size * num_tiles;
    uint32_t output_buffer_size = output_single_tile_size * num_tiles;

    tt_metal::InterleavedBufferConfig input_dram_config{
        .device = device,
        .size = input_buffer_size,
        .page_size = input_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig output_dram_config{
        .device = device,
        .size = output_buffer_size,
        .page_size = output_buffer_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt_metal::Buffer> src0_dram_buffer = CreateBuffer(input_dram_config);
    uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(output_dram_config);
    uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(input_buffer_size, {{src0_cb_index, test_config.input_data_format}})
            .set_page_size(src0_cb_index, input_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(output_buffer_size, {{ouput_cb_index, test_config.output_data_format}})
            .set_page_size(ouput_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_push_n.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

    auto unary_writer_kernel = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    vector<uint32_t> compute_kernel_args = {uint(test_config.num_tiles_c)};

    std::map<std::string, std::string> defines = {};

    if (test_config.fp32_dest_acc_en) {
        defines["DST_ACCUM_MODE"] = "1";
    }
    if (test_config.fast_tilize) {
        defines["FAST_TILIZE"] = "1";
    }

    auto eltwise_unary_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/compute/tilize_perf.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en, .compile_args = compute_kernel_args, .defines = defines});

    std::vector<uint32_t> src0_vec = std::vector<uint32_t>(input_buffer_size / sizeof(uint32_t));

    tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        core,
        {dram_buffer_src0_addr,
         (uint32_t)0,  // dram bank id
         num_tiles,
         src0_cb_index,
         test_config.num_tiles_c,
         false});

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel, core, {dram_buffer_dst_addr, (uint32_t)0, num_tiles});

    tt_metal::detail::LaunchProgram(device, program);

    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    log_info(
        tt::LogTest,
        "Done running test with: num_tiles_c = {}, InputDF = {}, OutputDF = {}, FP32_DestAcc = {}, FastTilize = {}",
        test_config.num_tiles_c,
        test_config.input_data_format,
        test_config.output_data_format,
        test_config.fp32_dest_acc_en,
        test_config.fast_tilize);
}

}  // namespace unit_tests::compute::tilize_perf

constexpr bool fast_tilize = false;

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32FP32) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Float32,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32FP16) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Float16_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32BFP8) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Bfp8_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16FP32) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Float32,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16FP16) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Float16_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16BFP8) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = false,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Bfp8_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32FP32D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Float32,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32FP16D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Float16_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP32BFP8D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float32,
            .output_data_format = tt_metal::DataFormat::Bfp8_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16FP32D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Float32,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16FP16D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Float16_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

TEST_F(DeviceFixture, TensixComputeUnpackTilizeBenchmarkFP16BFP8D) {
    for (int i = 1; i <= 128; i += 1) {
        unit_tests::compute::tilize_perf::TestConfig test_config = {
            .fast_tilize = fast_tilize,
            .fp32_dest_acc_en = true,
            .input_data_format = tt_metal::DataFormat::Float16_b,
            .output_data_format = tt_metal::DataFormat::Bfp8_b,
            .num_tiles_c = i};
        unit_tests::compute::tilize_perf::run_single_core_tilize_program(this->devices_.at(0), test_config);
    }
}

}  // namespace tt::tt_metal
