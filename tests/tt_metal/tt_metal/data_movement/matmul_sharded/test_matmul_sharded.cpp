// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device_fixture.hpp"
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "dm_common.hpp"

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::matmul_sharded {

constexpr uint32_t START_ID = 500;

// Test config, i.e. test parameters
struct MatmulShardedConfig {
    uint32_t test_id = 0;
};

/// @brief Does L1 Sender Core --> L1 Receiver Core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool run_dm(IDevice* device, const MatmulShardedConfig& test_config) {
    // Program
    Program program = CreateProgram();

    CoreCoord origin_core(0, 0);
    CoreCoord start_core(0, 0);
    CoreCoord end_core(3, 3);

    CoreRangeSet matmul_cores({CoreRange(start_core, end_core)});

    CoreCoord origin_core_worker = device->worker_core_from_logical_core(origin_core);
    CoreCoord start_core_worker = device->worker_core_from_logical_core(start_core);
    CoreCoord end_core_worker = device->worker_core_from_logical_core(end_core);

    std::vector<uint32_t> risc0_compile_args = {
        origin_core_worker.x,  // Origin X coordinate
        origin_core_worker.y,  // Origin Y coordinate
        start_core_worker.x,   // Start X coordinate
        start_core_worker.y,   // Start Y coordinate
        end_core_worker.x,     // End X coordinate
        end_core_worker.y,     // End Y coordinate
        0,                     // RISC core ID
        test_config.test_id,   // Test ID
    };

    std::vector<uint32_t> risc1_compile_args = {
        origin_core_worker.x,  // Origin X coordinate
        origin_core_worker.y,  // Origin Y coordinate
        start_core_worker.x,   // Start X coordinate
        start_core_worker.y,   // Start Y coordinate
        end_core_worker.x,     // End X coordinate
        end_core_worker.y,     // End Y coordinate
        1,                     // RISC core ID
        test_config.test_id,   // Test ID
    };

    // Kernels
    auto risc0_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul_sharded/kernels/2d_matmul_in0_sharded.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = risc0_compile_args});

    auto risc1_kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/data_movement/matmul_sharded/kernels/2d_matmul_in0_sharded.cpp",
        matmul_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = risc1_compile_args});

    // Assign unique id
    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", test_config.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Launch program
    MetalContext::instance().get_cluster().l1_barrier(device->id());
    detail::LaunchProgram(device, program);

    return true;
}
}  // namespace unit_tests::dm::matmul_sharded

TEST_F(DeviceFixture, TensixDataMovementMatmulSharded) {
    // Parameters
    uint32_t test_id = unit_tests::dm::matmul_sharded::START_ID + 0;

    // Test config
    unit_tests::dm::matmul_sharded::MatmulShardedConfig test_config = {.test_id = test_id};

    // Run
    for (unsigned int id = 0; id < num_devices_; id++) {
        EXPECT_TRUE(run_dm(devices_.at(id), test_config));
    }
}

}  // namespace tt::tt_metal
