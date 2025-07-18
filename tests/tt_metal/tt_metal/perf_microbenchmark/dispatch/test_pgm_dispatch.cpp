#include <benchmark/benchmark.h>
#include <chrono>
#include <fmt/base.h>
#include <stdint.h>
#include <tt-metalium/command_queue.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>
#include <array>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/distributed.hpp>
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include "impl/context/metal_context.hpp"
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include "umd/device/types/xy_pair.h"
#include <tt-metalium/math.hpp>
#include "tt_metal/impl/dispatch/device_command.hpp"

constexpr uint32_t DEFAULT_ITERATIONS = 10000;
constexpr uint32_t DEFAULT_WARMUP_ITERATIONS = 2000;
constexpr uint32_t MIN_KERNEL_SIZE_BYTES = 32;
constexpr uint32_t DEFAULT_KERNEL_SIZE_K = 1;

//////////////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////////////////
using std::vector;
using namespace tt;
using namespace tt::tt_metal::distributed;

struct TestInfo {
    uint32_t iterations = DEFAULT_ITERATIONS;
    uint32_t warmup_iterations = DEFAULT_WARMUP_ITERATIONS;
    CoreRange workers = {{0, 0}, {0, 0}};
    uint32_t kernel_size{DEFAULT_KERNEL_SIZE_K * 1024};
    uint32_t kernel_cycles{0};
    uint32_t n_common_args{0};
    bool brisc_enabled{true};
    bool ncrisc_enabled{true};
    bool trisc_enabled{true};
    bool dispatch_from_eth{false};
    bool use_all_cores{false};
};

std::tuple<uint32_t, uint32_t> get_core_count() {
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    std::string arch_name = tt::tt_metal::hal::get_arch_name();
    if (arch_name == "grayskull") {
        core_x = 11;
        core_y = 8;
    } else if (arch_name == "wormhole_b0") {
        core_x = 7;
        core_y = 6;
    } else if (arch_name == "blackhole") {
        core_x = 12;
        core_y = 9;
    } else {
        log_fatal(tt::LogTest, "Unexpected ARCH_NAME {}", arch_name);
        exit(0);
    }
    return std::make_tuple(core_x, core_y);
}

bool initialize_program(
    const TestInfo& info, std::shared_ptr<MeshDevice> mesh_device, tt_metal::Program& program, uint32_t run_cycles) {
    program = tt_metal::CreateProgram();

    std::map<std::string, std::string> defines = {{"KERNEL_BYTES", std::to_string(info.kernel_size)}};
    if (run_cycles != 0) {
        defines.insert(std::pair<std::string, std::string>("KERNEL_RUN_TIME", std::to_string(run_cycles)));
    }

    std::vector<uint32_t> common_args(info.n_common_args);
    if (info.brisc_enabled) {
        auto kernel_id = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            info.workers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        tt_metal::SetCommonRuntimeArgs(program, kernel_id, common_args);
    }
    if (info.ncrisc_enabled) {
        auto kernel_id = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            info.workers,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_1_default,
                .defines = defines});
        tt_metal::SetCommonRuntimeArgs(program, kernel_id, common_args);
    }
    if (info.trisc_enabled) {
        auto kernel_id = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/dispatch/kernels/pgm_dispatch_perf.cpp",
            info.workers,
            tt_metal::ComputeConfig{.defines = defines});
        tt_metal::SetCommonRuntimeArgs(program, kernel_id, common_args);
    }
    return true;
}

struct ProgramExecutor {
    std::function<void()> execute_program;
    std::function<void()> warmup_program;
    uint32_t total_program_iterations;
    ProgramExecutor(std::function<void()> exec, std::function<void()> warm, uint32_t total_iters) :
        execute_program(exec), warmup_program(warm), total_program_iterations(total_iters) {}
};

ProgramExecutor create_standard_executor(
    const TestInfo& info, MeshWorkload& mesh_workload, tt_metal::Program& program, MeshCommandQueue& mesh_cq) {
    AddProgramToMeshWorkload(
        mesh_workload, std::move(program), MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0)));
    std::function warmup_func{[&info, &mesh_cq, &mesh_workload]() {
        for (int i = 0; i < info.warmup_iterations; i++) {
            EnqueueMeshWorkload(mesh_cq, mesh_workload, false);
        }
    }};
    std::function execute_func{[&info, &mesh_cq, &mesh_workload]() {
        for (int i = 0; i < info.iterations; i++) {
            EnqueueMeshWorkload(mesh_cq, mesh_workload, false);
        }
    }};
    return ProgramExecutor(execute_func, warmup_func, info.iterations);
}

template <typename T>
void run_benchmark_timing_loop(
    T& state,
    const TestInfo& info,
    MeshCommandQueue& mesh_cq,
    ProgramExecutor& executor,
    std::shared_ptr<MeshDevice> mesh_device) {
    constexpr std::size_t cq_id = 0;
    auto execute_func = executor.execute_program;
    for (auto _ : state) {
        auto start = std::chrono::system_clock::now();
        execute_func();
        Finish(mesh_cq);
        auto end = std::chrono::system_clock::now();

        if constexpr (std::is_same_v<T, benchmark::State>) {
            auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
            state.SetIterationTime(elapsed_seconds.count());
        } else {
            std::chrono::duration<double> elapsed_seconds = (end - start);
            log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);
            log_info(
                LogTest,
                "Ran in {}us per iteration",
                elapsed_seconds.count() * 1000 * 1000 / executor.total_program_iterations);
        }
    }
}

template <typename T>
void set_benchmark_counters(
    T& state,
    const TestInfo& info,
    tt_metal::IDevice* device,
    uint32_t total_iterations,
    const std::unordered_map<std::string, uint32_t>& extra_counters = {}) {
    if constexpr (std::is_same_v<T, benchmark::State>) {
        state.counters["IterationTime"] = benchmark::Counter(
            total_iterations, benchmark::Counter::kIsIterationInvariantRate | benchmark::Counter::kInvert);
        state.counters["Clock"] = benchmark::Counter(get_tt_npu_clock(device), benchmark::Counter::kDefaults);
    }
}

CoreType dispatch_core_type_to_core_type(DispatchCoreType dispatch_core_type) {
    switch (dispatch_core_type) {
        case DispatchCoreType::WORKER: return CoreType::WORKER;
        case DispatchCoreType::ETH: return CoreType::ETH;
        default: TT_THROW("invalid dispatch core type");
    }
}

tt_metal::Program create_standard_program(
    const TestInfo& info, std::shared_ptr<MeshDevice> mesh_device, DispatchCoreType dispatch_core_type) {
    tt_metal::Program program;
    if (!initialize_program(info, mesh_device, program, info.kernel_cycles)) {
        throw std::runtime_error("Standard program creation failed");
    }
    return program;
}

template <typename T>
static int pgm_dispatch(T& state, TestInfo info) {
    if constexpr (std::is_same_v<T, benchmark::State>) {
        log_info(LogTest, "Running {}", state.name());
    }

    if (info.use_all_cores) {
        auto core_count = get_core_count();
        info.workers = CoreRange({0, 0}, {std::get<0>(core_count), std::get<1>(core_count)});
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(true);

    bool pass = true;
    std::shared_ptr<MeshDevice> mesh_device;
    try {
        const chip_id_t device_id = 0;
        const std::size_t cq_id = 0;
        DispatchCoreType dispatch_core_type = info.dispatch_from_eth ? DispatchCoreType::ETH : DispatchCoreType::WORKER;
        mesh_device = MeshDevice::create_unit_mesh(
            device_id, DEFAULT_L1_SMALL_SIZE, 900 * 1024 * 1024, 1, DispatchCoreConfig{dispatch_core_type});
        auto& mesh_cq = mesh_device->mesh_command_queue(cq_id);
        ProgramExecutor executor([]() {}, []() {}, 0);
        MeshWorkload mesh_workload;
        auto program = create_standard_program(info, mesh_device, dispatch_core_type);
        executor = create_standard_executor(info, mesh_workload, program, mesh_cq);
        set_benchmark_counters(state, info, mesh_device->get_device(0, 0), executor.total_program_iterations);
        executor.warmup_program();
        run_benchmark_timing_loop(state, info, mesh_cq, executor, mesh_device);
        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        pass = false;
        mesh_device->close();
        log_fatal(tt::LogTest, "{}", e.what());
    }

    tt::tt_metal::MetalContext::instance().rtoptions().set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        if constexpr (std::is_same_v<T, benchmark::State>) {
            state.SkipWithError("Test failed");
        } else {
            log_info(LogTest, "Test failed");
        }
        return 1;
    }
}

static void BM_pgm_dispatch(benchmark::State& state, TestInfo info) {
    info.n_common_args = state.range(0);
    info.kernel_cycles = state.range(1);
    pgm_dispatch(state, info);
}

static void SweepKernelCyclesAndCommonArgs(benchmark::internal::Benchmark* b) {
    for (int n_common_args = 0; n_common_args <= 256; n_common_args += 32) {
        for (int kernel_cycles = 0; kernel_cycles <= 1024; kernel_cycles += 64) {
            b->Args({n_common_args, kernel_cycles});
        }
    }
}

BENCHMARK_CAPTURE(BM_pgm_dispatch, n_common_args__kernel_cycles, TestInfo{.use_all_cores = true})
    ->Apply(SweepKernelCyclesAndCommonArgs)
    ->UseManualTime();

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
