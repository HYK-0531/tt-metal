
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "moreh_bug_report_rank_crta_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <ttnn/tensor/tensor_accessor_args.hpp>

namespace ttnn::operations::moreh::moreh_bug_report_rank_crta {
MorehBugReportRankCrtaOperation::ProgramFactory::cached_program_t
MorehBugReportRankCrtaOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    auto& input = tensor_args.input;
    auto& other = tensor_args.other;
    auto& output = tensor_args.output;
    auto compute_kernel_config = operation_attributes.compute_kernel_config;

    auto input_dtype = input.dtype();
    auto other_dtype = other.dtype();
    auto output_dtype = output.dtype();

    auto shape = input.logical_shape();
    Program program{};

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::IDevice* device = input.device();
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y{grid.y};
    uint32_t units_to_divide = input.volume() / tt::constants::TILE_HW;
    uint32_t core_w = grid.x;
    uint32_t core_h = grid.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, units_to_divide);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    auto other_data_format = tt::tt_metal::datatype_to_dataformat_converter(other.dtype());
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        output_data_format,
        {
            {tt::CBIndex::c_0, 2, input_data_format},  // input
            {tt::CBIndex::c_1, 2, other_data_format},  // other
            {tt::CBIndex::c_2, 2},                     // output
        });

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels defines
    ////////////////////////////////////////////////////////////////////////////
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    std::map<std::string, std::string> compute_defines;

    if (fp32_dest_acc_en) {
        reader_defines["FP32_DEST_ACC_EN"] = "1";
        writer_defines["FP32_DEST_ACC_EN"] = "1";
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////

    const auto input_accessor_args = TensorAccessorArgs(*input.buffer());
    const auto other_accessor_args = TensorAccessorArgs(*other.buffer());
    const auto output_accessor_args = TensorAccessorArgs(*output.buffer());
    std::vector<uint32_t> input_compile_time_args = input_accessor_args.get_compile_time_args();
    std::vector<uint32_t> other_compile_time_args = other_accessor_args.get_compile_time_args();
    std::vector<uint32_t> output_compile_time_args = output_accessor_args.get_compile_time_args();

    std::vector<uint32_t> reader_compile_time_args{
        static_cast<uint32_t>(is_dram(input)),
        static_cast<uint32_t>(is_dram(other)),
    };
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), input_compile_time_args.begin(), input_compile_time_args.end());
    reader_compile_time_args.insert(
        reader_compile_time_args.end(), other_compile_time_args.begin(), other_compile_time_args.end());

    std::vector<uint32_t> writer_compile_time_args{static_cast<uint32_t>(is_dram(output))};
    writer_compile_time_args.insert(
        writer_compile_time_args.end(), output_compile_time_args.begin(), output_compile_time_args.end());

    const auto reader_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_bug_report_rank_crta/device/kernels/"
        "reader_moreh_bug_report_rank_crta.cpp";
    const auto writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_bug_report_rank_crta/device/kernels/"
        "writer_moreh_bug_report_rank_crta.cpp";

    const auto reader_kernel_id =
        CreateReadKernel(program, reader_kernel_file, all_cores, reader_compile_time_args, reader_defines);
    const auto writer_kernel_id =
        CreateWriteKernel(program, writer_kernel_file, all_cores, writer_compile_time_args, writer_defines);

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // const std::vector<uint32_t> compute_args_group_1{num_tiles_per_core_group_1};

    const auto compute_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_bug_report_rank_crta/device/kernels/"
        "moreh_bug_report_rank_crta.cpp";

    auto compute_kernel_id = CreateComputeKernel(
        program,
        compute_kernel_file,
        {
            {core_group_1, num_tiles_per_core_group_1, {num_tiles_per_core_group_1}},
            {core_group_2, num_tiles_per_core_group_2, {num_tiles_per_core_group_2}},
        },
        compute_defines,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode);

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core{0};
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {
                num_tiles_read,
                num_tiles_per_core,
            });
        SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {
                num_tiles_read,
                num_tiles_per_core,
            });

        num_tiles_read += num_tiles_per_core;
    }

    std::vector<uint32_t> input_runtime_args = input_accessor_args.get_common_runtime_args();
    std::vector<uint32_t> other_runtime_args = other_accessor_args.get_common_runtime_args();
    std::vector<uint32_t> output_runtime_args = output_accessor_args.get_common_runtime_args();

    const auto input_addr = input.buffer()->address();
    const auto other_addr = other.buffer()->address();
    const auto output_addr = output.buffer()->address();

    std::vector<uint32_t> reader_crtas = {
        input.buffer()->address(),
        other.buffer()->address(),
    };
    std::vector<uint32_t> writer_crtas = {
        output.buffer()->address(),
    };

    reader_crtas.insert(reader_crtas.end(), input_runtime_args.begin(), input_runtime_args.end());
    reader_crtas.insert(reader_crtas.end(), other_runtime_args.begin(), other_runtime_args.end());
    writer_crtas.insert(writer_crtas.end(), output_runtime_args.begin(), output_runtime_args.end());

    SetCommonRuntimeArgs(program, reader_kernel_id, reader_crtas);
    SetCommonRuntimeArgs(program, writer_kernel_id, writer_crtas);

    return {std::move(program), {reader_kernel_id, writer_kernel_id}};
}

void MorehBugReportRankCrtaOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernels_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernels_id;
}
}  // namespace ttnn::operations::moreh::moreh_bug_report_rank_crta
