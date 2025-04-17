#include <tt-metalium/work_split.hpp>
#include "my_new_op_operation.hpp"

namespace ttnn::operations::data_movement {

MyNewOpDeviceOperation::MyDeviceProgramFactory::cached_program_t MyNewOpDeviceOperation::MyDeviceProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    tt::tt_metal::Program program{};

    uint32_t num_tiles = input_tensor_a.volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input_tensor_a.device();

    CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t output_cb_index = tt::CBIndex::c_2;

    bool src0_is_dram = src_buffer_a->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src_buffer_b->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {src0_cb_index, src1_cb_index, src0_is_dram, src1_is_dram};
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index, 1};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/my_new_op_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/dataflow/my_new_op_writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/compute/my_new_op_compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    auto eltwise_unary_kernel_group_2_id = eltwise_unary_kernel_group_1_id;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/my_new_op/device/kernels/compute/my_new_op_compute.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {src_buffer_a->address(),
             num_tiles_written,
             src_buffer_b->address(),
             num_tiles_written,
             num_tiles_per_core});
        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_written, num_tiles_per_core});
        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = unary_reader_kernel_id,
         .writer_kernel_id = unary_writer_kernel_id,
         .compute_kernel_id_1 = eltwise_unary_kernel_group_1_id,
         .compute_kernel_id_2 = eltwise_unary_kernel_group_2_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void MyNewOpDeviceOperation::MyDeviceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& compute_kernel_id_1 = cached_program.shared_variables.compute_kernel_id_1;
    auto& compute_kernel_id_2 = cached_program.shared_variables.compute_kernel_id_2;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    auto& output_tensor = tensor_return_value;

    auto src_buffer_a = input_tensor_a.buffer();
    auto src_buffer_b = input_tensor_b.buffer();
    auto dst_buffer = output_tensor.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        tt::tt_metal::CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[2] = src_buffer_b->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::data_movement
