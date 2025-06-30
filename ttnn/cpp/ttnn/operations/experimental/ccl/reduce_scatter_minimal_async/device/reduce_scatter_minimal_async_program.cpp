// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_minimal_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "tt_metal/fabric/fabric_mux_config.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

using namespace ccl;

void append_fabric_mux_connection_ct_args(
    const CoreCoord& mux_virtual_core,
    const uint32_t num_buffers,
    const uint32_t buffer_size_bytes,
    const tt::tt_fabric::FabricMuxChannelType channel_type,
    uint32_t worker_id,
    const tt::tt_fabric::FabricMuxConfig& mux_kernel_config,
    std::vector<uint32_t>& writer_ct_args) {
    writer_ct_args.push_back(mux_virtual_core.x);
    writer_ct_args.push_back(mux_virtual_core.y);
    writer_ct_args.push_back(num_buffers);
    writer_ct_args.push_back(buffer_size_bytes);
    writer_ct_args.push_back(mux_kernel_config.get_channel_base_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_info_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_connection_handshake_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_flow_control_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_buffer_index_address(channel_type, worker_id));
    writer_ct_args.push_back(mux_kernel_config.get_status_address());
    writer_ct_args.push_back(mux_kernel_config.get_channel_credits_stream_id(channel_type, worker_id));
}

void append_fabric_mux_connection_rt_args(
    const CoreCoord& worker_logical_core, tt::tt_metal::Program& program, std::vector<uint32_t>& worker_rt_args) {
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
    worker_rt_args.push_back(CreateSemaphore(program, {worker_logical_core}, 0));
}

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<uint32_t>& cluster_axis) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler> empty_fused_op_signaler;
    return reduce_scatter_minimal_async_helper(
        program,
        input_tensor,
        intermediate_tensor,
        sender_device,
        forward_device,
        backward_device,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        sub_device_id,
        empty_fused_op_signaler);
}

tt::tt_metal::operation::ProgramWithCallbacks reduce_scatter_minimal_async_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::ReduceScatterFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset) {
    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    bool fuse_op = fused_op_signaler.has_value();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_tensor, output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores
    // 2 senders per direction (2: forward, backward) per link (num_links)
    // Each sender is reader + compute + writer
    uint32_t num_directions_per_link = 2;
    uint32_t num_mux_cores_per_direction_per_link = 1;
    uint32_t num_workers_per_direction = 1;
    uint32_t num_cores_per_link =
        num_directions_per_link * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
    uint32_t num_workers_per_link = num_directions_per_link * num_workers_per_direction;
    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);
    std::set<CoreRange> sender_worker_core_ranges;
    std::set<CoreRange> sender_forward_core_ranges;
    std::set<CoreRange> sender_backward_core_ranges;
    std::set<CoreRange> mux_forward_core_ranges;
    std::set<CoreRange> mux_backward_core_ranges;
    uint32_t core_id = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            const auto& mux_core = all_cores[core_id++];
            if (dir) {
                mux_forward_core_ranges.insert(CoreRange(mux_core));
            } else {
                mux_backward_core_ranges.insert(CoreRange(mux_core));
            }
            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                const auto& worker_core = all_cores[core_id++];
                if (dir) {
                    sender_forward_core_ranges.insert(CoreRange(worker_core));
                } else {
                    sender_backward_core_ranges.insert(CoreRange(worker_core));
                }
                sender_worker_core_ranges.insert(CoreRange(worker_core));
            }
        }
    }
    CoreRangeSet sender_worker_core_range_set = CoreRangeSet(sender_worker_core_ranges);
    CoreRangeSet sender_forward_core_range_set = CoreRangeSet(sender_forward_core_ranges);
    CoreRangeSet sender_backward_core_range_set = CoreRangeSet(sender_backward_core_ranges);
    CoreRangeSet mux_forward_core_range_set = CoreRangeSet(mux_forward_core_ranges);
    CoreRangeSet mux_backward_core_range_set = CoreRangeSet(mux_backward_core_ranges);

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto& input_tensor_shape = input_tensor.get_padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto num_batches = input_tensor_shape[0];
    const auto batch_slice_num_pages = input_tensor_num_pages / ring_size / num_batches;
    const auto batch_slice_num_pages_per_worker = batch_slice_num_pages / (num_workers_per_link * num_links);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t tiles_to_write_per_packet = 1;
    uint32_t tile_granularity = num_pages_per_packet < 4 ? 4 * num_pages_per_packet : 8;
    uint32_t cb_num_pages = 3 * tile_granularity;  // double buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{input_cb_index, df}})
            .set_page_size(input_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_input_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_input_config);
    uint32_t intermediate_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig cb_intermediate_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{intermediate_cb_index, df}})
            .set_page_size(intermediate_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_intermediate_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_intermediate_config);
    uint32_t reader_output_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_reader_output_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{reader_output_cb_index, df}})
            .set_page_size(reader_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_reader_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reader_output_config);
    uint32_t compute_output_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_compute_output_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{compute_output_cb_index, df}})
            .set_page_size(compute_output_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_compute_output_workers =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_compute_output_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    static constexpr auto num_packet_headers_storable = 4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range_set, cb_reserved_packet_header_config);

    TT_FATAL(
        !(input_tensor_shape[3] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[3],
        tt::constants::TILE_WIDTH);
    uint32_t input_tensor_Wt = input_tensor_shape[3] / tt::constants::TILE_WIDTH;

    // KERNEL CREATION
    std::vector<KernelHandle> reader_kernel_ids;
    std::vector<KernelHandle> writer_kernel_ids;
    std::vector<KernelHandle> reduce_kernel_ids;
    std::vector<KernelHandle> mux_kernel_ids;

    if (fuse_op) {
        fused_op_signaler->init_reduce_scatter(program, mesh_device, sender_worker_core_range_set);
    }

    // Kernel Runtime Args
    CoreCoord drain_sync_core;
    for (uint32_t link = 0; link < num_links; link++) {
        for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
            // Fabrix mux kernel
            uint32_t mux_core_offset =
                link * num_cores_per_link + dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
            CoreCoord mux_logical_core = all_cores[mux_core_offset];
            CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);

            auto num_full_size_channels = num_workers_per_direction;
            auto num_header_only_channels = 0;
            uint32_t payload_size_bytes = tiles_to_write_per_packet * op_config.get_page_size();
            uint32_t num_buffers_full_size_channels = num_full_size_channels * 8;
            size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_max_buffer_size_bytes_full_size_channel();
            const uint32_t l1_unreserved_base_address =
                sender_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
            const size_t mux_base_l1_address = l1_unreserved_base_address;
            uint32_t num_full_size_channel_iters = 1;

            auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
                num_full_size_channels,
                num_header_only_channels,
                num_buffers_full_size_channels,
                0,
                buffer_size_bytes_full_size_channel,
                mux_base_l1_address);
            if (num_full_size_channel_iters > 1) {
                mux_kernel_config.set_num_full_size_channel_iters(num_full_size_channel_iters);
            }
            size_t mux_termination_signal_address = mux_kernel_config.get_termination_signal_address();

            auto mux_kernel_id = tt::tt_metal::CreateKernel(
                program,
                "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                {mux_logical_core},
                tt::tt_metal::DataMovementConfig{
                    .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                    .noc = tt::tt_metal::NOC::RISCV_0_default,
                    .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
                    .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            mux_kernel_ids.push_back(mux_kernel_id);

            std::vector<std::pair<size_t, size_t>> addresses_to_clear = {std::make_pair(
                mux_kernel_config.get_start_address_to_clear(), mux_kernel_config.get_num_bytes_to_clear())};
            std::vector<uint32_t> mux_rt_args = {};
            if (dir) {  // forward
                tt::tt_fabric::append_fabric_connection_rt_args(
                    sender_device->id(), forward_device.value()->id(), link, program, {mux_logical_core}, mux_rt_args);
            } else {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    sender_device->id(), backward_device.value()->id(), link, program, {mux_logical_core}, mux_rt_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);

            for (const auto& [start_address, num_bytes] : addresses_to_clear) {
                std::vector<uint32_t> zero_vec((num_bytes / sizeof(uint32_t)), 0);
                tt::tt_metal::detail::WriteToDeviceL1(sender_device, mux_logical_core, start_address, zero_vec);
            }

            for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                drain_sync_core = mesh_device->worker_core_from_logical_core(core);

                uint32_t worker_id = link * num_workers_per_direction + worker;
                uint32_t num_workers = num_links * num_workers_per_direction;

                // Reader
                auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
                sender_reader_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    input_cb_index,                                          // cb_input_id
                    intermediate_cb_index,                                   // cb_intermediate_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    fuse_op,                                                 // fused op
                    tiles_to_write_per_packet,                               // contig_pages_advanced
                    dir,                                                     // direction
                };
                auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "reduce_scatter_minimal_async_reader.cpp",
                    {core},
                    sender_reader_kernel_config);
                reader_kernel_ids.push_back(worker_sender_reader_kernel_id);

                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),                 // input_tensor_address
                    intermediate_tensor.buffer()->address(),          // intermediate_tensor_address
                    semaphore.at(dir).address(),                      // out_ready_semaphore
                    semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    worker_id,                                        // link
                    num_workers,                                      // num_links
                    input_tensor_Wt / ring_size,                      // slice_Wt
                    (worker_id * batch_slice_num_pages / num_workers) %
                        (input_tensor_Wt / ring_size),  // start_pages_read_in_row
                    (worker_id * batch_slice_num_pages / num_workers) / (input_tensor_Wt / ring_size) *
                        input_tensor_Wt,                                   // start_row_offset
                    worker_id * batch_slice_num_pages / num_workers,       // start_tiles_read
                    (worker_id + 1) * batch_slice_num_pages / num_workers  // start_tiles_to_read
                };
                if (fuse_op) {
                    fused_op_signaler->push_reduce_scatter_fused_op_rt_args(reader_rt_args);
                }
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

                // Writer
                auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
                sender_writer_kernel_config.compile_args = {
                    ring_index,                                              // my_chip_id
                    reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
                    num_packet_headers_storable,                             // num_packet_headers_storable
                    static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
                    static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
                    compute_output_cb_index,                                 // cb_compute_output_id
                    reader_output_cb_index,                                  // cb_reader_output_id
                    tile_granularity,                                        // packet_size_in_pages
                    op_config.get_page_size(),                               // tensor0_page_size
                    input_tensor_Wt,                                         // input_tensor_Wt
                    batch_slice_num_pages,                                   // batch_slice_num_pages
                    ring_size,                                               // ring_size
                    num_batches,                                             // num_batches
                    tiles_to_write_per_packet,                               // contig_pages_advanced
                    dir,                                                     // direction
                    mux_termination_signal_address,  // termination address for this link dir mux
                    !worker                          // master worker
                };
                append_fabric_mux_connection_ct_args(
                    mux_virtual_core,
                    num_buffers_full_size_channels,
                    buffer_size_bytes_full_size_channel,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    worker,
                    mux_kernel_config,
                    sender_writer_kernel_config.compile_args);
                auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "reduce_scatter_minimal_async_writer.cpp",
                    {core},
                    sender_writer_kernel_config);
                writer_kernel_ids.push_back(worker_sender_writer_kernel_id);

                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),          // intermediate_tensor_address
                    output_tensor.buffer()->address(),                // output_tensor_address
                    drain_sync_core.x,                                // out_ready_sem_noc0_x
                    drain_sync_core.y,                                // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                      // out_ready_semaphore
                    semaphore.at(num_directions_per_link).address(),  // batch_ready_semaphore
                    worker_id,                                        // link
                    num_workers,                                      // num_links
                    input_tensor_Wt / ring_size,                      // slice_Wt
                    (worker_id * batch_slice_num_pages / num_workers) %
                        (input_tensor_Wt / ring_size),  // start_pages_read_in_row
                    (worker_id * batch_slice_num_pages / num_workers) / (input_tensor_Wt / ring_size) *
                        input_tensor_Wt,                                   // start_row_offset
                    worker_id * batch_slice_num_pages / num_workers,       // tiles_read
                    (worker_id + 1) * batch_slice_num_pages / num_workers  // tiles_to_read
                };
                append_fabric_mux_connection_rt_args(core, program, writer_rt_args);
                tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

                // Reduce kernel
                auto sender_reduce_kernel_config = tt::tt_metal::ComputeConfig{};
                sender_reduce_kernel_config.compile_args = {
                    input_cb_index,
                    intermediate_cb_index,
                    compute_output_cb_index,
                    batch_slice_num_pages,
                    tile_granularity,
                    ring_size,
                    num_batches,
                    num_links,
                    dir};

                auto sender_reduce_kernel_id = tt::tt_metal::CreateKernel(
                    program,
                    "ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/kernels/"
                    "reduction.cpp",
                    {core},
                    sender_reduce_kernel_config);
                reduce_kernel_ids.push_back(sender_reduce_kernel_id);

                std::vector<uint32_t> reduce_rt_args = {link};
                tt::tt_metal::SetRuntimeArgs(program, sender_reduce_kernel_id, {core}, reduce_rt_args);
            }
        }
    }

    auto override_runtime_arguments_callback =
        [reader_kernel_ids,
         writer_kernel_ids,
         all_cores,
         num_links,
         num_directions_per_link,
         num_workers_per_direction,
         num_mux_cores_per_direction_per_link,
         num_cores_per_link](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[1];
            const auto& intermed = output_tensors[0];

            // update senders
            uint32_t core_idx = 0;
            for (uint32_t link = 0; link < num_links; link++) {
                for (uint32_t dir = 0; dir < num_directions_per_link; dir++) {
                    for (uint32_t worker = 0; worker < num_workers_per_direction; worker++) {
                        uint32_t mux_core_offset =
                            link * num_cores_per_link +
                            dir * (num_mux_cores_per_direction_per_link + num_workers_per_direction);
                        CoreCoord core = all_cores[mux_core_offset + num_mux_cores_per_direction_per_link + worker];
                        std::vector<std::vector<RuntimeArgsData>> reader_runtime_args =
                            GetRuntimeArgs(program, reader_kernel_ids[core_idx]);
                        std::vector<std::vector<RuntimeArgsData>> writer_runtime_args =
                            GetRuntimeArgs(program, writer_kernel_ids[core_idx]);

                        // sender reader
                        auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                        worker_reader_sender_runtime_args[0] = input.buffer()->address();
                        worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                        // sender writer
                        auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                        worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                        worker_writer_sender_runtime_args[1] = output.buffer()->address();

                        core_idx++;
                    }
                }
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
