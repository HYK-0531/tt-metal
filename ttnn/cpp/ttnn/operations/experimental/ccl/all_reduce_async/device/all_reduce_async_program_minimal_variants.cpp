// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "all_reduce_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
using namespace tt::constants;

namespace ttnn {

using namespace ccl;

CoreRangeSet cores_to_corerangeset(const std::vector<CoreCoord>& cores) {
    std::vector<CoreRange> core_ranges;
    for (const auto& core : cores) {
        core_ranges.push_back(CoreRange(core));
    }
    return CoreRangeSet(core_ranges);
}
std::vector<CoreCoord> compute_top_row_ethernet_cores(
    IDevice* device, std::optional<IDevice*> forward_device, std::optional<IDevice*> backward_device) {
    std::vector<CoreCoord> reordered_ethernet_cores;
    if (forward_device.has_value()) {
        for (auto core : device->get_ethernet_sockets(forward_device.value()->id())) {
            auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
            reordered_ethernet_cores.push_back(core_virtual);
        }
        std::sort(reordered_ethernet_cores.begin(), reordered_ethernet_cores.end(), [](auto& a, auto& b) {
            return a.x < b.x;
        });
    } else if (backward_device.has_value()) {
        for (auto core : device->get_ethernet_sockets(backward_device.value()->id())) {
            auto core_virtual = device->virtual_core_from_logical_core(core, CoreType::ETH);
            reordered_ethernet_cores.push_back(core_virtual);
        }
        std::sort(reordered_ethernet_cores.begin(), reordered_ethernet_cores.end(), [](auto& a, auto& b) {
            return a.x < b.x;
        });
    }
    for (auto& eth_core : reordered_ethernet_cores) {
        eth_core.y = 16;
    }
    return reordered_ethernet_cores;
}

bool core_vector_contains_core(std::vector<CoreCoord> core_vector, CoreCoord core) {
    auto it = std::find(core_vector.begin(), core_vector.end(), core);
    return it != core_vector.end();
}

CoreCoord wh_glx_physical_worker_core_from_logical_core(CoreCoord logical_core) {
    auto physical_x = logical_core.x <= 3 ? logical_core.x + 1 : logical_core.x + 2;
    auto physical_y = logical_core.y <= 4 ? logical_core.y + 1 : logical_core.y + 2;
    return CoreCoord(physical_x, physical_y);
}
CoreCoord wh_glx_logical_worker_core_from_physical_core(CoreCoord physical_core) {
    auto logical_x = physical_core.x <= 4 ? physical_core.x - 1 : physical_core.x - 2;
    auto logical_y = physical_core.y <= 5 ? physical_core.y - 1 : physical_core.y - 2;
    return CoreCoord(logical_x, logical_y);
}

std::tuple<CoreRangeSet, std::vector<CoreCoord>> get_optimal_worker_core_placement(
    std::vector<CoreCoord> ethernet_cores_virtual, const CoreRangeSet& available_corerangeset, uint32_t num_links) {
    std::vector<CoreCoord> sender_worker_cores;

    // Get all available ranges from the CoreRangeSet
    auto available_ranges = available_corerangeset.ranges();

    // Convert available ranges to cores
    auto available_cores = corerange_to_cores(available_corerangeset, std::nullopt, true);

    for (uint32_t link = 0; link < num_links; link++) {
        auto core_virtual = ethernet_cores_virtual[link];
        CoreCoord eth_core_physical;
        eth_core_physical.x = core_virtual.x >= 22 ? (core_virtual.x - 16) : (core_virtual.x - 17);
        eth_core_physical.y = (core_virtual.y - 16) * 6;

        // Shift down the worker core below the ethernet core
        CoreCoord worker_core_physical = CoreCoord(eth_core_physical.x, eth_core_physical.y + 1);
        CoreCoord worker_core_logical = wh_glx_logical_worker_core_from_physical_core(worker_core_physical);

        bool found_valid_core = false;

        // Iterate through ranges
        for (size_t i = 0; i < available_ranges.size(); i++) {
            auto current_range_start = available_ranges[i].start_coord;
            auto current_range_end = available_ranges[i].end_coord;

            if (i < available_ranges.size() - 1) {
                auto next_range_start = available_ranges[i + 1].start_coord;
                // check if its in the middle of the ranges and check which range to prefer
                if (worker_core_logical.x > current_range_end.x && worker_core_logical.x < next_range_start.x) {
                    if (next_range_start.x - worker_core_logical.x < worker_core_logical.x - current_range_end.x) {
                        worker_core_logical.x = next_range_start.x;
                        continue;  // Skips as it is closer to the next range
                    } else {
                        worker_core_logical.x = current_range_end.x;
                    }
                    // skip if the core is in the next range or beyond
                } else if (worker_core_logical.x >= next_range_start.x) {
                    continue;
                }
            }
            // Either core is front of the range, or inside the range or at the back in case of last range
            // bring the core in the bounds of the range
            if (worker_core_logical.x < current_range_start.x) {
                worker_core_logical.x = current_range_start.x;
            } else if (worker_core_logical.x > current_range_end.x) {
                worker_core_logical.x = current_range_end.x;
            }

            // Try to find a valid core within this range
            for (uint32_t core_y = worker_core_logical.y; core_y <= current_range_end.y; core_y++) {
                worker_core_logical.y = core_y;
                if (!core_vector_contains_core(sender_worker_cores, worker_core_logical) &&
                    core_vector_contains_core(available_cores, worker_core_logical)) {
                    found_valid_core = true;
                    break;
                }
            }

            if (found_valid_core) {
                break;
            }
        }

        if (found_valid_core) {
            sender_worker_cores.push_back(worker_core_logical);
        } else {
            TT_ASSERT("No valid worker core found for a link");
        }
    }

    std::set<CoreRange> sender_worker_cores_set;
    for (const auto& core : sender_worker_cores) {
        sender_worker_cores_set.insert(CoreRange(core));
    }
    CoreRangeSet sender_worker_corerangeset = CoreRangeSet(sender_worker_cores_set);

    return {sender_worker_corerangeset, sender_worker_cores};
}

tt::tt_metal::operation::ProgramWithCallbacks all_reduce_async_minimal_multi_core_with_workers(
    const Tensor& input_tensor,
    const Tensor& buffer_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool enable_persistent_fabric_mode) {
    tt::tt_metal::Program program{};

    IDevice* device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            forward_device.value_or(nullptr),
            backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric_mode,
            num_links,
            topology);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    size_t num_targets_forward = 0;
    size_t num_targets_backward = 0;
    bool dynamic_alternate = false;
    if (topology == ccl::Topology::Linear) {
        LineTopology line_topology(ring_size, ring_index);
        num_targets_forward =
            line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
        num_targets_backward =
            line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);
    } else if (topology == ccl::Topology::Ring) {
        // TODO: Commonize
        num_targets_forward = tt::div_up(ring_size - 1, 2);
        num_targets_backward = ring_size - 1 - num_targets_forward;
        constexpr bool static_alternate = true;
        if constexpr (static_alternate) {
            if (ring_index % 2 == 0) {
                std::swap(num_targets_forward, num_targets_backward);
            }
        }
        // Even ring size will result in uneven fwd/backward distances
        dynamic_alternate = ring_size % 2 == 0;
    }

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto num_input_cores = input_tensor_cores.num_cores();
    const auto output_tensor_num_pages = output_tensor.buffer()->num_pages();
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;
    const auto num_output_cores = output_tensor_cores.num_cores();

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    bool optimal_placement = true;
    auto sub_device_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value_or(device->get_sub_device_ids().at(0)));

    std::vector<CoreCoord> ethernet_cores_virtual =
        compute_top_row_ethernet_cores(device, forward_device, backward_device);

    CoreRangeSet sender_worker_core_range;
    std::vector<CoreCoord> sender_worker_cores;
    if (optimal_placement) {
        std::tie(sender_worker_core_range, sender_worker_cores) =
            get_optimal_worker_core_placement(ethernet_cores_virtual, sub_device_cores, num_links);
    } else {
        std::tie(sender_worker_core_range, sender_worker_cores) =
            choose_worker_cores(num_links, num_workers_per_link, enable_persistent_fabric_mode, sub_device_cores);
    }

    if (device->id() == 4) {
        tt::log_info("dev {} ethernet_cores: {}", device->id(), ethernet_cores_virtual);
        tt::log_info("sender_worker_cores: {}", sender_worker_cores);
    }

    auto worker_reducer_cores = output_tensor_cores.merge(sender_worker_core_range);

    std::vector<CoreRange> worker_reducer_corerange_vec;
    for (const auto& cr : sub_device_cores.ranges()) {
        const auto intersection = worker_reducer_cores.intersection(cr);
        if (intersection.size() > 0) {
            worker_reducer_corerange_vec.push_back(intersection.bounding_box());
        }
    }

    // worker_reducer_corerange_vec is the bounding box of the output_tensor_cores + worker and but respecting
    // boundaries of subdevice grids
    CoreRangeSet worker_reducer_cores_all(worker_reducer_corerange_vec);
    // worker_reducer_cores_unused is the cores that should do no work
    auto worker_reducer_cores_unused = worker_reducer_cores_all.subtract(worker_reducer_cores);

    auto non_worker_reducer_cores = output_tensor_cores.subtract(sender_worker_core_range);
    auto non_reducer_worker_cores = sender_worker_core_range.subtract(output_tensor_cores);

    tt::log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    tt::log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    tt::log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    tt::log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    tt::log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    tt::log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = tt::div_up(output_tensor_cores.num_cores(), num_links) * output_tensor_shard_num_pages;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CBIndex::c_3;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Reduction kernel setup
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);

    // Create output tensor splits
    // TODO: Currently does not support output shards being split across multiple links
    std::vector<CoreRangeSet> output_corerangeset_per_link;
    std::vector<uint32_t> num_output_cores_in_link(num_links, 0);
    uint32_t output_cores_per_link = tt::div_up(output_tensor_cores.num_cores(), num_links);
    uint32_t num_assigned_cores = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_cores_this_link = std::min(output_cores_per_link, num_output_cores - num_assigned_cores);
        output_corerangeset_per_link.emplace_back(
            cores_to_corerangeset(std::vector<CoreCoord>(
                                      output_cores_vec.begin() + num_assigned_cores,
                                      output_cores_vec.begin() + num_assigned_cores + num_cores_this_link))
                .merge_ranges());
        num_output_cores_in_link[link] = num_cores_this_link;
        num_assigned_cores += num_cores_this_link;
    }

    // Create output tensor page splits
    std::vector<uint32_t> output_tensor_pages_in_link(num_links, 0);
    uint32_t num_assigned_pages = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_output_pages_per_link = output_tensor_shard_num_pages * num_output_cores_in_link[link];
        uint32_t num_pages_this_link =
            std::min(num_output_pages_per_link, output_tensor_num_pages - num_assigned_pages);
        output_tensor_pages_in_link[link] = num_pages_this_link;
        num_assigned_pages += num_pages_this_link;
    }

    // Create input tensor splits
    /*
        Overview of algorithm:

        - Ouput: each link gets assigned a start and end core index, since multiple links
            may have to read different offesets within a shard on the same core
        - First, assign all the necessary cores needed for a link. This may result in the link
            containing extra pages. This will result in an overflow, which is used to detect
            the tile offset (within a shard) for the next link
        - Once you have the start_core_idx, the end_core_idx is calculated by
            getting the upper bound on the number of cores needed to read the pages assigned
            to the link, accounting for the tile offset. This calculation is done by dividing
            the upper bound on the number of pages assigned to this link
            (num_pages_this_link + input_tensor_tile_offset) by the number of pages in a shard.
            This gives the number of cores needed to read the pages assigned to this link.
        - If an overflow is detected, then the start_core_idx for the next link is set
            to the end_core_idx of the current link. Ie, 2 links read from the same core
    */
    std::vector<std::pair<uint32_t, uint32_t>> input_cores_idx_per_link(num_links, {0, 0});
    std::vector<uint32_t> input_tensor_tile_offset_per_link(num_links, 0);
    uint32_t start_core_idx = 0;
    uint32_t num_pages_overflow = 0;
    for (uint32_t link = 0; link < num_links; link++) {
        uint32_t num_pages_this_link = output_tensor_pages_in_link[link];

        // Get offset based on previous overflow
        uint32_t input_tensor_tile_offset =
            (input_tensor_shard_num_pages - num_pages_overflow) % input_tensor_shard_num_pages;
        input_tensor_tile_offset_per_link[link] = input_tensor_tile_offset;

        uint32_t end_core_idx = std::min(
            start_core_idx + tt::div_up(num_pages_this_link + input_tensor_tile_offset, input_tensor_shard_num_pages),
            num_input_cores);

        // Num pages allocated based on number of input cores selected for this link
        uint32_t num_pages_allocated =
            (end_core_idx - start_core_idx) * input_tensor_shard_num_pages - input_tensor_tile_offset;

        // Update overflow
        num_pages_overflow = num_pages_allocated - num_pages_this_link;

        // Store core indices
        input_cores_idx_per_link[link] = {start_core_idx, end_core_idx};

        // Set start index based on overflow
        if (num_pages_overflow > 0) {
            start_core_idx = end_core_idx - 1;
        } else {
            start_core_idx = end_core_idx;
        }
    }

    // Create reduction semaphores for each link
    std::vector<uint32_t> reduction_semaphore_ids(num_links, 0);
    for (uint32_t link = 0; link < num_links; link++) {
        reduction_semaphore_ids[link] = tt::tt_metal::CreateSemaphore(program, worker_reducer_cores_all, 0);
    }

    /* reduction cb */
    uint32_t reduction_CB_single_tile_size = output_tensor.get_tensor_spec().tile().get_tile_size(df);
    uint32_t reduction_CB_tiles = output_tensor_shard_num_pages * ring_size;
    uint32_t reduction_CB_size = reduction_CB_tiles * reduction_CB_single_tile_size;

    uint32_t reduction_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig reduction_cb_config =
        tt::tt_metal::CircularBufferConfig(reduction_CB_size, {{reduction_cb_index, df}})
            .set_page_size(reduction_cb_index, reduction_CB_single_tile_size)
            .set_globally_allocated_address(*buffer_tensor.buffer());
    auto cb_reduction = tt::tt_metal::CreateCircularBuffer(program, worker_reducer_cores_all, reduction_cb_config);

    /* out cb */
    uint32_t out_CB_single_tile_size = output_tensor.get_tensor_spec().tile().get_tile_size(df);
    uint32_t out_CB_tiles = output_tensor_shard_num_pages;
    uint32_t out_CB_size = out_CB_tiles * out_CB_single_tile_size;

    uint32_t out_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig out_cb_config =
        tt::tt_metal::CircularBufferConfig(out_CB_size, {{out_cb_index, df}})
            .set_page_size(out_cb_index, out_CB_single_tile_size)
            .set_globally_allocated_address(*output_tensor.buffer());  // TODO: Remove once new cb attached for output
    auto cb_out = tt::tt_metal::CreateCircularBuffer(
        program, output_tensor_cores, out_cb_config);  // TODO: This should be the output cores instead

    // KERNEL CREATION
    tt::tt_metal::NOC reader_noc = tt::tt_metal::NOC::NOC_1;
    tt::tt_metal::NOC writer_noc = tt::tt_metal::NOC::NOC_0;
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
        reduction_cb_index,         // reduction_cb_index
        reduction_CB_tiles,         // total_num_reduction_tiles
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    auto worker_reducer_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_reader.cpp",
        worker_reducer_cores_all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = reader_noc,
            .compile_args = reader_compile_args});

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        ring_index,                               // my_chip_id
        reserved_packet_header_CB_index,          // reserved_packet_header_cb_id
        num_packet_headers_storable,              // num_packet_headers_storable
        src0_cb_index,                            // cb0_id
        num_pages_per_packet,                     // packet_size_in_pages
        op_config.get_page_size(),                // tensor0_page_size
        num_targets_forward,                      // num_targets_forward_direction
        num_targets_backward,                     // num_targets_backward_direction
        static_cast<uint32_t>(dynamic_alternate)  // dynamic_alternate
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    auto worker_reducer_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/dataflow/"
        "worker_writer.cpp",
        worker_reducer_cores_all,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = writer_noc,
            .compile_args = writer_compile_args});

    // Create reduction dataflow kernel
    if (worker_reducer_cores_unused.size() > 0) {
        tt::tt_metal::SetRuntimeArgs(program, worker_reducer_reader_kernel_id, worker_reducer_cores_unused, {0, 0});
        tt::tt_metal::SetRuntimeArgs(program, worker_reducer_writer_kernel_id, worker_reducer_cores_unused, {0, 0});
    }
    tt::tt_metal::SetRuntimeArgs(program, worker_reducer_writer_kernel_id, non_worker_reducer_cores, {0, 1});

    // Create reduction dataflow kernel
    auto reduction_kernel_config = tt::tt_metal::ComputeConfig{};
    reduction_kernel_config.compile_args = {
        reduction_cb_index,  // reduction_cb_index
        out_cb_index,        // out_cb_index
    };
    auto reduction_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/kernels/compute/"
        "reduction.cpp",
        worker_reducer_cores_all,
        reduction_kernel_config);
    tt::tt_metal::SetRuntimeArgs(
        program, reduction_kernel_id, output_tensor_cores, {0, 1, ring_size, output_tensor_shard_num_pages});
    tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, non_reducer_worker_cores, {1, 0});
    if (worker_reducer_cores_unused.size() > 0) {
        tt::tt_metal::SetRuntimeArgs(program, reduction_kernel_id, worker_reducer_cores_unused, {0, 0});
    }

    // Kernel Runtime Args
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord drain_sync_core = device->worker_core_from_logical_core(core);
        uint32_t worker_num_tiles_to_read = output_tensor_pages_in_link[link];

        uint32_t input_first_core_tile_start_offset = input_tensor_tile_offset_per_link[link];
        uint32_t output_first_core_tile_start_offset = 0;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_cores_idx_per_link[link].first; i < input_cores_idx_per_link[link].second; i++) {
            auto this_core = device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = output_cores_per_link * link;
             i < output_cores_per_link * link + num_output_cores_in_link[link];
             i++) {
            auto this_core = device->worker_core_from_logical_core(output_cores_vec[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> forward_fabric_connection =
            !forward_device.has_value()
                ? std::nullopt
                : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
        std::optional<tt::tt_fabric::SenderWorkerAdapterSpec> backward_fabric_connection =
            !backward_device.has_value()
                ? std::nullopt
                : std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>(local_fabric_handle->uniquely_connect_worker(
                      device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

        // Set reader runtime args
        bool worker_reducer_overlap = false;
        auto overlap_reduction_semaphore_id = reduction_semaphore_ids[link];
        for (uint32_t inner_link = 0; inner_link < num_links; inner_link++) {
            worker_reducer_overlap = output_corerangeset_per_link[inner_link].contains(core);
            if (worker_reducer_overlap) {
                overlap_reduction_semaphore_id = reduction_semaphore_ids[inner_link];
                break;
            }
        }
        std::vector<uint32_t> worker_reader_rt_args = {
            1,                                   // is_worker
            worker_reducer_overlap ? 1 : 0,      // is_reducer
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        worker_reader_rt_args.insert(
            worker_reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        worker_reader_rt_args.insert(
            worker_reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());

        if (worker_reducer_overlap) {
            worker_reader_rt_args.push_back(overlap_reduction_semaphore_id);  // Add reduction semaphore ID
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_reducer_reader_kernel_id, {core}, worker_reader_rt_args);

        // Set reduction reader runtime args for remaining cores
        auto non_overlapping_reducer_cores = output_corerangeset_per_link[link].subtract(sender_worker_core_range);
        std::vector<uint32_t> reducer_reader_rt_args = {
            0,                              // is_worker
            1,                              // is_reducer
            reduction_semaphore_ids[link],  // reduction_semaphore_id
        };
        tt::tt_metal::SetRuntimeArgs(
            program, worker_reducer_reader_kernel_id, non_overlapping_reducer_cores, reducer_reader_rt_args);

        // Set writer runtime args
        std::vector<uint32_t> mcast_start_x;
        std::vector<uint32_t> mcast_start_y;
        std::vector<uint32_t> mcast_end_x;
        std::vector<uint32_t> mcast_end_y;

        uint32_t num_mcast_cores = 0;
        for (const auto& range : output_corerangeset_per_link[link].ranges()) {
            auto start_core = device->worker_core_from_logical_core(range.start_coord);
            auto end_core = device->worker_core_from_logical_core(range.end_coord);
            num_mcast_cores += range.size();
            bool mcast_range_contains_self = range.contains(core);
            if (mcast_range_contains_self) {
                num_mcast_cores -= 1;
            }
            if (writer_noc == tt::tt_metal::NOC::NOC_1) {
                std::swap(start_core, end_core);
            }
            mcast_start_x.push_back(start_core.x);
            mcast_start_y.push_back(start_core.y);
            mcast_end_x.push_back(end_core.x);
            mcast_end_y.push_back(end_core.y);
        }

        uint32_t out_ready_sem_wait_value = dynamic_alternate ? (ring_size + 1) : ring_size;
        std::vector<uint32_t> writer_rt_args = {
            1,                                    // is_worker
            worker_reducer_overlap ? 1 : 0,       // is_reducer
            reduction_cb_index,                   // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            num_mcast_cores,                      // num_mcast_cores
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            reduction_semaphore_ids[link],        // reduction_semaphore_id
            mcast_start_x.size(),                 // num_mcast_ranges
            link,                                 // link
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());

        writer_rt_args.insert(writer_rt_args.end(), mcast_start_x.begin(), mcast_start_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_start_y.begin(), mcast_start_y.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_x.begin(), mcast_end_x.end());
        writer_rt_args.insert(writer_rt_args.end(), mcast_end_y.begin(), mcast_end_y.end());

        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_fabric_connection.has_value());
        if (forward_fabric_connection.has_value()) {
            auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
            append_worker_to_fabric_edm_sender_rt_args(
                forward_fabric_connection.value(),
                sender_worker_flow_control_semaphore_id,
                sender_worker_teardown_semaphore_id,
                sender_worker_buffer_index_semaphore_id,
                writer_rt_args);
        }
        writer_rt_args.push_back(backward_fabric_connection.has_value());
        if (backward_fabric_connection.has_value()) {
            auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
            auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
            append_worker_to_fabric_edm_sender_rt_args(
                backward_fabric_connection.value(),
                sender_worker_flow_control_semaphore_id,
                sender_worker_teardown_semaphore_id,
                sender_worker_buffer_index_semaphore_id,
                writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_reducer_writer_kernel_id, {core}, writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_reducer_reader_kernel_id, worker_reducer_writer_kernel_id, sender_worker_cores, cb_out, cb_reduction](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];
            const auto& buffer_tensor = input_tensors[1];

            auto semaphore = static_cast<const ttnn::AllReduceAsync*>(operation)->semaphore;

            // update senders
            auto& worker_reducer_reader_runtime_args_by_core = GetRuntimeArgs(program, worker_reducer_reader_kernel_id);
            auto& worker_reducer_writer_runtime_args_by_core = GetRuntimeArgs(program, worker_reducer_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reducer_reader_runtime_args = worker_reducer_reader_runtime_args_by_core[core.x][core.y];
                worker_reducer_reader_runtime_args[2] = input.buffer()->address();
                // writer
                auto& worker_reducer_writer_runtime_args = worker_reducer_writer_runtime_args_by_core[core.x][core.y];
                worker_reducer_writer_runtime_args[3] = semaphore.address();
            }
            UpdateDynamicCircularBufferAddress(program, cb_out, *output.buffer());
            UpdateDynamicCircularBufferAddress(program, cb_reduction, *buffer_tensor.buffer());
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
