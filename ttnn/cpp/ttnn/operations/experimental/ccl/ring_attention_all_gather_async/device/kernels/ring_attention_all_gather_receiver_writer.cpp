// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType output_tensor_buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr uint32_t cb_intermediate_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t output_tensor_page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr Topology topology = static_cast<Topology>(get_compile_time_arg_val(7));
constexpr bool direction = get_compile_time_arg_val(8);
constexpr bool fuse_op = get_compile_time_arg_val(9);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(10);
constexpr uint32_t num_inputs = get_compile_time_arg_val(11);

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint8_t r = 0; r < 32; ++r) {
        SliceRange sr_left = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 0, .w1 = 16, .ws = 1};
        SliceRange sr_right = SliceRange{.h0 = r, .h1 = (uint8_t)(r + 1), .hs = 1, .w0 = 17, .w1 = 32, .ws = 1};
        DPRINT << (uint)r << ": " << TileSlice(cb_id, tile_id, sr_left, false, untilize) << " "
               << TileSlice(cb_id, tile_id, sr_right, true, untilize) << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    uint32_t input_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t output_tensor_Ht = get_arg_val<uint32_t>(arg_idx++);
    uint32_t gather_dim = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_batch_head_count = get_arg_val<uint32_t>(arg_idx++);
    uint32_t slice_num_pages = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    address_t output_tensor_addresses[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        address_t output_buffer_addr = get_arg_val<address_t>(arg_idx++);
        output_tensor_addresses[input_idx] = output_buffer_addr;
    }

    OpSignaler op_signaler;
    if constexpr (fuse_op) {
        op_signaler = OpSignaler(arg_idx);
    }

    // interleaved addrgen
    constexpr bool output_is_dram = output_tensor_buffer_type == tt::tt_metal::BufferType::DRAM;
    InterleavedAddrGenFast<output_is_dram> output_tensor_addrgens[num_inputs];
    for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
        auto output_tensor_addrgen = InterleavedAddrGenFast<output_is_dram>{
            .bank_base_address = output_tensor_addresses[input_idx],
            .page_size = output_tensor_page_size,
            .data_format = get_dataformat(cb_intermediate_id)};
        output_tensor_addrgens[input_idx] = output_tensor_addrgen;
    }

    uint32_t forward_writes = 0;
    uint32_t backward_writes = 0;

    uint32_t forward_writes_expected, backward_writes_expected;
    if (topology == Topology::Linear) {
        forward_writes_expected = num_targets_backward_direction;
        backward_writes_expected = num_targets_forward_direction;
    } else if (topology == Topology::Ring) {
        forward_writes_expected = num_targets_forward_direction - 1;
        backward_writes_expected = num_targets_backward_direction - 1;
    }

    uint32_t slices_received = 0;
    uint32_t slices_expected;
    if (topology == Topology::Linear) {
        if (direction == 1) {
            slices_expected = num_targets_forward_direction;
        } else {
            slices_expected = num_targets_backward_direction;
        }
    } else if (topology == Topology::Ring) {
        if (direction == 1) {
            slices_expected = num_targets_backward_direction;
        } else {
            slices_expected = num_targets_forward_direction;
        }
    }

    while (slices_received < slices_expected) {
        slices_received++;

        int sender_chip_id;
        uint32_t actual_sender_chip_id;
        if (direction == 1) {
            sender_chip_id = my_chip_id + slices_received;
            actual_sender_chip_id = (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
        } else {
            sender_chip_id = my_chip_id - slices_received;
            actual_sender_chip_id = (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
        }

        for (uint32_t input_idx = 0; input_idx < num_inputs; input_idx++) {
            uint32_t row_offset = 0;
            uint32_t tile_id_start = 0;
            uint32_t payload_size_bytes = contig_pages_advanced * output_tensor_page_size;
            if (gather_dim == 3) {
                tile_id_start = actual_sender_chip_id * input_tensor_Wt;
            } else {
                tile_id_start = actual_sender_chip_id * input_tensor_Ht * input_tensor_Wt;
            }
            for (uint32_t bh_idx = 0; bh_idx < input_batch_head_count; bh_idx++) {
                for (uint32_t row_idx = 0; row_idx < input_tensor_Ht; row_idx++) {
                    for (uint32_t col_idx = 0; col_idx < input_tensor_Wt; col_idx += packet_size_in_pages) {
                        cb_wait_front(cb_intermediate_id, packet_size_in_pages);
                        size_t l1_read_addr = get_read_ptr(cb_intermediate_id);

                        for (uint32_t j = 0; j < packet_size_in_pages; j += contig_pages_advanced) {
                            uint32_t tile_id = tile_id_start + row_offset + col_idx + j;
                            noc_async_write_tile(tile_id, output_tensor_addrgens[input_idx], l1_read_addr);

                            l1_read_addr += payload_size_bytes;
                        }
                        cb_pop_front(cb_intermediate_id, packet_size_in_pages);
                    }
                    row_offset += output_tensor_Wt;
                }
                row_offset = 0;
                tile_id_start += output_tensor_Wt * output_tensor_Ht;
            }
        }

        if (fuse_op) {
            // Signal matmul to go
            op_signaler.synchronize_workers_and_signal_op(actual_sender_chip_id);
        }
    }

    noc_async_write_barrier();
}
