// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);  // 4
constexpr BufferType intermediate_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr BufferType output_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(5);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(6);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(7);
constexpr uint32_t intermediate_page_size = get_compile_time_arg_val(8);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(9);
constexpr uint32_t batch_slice_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t ring_size = get_compile_time_arg_val(11);
constexpr uint32_t num_batches = get_compile_time_arg_val(12);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(13);
constexpr bool direction = get_compile_time_arg_val(14);
constexpr size_t fabric_mux_termination_address = get_compile_time_arg_val(15);
constexpr bool fabric_mux_worker_master = get_compile_time_arg_val(16);

constexpr uint8_t fabric_mux_x = get_compile_time_arg_val(17);
constexpr uint8_t fabric_mux_y = get_compile_time_arg_val(18);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_channel_base_address = get_compile_time_arg_val(21);
constexpr size_t fabric_mux_connection_info_address = get_compile_time_arg_val(22);
constexpr size_t fabric_mux_connection_handshake_address = get_compile_time_arg_val(23);
constexpr size_t fabric_mux_flow_control_address = get_compile_time_arg_val(24);
constexpr size_t fabric_mux_buffer_index_address = get_compile_time_arg_val(25);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(26);
constexpr uint8_t fabric_mux_channel_id = get_compile_time_arg_val(27);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t batch_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    uint32_t link = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_links = get_arg_val<uint32_t>(arg_idx++);

    uint32_t slice_Wt = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    int32_t start_tiles_read = get_arg_val<int32_t>(arg_idx++);
    uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);

    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    auto mux_connection_handle = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
        fabric_mux_x,
        fabric_mux_y,
        fabric_mux_channel_id,
        fabric_mux_num_buffers_per_channel,
        fabric_mux_channel_buffer_size_bytes,
        fabric_mux_channel_base_address,
        fabric_mux_connection_info_address,
        fabric_mux_connection_handshake_address,
        fabric_mux_flow_control_address,
        fabric_mux_buffer_index_address,
        local_flow_control_address,
        local_teardown_address,
        local_buffer_index_address);

    // need to wait for fabric mux to be ready to accept connections
    tt::tt_fabric::wait_for_fabric_endpoint_ready(
        fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr);
    pkt_hdr->to_chip_unicast(1);

    // interleaved addrgen
    constexpr bool intermediate_is_dram = intermediate_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_addrgen = InterleavedAddrGenFast<intermediate_is_dram>{
        .bank_base_address = intermediate_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};
    constexpr bool output_is_dram = output_type == tt::tt_metal::BufferType::DRAM;
    auto output_addrgen = InterleavedAddrGenFast<output_is_dram>{
        .bank_base_address = output_address,
        .page_size = intermediate_page_size,
        .data_format = get_dataformat(cb_compute_output_id)};

    tt::tt_fabric::fabric_client_connect(mux_connection_handle);

    for (uint32_t b = 0; b < num_batches; b++) {
        int slice_idx = direction ? my_chip_id - 1 : my_chip_id + 1;

        uint32_t batch_slice_offset = batch_slice_num_pages * b;
        for (uint32_t i = 0; i < ring_size; ++i) {
            uint32_t actual_slice_idx;
            if (direction) {
                actual_slice_idx = slice_idx < 0 ? slice_idx + ring_size : slice_idx;
            } else {
                actual_slice_idx = slice_idx >= (int)ring_size ? (uint32_t)slice_idx - ring_size : (uint32_t)slice_idx;
            }

            uint32_t cb_output_id = i > 0 ? cb_compute_output_id : cb_reader_output_id;
            // If not the last slice, write what's on cb_output_id forward
            if (i < (ring_size - 1)) {
                uint32_t stride_Wt = input_tensor_Wt;
                uint32_t pages_read_in_row = start_pages_read_in_row;
                uint32_t row_offset = start_row_offset;
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t input_tile_id_start = actual_slice_idx * slice_Wt;
                if (!direction) {
                    uint32_t backwards_offset = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    tiles_read += backwards_offset;
                    pages_read_in_row += backwards_offset;

                    if (pages_read_in_row >= slice_Wt) {
                        row_offset += stride_Wt;
                        pages_read_in_row = pages_read_in_row - slice_Wt;
                    }
                }

                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = 0;
                    if (direction) {
                        num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                    }
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);

                    for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                        uint32_t payload_size_bytes =
                            std::min(contig_pages_advanced, num_pages_to_read - j) * intermediate_page_size;
                        uint64_t remote_noc0_dest_noc_addr = get_noc_addr(
                            input_tile_id_start + row_offset + pages_read_in_row,
                            intermediate_addrgen,
                            0 /*offset*/,
                            0 /*noc_id*/);
                        pkt_hdr->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{remote_noc0_dest_noc_addr}, payload_size_bytes);
                        tt::tt_fabric::fabric_async_write(
                            mux_connection_handle, pkt_hdr, l1_read_addr, payload_size_bytes);
                        // Note: Must flush write for correctness
                        noc_async_writes_flushed();
                        l1_read_addr += payload_size_bytes;
                        tiles_read++;

                        pages_read_in_row++;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = 0;
                        }
                    }

                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    if (tiles_read < tiles_to_read) {
                        num_pages_to_read = 0;
                        if (!direction) {
                            num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                        } else {
                            num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                        }
                        tiles_read += num_pages_to_read;
                        pages_read_in_row += num_pages_to_read;
                        if (pages_read_in_row >= slice_Wt) {
                            row_offset += stride_Wt;
                            pages_read_in_row = pages_read_in_row - slice_Wt;
                        }
                    }
                }

                // 2. unicast output ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);
                auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
                pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the unicast packet
                pkt_hdr->to_chip_unicast(1);
                tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr);
                noc_async_writes_flushed();
            } else {
                // Otherwise, on the last slice, write it to output buffer
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;
                uint32_t tile_id_start = batch_slice_offset;
                if (!direction) {
                    tiles_read += std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                }
                while (tiles_read < tiles_to_read) {
                    uint32_t num_pages_to_read = 0;
                    if (direction) {
                        num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                    } else {
                        num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                    }
                    cb_wait_front(cb_output_id, tile_granularity);
                    size_t l1_read_addr = get_read_ptr(cb_output_id);

                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        noc_async_write_tile(tile_id_start + tiles_read, output_addrgen, l1_read_addr);
                        l1_read_addr += intermediate_page_size;
                        tiles_read++;
                    }

                    noc_async_writes_flushed();
                    cb_pop_front(cb_output_id, tile_granularity);

                    // Skip the tiles going the other direction
                    if (tiles_read < tiles_to_read) {
                        num_pages_to_read = 0;
                        if (!direction) {
                            num_pages_to_read = std::min((tiles_to_read - tiles_read) / 2, tile_granularity);
                        } else {
                            num_pages_to_read = std::min(tiles_to_read - tiles_read, tile_granularity);
                        }
                        tiles_read += num_pages_to_read;
                    }
                }
                noc_async_write_barrier();

                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem) = 0;

                // 2. mcast half batch ready semaphore
                uint64_t out_ready_sem_noc_addr_in_pkt =
                    safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, batch_ready_sem, 0);
                auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
                pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    out_ready_sem_noc_addr_in_pkt,
                    static_cast<uint16_t>(1),  // increment 1
                    32});
                // Write the mcast packet
                pkt_hdr->to_chip_multicast(
                    tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(ring_size - 1)});
                tt::tt_fabric::fabric_atomic_inc(mux_connection_handle, pkt_hdr);
                noc_async_writes_flushed();
            }

            // Next slice idx
            if (direction) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }
        // Reset the global semaphore before the next batch
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem) < ring_size - 1);
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(batch_ready_sem) = 0;
    }

    tt::tt_fabric::fabric_client_disconnect(mux_connection_handle);
    if (fabric_mux_worker_master) {
        uint64_t dest_addr = get_noc_addr(fabric_mux_x, fabric_mux_y, fabric_mux_termination_address);
        noc_inline_dw_write(dest_addr, tt::tt_fabric::TerminationSignal::IMMEDIATELY_TERMINATE);
    }

    noc_async_write_barrier();
}
