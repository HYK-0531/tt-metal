// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"
#include "debug/pause.h"

#include "cross_core_data_exchange_common.hpp"
#include "sort_dataflow_common.hpp"

#include <cstdint>
#include <utility>

void kernel_main() {
    // Runtime args
    const uint32_t output_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t physical_core_lookup_table_buffer_addr = get_arg_val<uint32_t>(1);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t index_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t value_tensor_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t value_tensor_peer_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t physical_core_lookup_table_cb_index = get_compile_time_arg_val(5);
    constexpr bool value_tensor_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr uint32_t Wt = get_compile_time_arg_val(7);
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(9);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(10);
    const uint32_t sem_exchange_addr = get_semaphore(get_compile_time_arg_val(11));
    constexpr bool is_32_bit_data = get_compile_time_arg_val(12) == 1;
    const uint32_t sem_barrier_addr = get_semaphore(get_compile_time_arg_val(13));
    constexpr uint32_t index_tensor_peer_cb_index = get_compile_time_arg_val(14);
    constexpr uint32_t index_tensor_intermediate_cb_index = get_compile_time_arg_val(15);
    constexpr bool physical_core_lookup_table_is_dram = get_compile_time_arg_val(16) == 1;

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;

    constexpr uint32_t start_core_id = 0;
    constexpr uint32_t leader_core_id = start_core_id;

    const uint32_t index_tile_size_bytes = get_tile_size(index_tensor_intermediate_cb_index);

    // Output tensor config
    const uint32_t value_tensor_tile_size_bytes = get_tile_size(value_tensor_cb_index);
    const DataFormat value_tensor_data_format = get_dataformat(value_tensor_cb_index);
    const InterleavedAddrGenFast<value_tensor_is_dram> output_tensor_accessor = {
        .bank_base_address = output_tensor_buffer_addr,
        .page_size = value_tensor_tile_size_bytes,
        .data_format = value_tensor_data_format};

    // Physical core lookup table config
    constexpr uint32_t physical_core_lookup_table_tile_size_bytes = get_tile_size(physical_core_lookup_table_cb_index);
    constexpr DataFormat physical_core_lookup_table_data_format = get_dataformat(physical_core_lookup_table_cb_index);
    const InterleavedAddrGenFast<physical_core_lookup_table_is_dram> physical_core_lookup_table_accessor = {
        .bank_base_address = physical_core_lookup_table_buffer_addr,
        .page_size = physical_core_lookup_table_tile_size_bytes,
        .data_format = physical_core_lookup_table_data_format};

    // Read lookup table for physical core IDs
    cb_reserve_back(physical_core_lookup_table_cb_index, one_tile);
    const uint32_t physical_core_lookup_table_l1_write_addr = get_write_ptr(physical_core_lookup_table_cb_index);
    uint64_t noc_addr = get_noc_addr(0, physical_core_lookup_table_accessor);
    noc_async_read(noc_addr, physical_core_lookup_table_l1_write_addr, physical_core_lookup_table_tile_size_bytes);
    noc_async_read_barrier();

    // Semaphore setup
    sem_ptr_t sem_self_exchange_ptr = reinterpret_cast<sem_ptr_t>(sem_exchange_addr);

    for (uint32_t h = 0; h < Ht; h++) {
        // Generate input index tiles
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            if (is_32_bit_data) {
                generate_index_tile<uint32_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            } else {
                generate_index_tile<uint16_t>(index_tensor_cb_index, core_id * number_of_tiles_per_core + w);
            }
        }  // w loop

        const uint32_t stages = ilog2(Wt);
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                const uint32_t sub_dist = 1 << (sub - 1);

                const uint32_t i = global_tile_start;
                const uint32_t j = i ^ sub_dist;

                if (!(i >= global_tile_start && i < global_tile_end && j >= global_tile_start && j < global_tile_end)) {
                    // Without this barrier, a faster core (in this scenario core C) could start a new exchange
                    // before its peer has finished the previous one, causing a conflict
                    // on the shared semaphore. For example, with three cores A, B, and C:
                    //  A     B     C
                    //  |     |     |
                    //  E <-> E     |   (A and B exchanging tiles)
                    //  E <-> E     |
                    //  E <-> E     |
                    //  E <---E-----|   (C starts exchange with A)
                    //  X     E     |   (A is now in an invalid state)
                    //  X     E     |
                    //
                    // This barrier ensures all cores reach the same stage before proceeding,
                    // preventing such conflicts.
                    sort_barrier(
                        physical_core_lookup_table_cb_index,
                        sem_barrier_addr,
                        core_id,
                        leader_core_id,
                        number_of_cores_used,
                        start_core_id);

                    const uint32_t other_core_id = j / number_of_tiles_per_core;
                    const std::pair<uint32_t, uint32_t> remote_core_physical =
                        get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);

                    sort_noc_exchange_tiles(
                        index_tensor_intermediate_cb_index,
                        index_tensor_peer_cb_index,
                        number_of_tiles_per_core,
                        index_tile_size_bytes,
                        remote_core_physical.first,
                        remote_core_physical.second,
                        sem_self_exchange_ptr);
                }  // if !(i >= global_tile_start && i < ...
            }  // sub
        }  // stages

        // Write value tensor to DRAM
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_wait_front(value_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr_val = get_read_ptr(value_tensor_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;

            noc_async_write_tile(tile_offset, output_tensor_accessor, l1_write_addr_val);
            noc_async_write_barrier();

            cb_pop_front(value_tensor_cb_index, one_tile);
        }  // Wt loop
    }  // h loop
    cb_push_back(physical_core_lookup_table_cb_index, one_tile);
}
