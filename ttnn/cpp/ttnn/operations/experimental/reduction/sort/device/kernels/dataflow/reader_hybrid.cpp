// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include "debug/dprint.h"

#include <cstdint>
#include <utility>

FORCE_INLINE std::pair<uint32_t, uint32_t> get_core_physical_coordinates(
    const uint32_t core_id, const uint32_t lookup_table_buffer_cb_index, const uint32_t tile_size = 1024) {
    // Initialize as max to indicate invalid coordinates
    uint32_t core_x = 0;
    uint32_t core_y = 0;

    if (2 * core_id >= tile_size) {
        return {core_x, core_y};  // Invalid core ID
    }

    const uint32_t l1_read_addr = get_read_ptr(lookup_table_buffer_cb_index);
    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_read_addr);

    core_x = ptr[core_id * 2];
    core_y = ptr[core_id * 2 + 1];

    return {core_x, core_y};
}

void kernel_main() {
    // Runtime args
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_buffer_addr = get_arg_val<uint32_t>(1);
    const uint32_t physical_core_lookup_table_buffer_addr = get_arg_val<uint32_t>(2);

    // Compile time args
    constexpr uint32_t compute_with_storage_grid_size_x = get_compile_time_arg_val(0);
    constexpr uint32_t compute_with_storage_grid_size_y = get_compile_time_arg_val(1);
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t index_tensor_output_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t physical_core_lookup_table_cb_index = get_compile_time_arg_val(4);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(5) == 1;
    constexpr bool index_tensor_output_is_dram = get_compile_time_arg_val(6) == 1;
    constexpr bool physical_core_lookup_table_is_dram = get_compile_time_arg_val(7) == 1;
    constexpr uint32_t Ht = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t number_of_tiles_per_core = get_compile_time_arg_val(10);
    constexpr uint32_t number_of_cores_used = get_compile_time_arg_val(11);
    constexpr bool ascending = get_compile_time_arg_val(12) == 1;
    const uint32_t semaphore = get_semaphore(get_compile_time_arg_val(13));
    constexpr uint32_t exchange_buffer_cb_index = get_compile_time_arg_val(14);
    constexpr uint32_t exchange_buffer_receive_cb_index = get_compile_time_arg_val(15);

    // Constants
    constexpr uint32_t one_tile = 1;
    const uint16_t core_id = get_absolute_logical_y() * compute_with_storage_grid_size_x + get_absolute_logical_x();
    const uint16_t global_tile_start = core_id * number_of_tiles_per_core;
    const uint16_t global_tile_end = global_tile_start + number_of_tiles_per_core;

    const uint16_t number_of_pairs_processed_by_each_core = number_of_tiles_per_core / 2;
    const uint16_t processing_pair_start = core_id * number_of_pairs_processed_by_each_core;
    const uint16_t processing_pair_end = processing_pair_start + number_of_pairs_processed_by_each_core;
    DPRINT << "READER: Core ID: " << core_id << ", Global Tile Start: " << global_tile_start
           << ", Global Tile End: " << global_tile_end << ", Processing Pair Start: " << processing_pair_start
           << ", Processing Pair End: " << processing_pair_end << ENDL();  // TODO: Remove

    // Sempahore cofig
    auto sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore);
    noc_semaphore_set(sem_ptr, 0);  // Reset the semaphore

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_accessor = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Index tensor config
    const uint32_t index_tensor_output_tile_size_bytes = get_tile_size(index_tensor_output_cb_index);
    const DataFormat index_tensor_output_data_format = get_dataformat(index_tensor_output_cb_index);
    const InterleavedAddrGenFast<index_tensor_output_is_dram> index_tensor_output_accessor = {
        .bank_base_address = index_tensor_buffer_addr,
        .page_size = index_tensor_output_tile_size_bytes,
        .data_format = index_tensor_output_data_format};

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
    DPRINT << "READER: Starting" << ENDL();  // TODO: Remove

    for (uint32_t h = 0; h < Ht; h++) {
        // Read input value data
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_reserve_back(input_tensor_cb_index, one_tile);
            const uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_read_tile(tile_offset, input_tensor_accessor, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(input_tensor_cb_index, one_tile);
        }  // w loop

        uint32_t stages = 0;
        for (uint32_t i = Wt; i > 1; i >>= 1) {
            stages++;
        }
        for (uint32_t stage = 2; stage <= stages; stage++) {
            for (uint32_t sub = stage; sub > 0; sub--) {
                uint32_t sub_dist = 1 << (sub - 1);
                for (uint32_t i = 0; i < Wt; i++) {
                    uint32_t j = i ^ sub_dist;
                    if (j > i) {
                        if (i >= global_tile_start && i < global_tile_end && j >= global_tile_start &&
                            j < global_tile_end) {
                            // NOTHING
                        } else if (i >= global_tile_start && i < global_tile_end) {
                            // J is on a remote core
                            // Get second core id
                            const uint32_t other_core_id = j / number_of_tiles_per_core;
                            const std::pair<uint32_t, uint32_t> remote_core_physical =
                                get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);

                            DPRINT << "1. READER: Processing tiles: " << i << " our core: " << core_id << " and " << j
                                   << " remote core: " << other_core_id << " physical: " << remote_core_physical.first
                                   << " " << remote_core_physical.second << ENDL();
                            // TODO: Swapping tiles
                            cb_wait_front(exchange_buffer_cb_index, one_tile);  // Receiving tile to be send from
                                                                                // compute core cb_reserve_back(
                                                                                //     exchange_buffer_receive_cb_index,
                            //     one_tile);  // Reserving space for tile that we will receive from other core
                            // DPRINT << "     > READER: 1.1" << ENDL();  // TODO: Remove
                            DPRINT << "     > 1 READER: Received tile from compute" << ENDL();  // TODO: Remove
                            // Indicate readiness to the ot her core
                            const uint64_t sem_addr =
                                get_noc_addr(remote_core_physical.first, remote_core_physical.second, semaphore);
                            // noc_semaphore_inc(sem_addr, 1);  // Indicate that we are ready to exchange
                            // DPRINT << "     > READER: 1.2" << ENDL();  // TODO: Remove
                            noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to be ready
                            noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore
                            // DPRINT << "     > READER: 1.3" << ENDL();  // TODO: Remove
                            const auto remote_core_data_ptr = get_write_ptr(exchange_buffer_receive_cb_index);
                            const uint64_t remote_core_noc_addr = get_noc_addr(
                                remote_core_physical.first, remote_core_physical.second, remote_core_data_ptr);
                            const auto data_l1_ptr = get_read_ptr(exchange_buffer_cb_index);
                            noc_async_write(data_l1_ptr, remote_core_noc_addr, input_tensor_tile_size_bytes);
                            noc_async_write_barrier();
                            // DPRINT << "     > READER: 1.4" << ENDL();  // TODO: Remove
                            noc_semaphore_inc(sem_addr, 1);  // Indicate that we finished the exchange
                            // TEN SKONCZYL WYSYLAC
                            DPRINT << "     > READER: 1 Finished sending tile to other core" << ENDL();  // TODO: Remove
                            cb_reserve_back(
                                exchange_buffer_receive_cb_index,
                                one_tile);  // Reserving space for tile that we will receive from other core
                            noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to be ready to exchange
                            noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore
                            noc_semaphore_inc(sem_addr, 1);  // Indicate that we are ready for exchange

                            noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to finish the exchange
                            noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore
                            DPRINT << "     > READER: 1 Received tile from other core" << ENDL();  // TODO: Remove
                            // DPRINT << "     > READER: 1.5" << ENDL();  // TODO: Remove
                            cb_pop_front(exchange_buffer_cb_index, one_tile);  // Pop the tile that we sent
                            cb_pop_front(exchange_buffer_cb_index, one_tile);  // Pop the tile that we sent
                            cb_push_back(
                                exchange_buffer_receive_cb_index,
                                one_tile);  // Push the received tile to the compute kernels
                                            // DPRINT << "     > READER: 1.6" << ENDL();  // TODO: Remove
                        } else if (j >= global_tile_start && j < global_tile_end) {
                            // I is on a remote core
                            // Get second core id
                            const uint32_t other_core_id = i / number_of_tiles_per_core;
                            const std::pair<uint32_t, uint32_t> remote_core_physical =
                                get_core_physical_coordinates(other_core_id, physical_core_lookup_table_cb_index);
                            // TODO: Swapping tiles

                            DPRINT << "2. READER: Processing tiles: " << i << " remote core: " << other_core_id
                                   << " and " << j << " our core: " << core_id
                                   << " physical: " << remote_core_physical.first << " " << remote_core_physical.second
                                   << ENDL();

                            cb_wait_front(
                                exchange_buffer_cb_index, one_tile);  // Receiving tile to be send from compute core
                            cb_reserve_back(
                                exchange_buffer_receive_cb_index,
                                one_tile);  // Reserving space for tile that we will receive from other core
                            DPRINT << "     > READER: 2 Received tile from compute kernel" << ENDL();  // TODO: Remove
                            // DPRINT << "     > READER: 2.1" << ENDL();  // TODO: Remove
                            // Indicate readiness to the other core
                            const uint64_t sem_addr =
                                get_noc_addr(remote_core_physical.first, remote_core_physical.second, semaphore);
                            noc_semaphore_inc(sem_addr, 1);  // Indicate that we are ready to exchange
                            // Add a delay loop using NOPs

                            // DPRINT << "     > READER: 2.2" << ENDL();  // TODO: Remove
                            noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to finish the exchange
                            noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore
                            // DPRINT << "     > READER: 2.3" << ENDL();  // TODO: Remove
                            // TEN SKONCZYL ODBIERAC
                            DPRINT << "     > READER: 2 Finished receiving tile from other core"
                                   << ENDL();                // TODO: Remove
                            noc_semaphore_inc(sem_addr, 1);  // Indicate that we are ready to exchange
                            noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to be confirm
                            noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore

                            const auto remote_core_data_ptr = get_write_ptr(exchange_buffer_receive_cb_index);
                            const uint64_t remote_core_noc_addr = get_noc_addr(
                                remote_core_physical.first, remote_core_physical.second, remote_core_data_ptr);
                            const auto data_l1_ptr = get_read_ptr(exchange_buffer_cb_index);
                            noc_async_write(data_l1_ptr, remote_core_noc_addr, input_tensor_tile_size_bytes);
                            noc_async_write_barrier();
                            // DPRINT << "     > READER: 2.4" << ENDL();  // TODO: Remove
                            noc_semaphore_inc(sem_addr, 1);  // Indicate that we finished the exchange

                            // noc_semaphore_wait(sem_ptr, 1);  // Wait for other kernel to finish
                            // noc_semaphore_set(sem_ptr, 0);   // Reset the semaphore
                            // DPRINT << "     > READER: 2.5" << ENDL();  // TODO: Remove
                            DPRINT << "     > READER: 2. Finished sending tile to other core"
                                   << ENDL();                                  // TODO: Remove
                            cb_pop_front(exchange_buffer_cb_index, one_tile);  // Pop the tile that we sent
                            cb_pop_front(exchange_buffer_cb_index, one_tile);  // Pop the tile that we sent
                            cb_push_back(
                                exchange_buffer_receive_cb_index,
                                one_tile);  // Push the received tile to the compute kernels
                                            // DPRINT << "     > READER: 2.6" << ENDL();  // TODO: Remove
                        }
                    }  // if j > i
                }  // i loop
            }  // sub loop
        }  // stage loop

        DPRINT << "READER: AFTER LOGIC:" << ENDL();  // TODO: Remove
        // Write output index data
        for (uint32_t w = 0; w < number_of_tiles_per_core; w++) {
            cb_wait_front(index_tensor_output_cb_index, one_tile);
            const uint32_t l1_write_addr_index = get_read_ptr(index_tensor_output_cb_index);
            const uint32_t tile_offset = h * Wt + core_id * number_of_tiles_per_core + w;
            noc_async_write_tile(tile_offset, index_tensor_output_accessor, l1_write_addr_index);
            noc_async_write_barrier();
            cb_pop_front(index_tensor_output_cb_index, one_tile);
        }  // Wt loop

    }  // h loop
    DPRINT << "READER: Finished reading and sorting tiles." << ENDL();  // TODO: Remove
}
