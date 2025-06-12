// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

#include <cstdint>

FORCE_INLINE void multicast_tile(
    const uint32_t start_core_physical_coord_x,
    const uint32_t start_core_physical_coord_y,
    const uint32_t end_core_physical_coord_x,
    const uint32_t end_core_physical_coord_y,
    const uint32_t number_of_dest,
    const uint32_t l1_write_addr,
    const uint32_t tile_size_bytes,
    const uint32_t semaphore_id) {
    // Multicast tile
    const uint64_t tile_global_multicast_addr = get_noc_multicast_addr(
        start_core_physical_coord_x,
        start_core_physical_coord_y,
        end_core_physical_coord_x,
        end_core_physical_coord_y,
        l1_write_addr);
    noc_async_write_multicast(l1_write_addr, tile_global_multicast_addr, tile_size_bytes, number_of_dest);

    // Indicate finish writing
    const uint64_t sempahore_global_multicast_addr = get_noc_multicast_addr(
        start_core_physical_coord_x,
        start_core_physical_coord_y,
        end_core_physical_coord_x,
        end_core_physical_coord_y,
        semaphore_id);
    noc_semaphore_set_multicast(semaphore_id, sempahore_global_multicast_addr, number_of_dest);

    noc_async_write_barrier();
}

void kernel_main() {
    // Runtime args
    const uint32_t start_core_physical_coord_x = get_arg_val<uint32_t>(0);
    const uint32_t start_core_physical_coord_y = get_arg_val<uint32_t>(1);
    const uint32_t end_core_physical_coord_x = get_arg_val<uint32_t>(2);
    const uint32_t end_core_physical_coord_y = get_arg_val<uint32_t>(3);
    const uint32_t start_core_physical_coord_x_residuum = get_arg_val<uint32_t>(4);
    const uint32_t start_core_physical_coord_y_residuum = get_arg_val<uint32_t>(5);
    const uint32_t end_core_physical_coord_x_residuum = get_arg_val<uint32_t>(6);
    const uint32_t end_core_physical_coord_y_residuum = get_arg_val<uint32_t>(7);
    const uint32_t coordinator_to_cores_semaphore_id = get_semaphore(get_arg_val<uint32_t>(8));
    const uint32_t cores_to_coordinator_semaphore_id = get_semaphore(get_arg_val<uint32_t>(9));
    const uint32_t number_of_dest_base = get_arg_val<uint32_t>(10);
    const uint32_t number_of_dest_residuum = get_arg_val<uint32_t>(11);
    const uint32_t input_tensor_buffer_addr = get_arg_val<uint32_t>(12);
    const uint32_t index_loop_count = get_arg_val<uint32_t>(13);

    // Compile time args
    constexpr uint32_t input_tensor_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t Ht = get_compile_time_arg_val(1);
    constexpr uint32_t Wt_input = get_compile_time_arg_val(2);
    constexpr uint32_t Wt_index = get_compile_time_arg_val(3);
    constexpr bool input_tensor_is_dram = get_compile_time_arg_val(4) == 1;

    constexpr uint32_t one_tile = 1;

    // Input tensor config
    constexpr uint32_t input_tensor_tile_size_bytes = get_tile_size(input_tensor_cb_index);
    constexpr DataFormat input_tensor_data_format = get_dataformat(input_tensor_cb_index);
    const InterleavedAddrGenFast<input_tensor_is_dram> input_tensor_addr_ger = {
        .bank_base_address = input_tensor_buffer_addr,
        .page_size = input_tensor_tile_size_bytes,
        .data_format = input_tensor_data_format};

    // Semaphore setup
    volatile tt_l1_ptr uint32_t* semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cores_to_coordinator_semaphore_id);
    noc_semaphore_set(semaphore_ptr, 0);  // Reset the semaphore
    const uint64_t semaphore_global_multicast_addr = get_noc_multicast_addr(
        start_core_physical_coord_x,
        start_core_physical_coord_y,
        end_core_physical_coord_x,
        end_core_physical_coord_y,
        coordinator_to_cores_semaphore_id);

    // Copy input data to output and generate index tiles
    for (uint32_t h = 0; h < Ht; h++) {
        for (uint32_t index_loop = 0; index_loop < index_loop_count; index_loop++) {
            for (uint32_t wi = 0; wi < Wt_input; wi++) {
                cb_reserve_back(input_tensor_cb_index, one_tile);

                const uint32_t l1_input_index_write_addr = get_write_ptr(input_tensor_cb_index);
                noc_async_read_tile(h * Wt_input + wi, input_tensor_addr_ger, l1_input_index_write_addr);
                noc_async_read_barrier();

                //  Wait for all cores to be ready
                noc_semaphore_wait(semaphore_ptr, (number_of_dest_base + number_of_dest_residuum));
                noc_semaphore_set(semaphore_ptr, 0);  // Reset the semaphore

                // Base multicast data
                multicast_tile(
                    start_core_physical_coord_x,
                    start_core_physical_coord_y,
                    end_core_physical_coord_x,
                    end_core_physical_coord_y,
                    number_of_dest_base,
                    l1_input_index_write_addr,
                    input_tensor_tile_size_bytes,
                    coordinator_to_cores_semaphore_id);

                // Residuum multicast data
                if (number_of_dest_residuum > 0) {
                    multicast_tile(
                        start_core_physical_coord_x_residuum,
                        start_core_physical_coord_y_residuum,
                        end_core_physical_coord_x_residuum,
                        end_core_physical_coord_y_residuum,
                        number_of_dest_residuum,
                        l1_input_index_write_addr,
                        input_tensor_tile_size_bytes,
                        coordinator_to_cores_semaphore_id);
                }

                // Reset input buffer
                cb_push_back(input_tensor_cb_index, one_tile);   // Push tile to the writer
                cb_wait_front(input_tensor_cb_index, one_tile);  // Wait for the writer to finish
                cb_pop_front(input_tensor_cb_index, one_tile);   // Remove data from local buffer
            }  // wi loop
        }  // index_loop loop
    }  // h loop
}
