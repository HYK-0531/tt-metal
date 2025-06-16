// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <sys/types.h>

#include <cstdint>
#include "dataflow_api.h"
#include "reader_pool2d_sharded_common.hpp"

#define ENABLE_DEBUG_PRINT 1

#if ENABLE_DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#endif

/**
 * Pool 2D (Max pool 2D and Avg pool 2D)
 */
void kernel_main() {
    constexpr uint32_t reader_nindices = get_compile_time_arg_val(0);
    constexpr uint32_t window_h = get_compile_time_arg_val(1);
    constexpr uint32_t window_w = get_compile_time_arg_val(2);

    constexpr int32_t pad_w = get_compile_time_arg_val(3);

    // channel size in bytes
    constexpr uint32_t in_nbytes_c = get_compile_time_arg_val(4);

    // input tensor height / width / channels
    constexpr int32_t in_w = get_compile_time_arg_val(5);

    constexpr uint32_t in_c = get_compile_time_arg_val(6);

    constexpr uint32_t split_reader = get_compile_time_arg_val(7);
    constexpr uint32_t reader_id = get_compile_time_arg_val(8);

    constexpr uint32_t bf16_scalar = get_compile_time_arg_val(9);
    constexpr uint32_t bf16_one_u32 = get_compile_time_arg_val(10);
    constexpr uint32_t bf16_init_value = get_compile_time_arg_val(11);

    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(12);
    constexpr uint32_t in_cb_sz = get_compile_time_arg_val(13);
    constexpr uint32_t max_rows_for_reduction = get_compile_time_arg_val(14);
    constexpr uint32_t ceil_pad_w = get_compile_time_arg_val(15);

    constexpr uint32_t TILE_HEIGHT = 32;
    constexpr uint32_t TILE_WIDTH = 32;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;  // hardware can do reduction of 8 tiles at a time
    constexpr uint32_t BYTES_PER_ELEM = in_nbytes_c / in_c;
    constexpr uint32_t MAX_ELE_PER_REDUCTION = 512;  // TILE_WIDTH * 8 * numbytes

    constexpr uint32_t in_cb_id = (reader_id == 1) ? get_compile_time_arg_val(17) : get_compile_time_arg_val(16);
    constexpr uint32_t in_shard_cb_id = get_compile_time_arg_val(18);
    constexpr uint32_t in_reader_indices_cb_id = get_compile_time_arg_val(19);
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(20);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(21);
    constexpr uint32_t interm_reduction_cb_id = get_compile_time_arg_val(22);
    constexpr uint32_t in_one_cb_id = get_compile_time_arg_val(23);
    constexpr uint32_t clear_value_cb_id = get_compile_time_arg_val(24);
    constexpr bool is_avg_pool = (bool)get_compile_time_arg_val(25);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(26);
    constexpr uint32_t config_cb_id = get_compile_time_arg_val(27);
    constexpr uint32_t multi_buffering_factor = get_compile_time_arg_val(28);
    constexpr uint32_t sync_cb_id1 = get_compile_time_arg_val(29);
    constexpr uint32_t sync_cb_id2 = get_compile_time_arg_val(30);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(31);

    constexpr uint32_t in_scalar_cb_id = in_scalar_cb_id_0;

    uint32_t scalar_index = 0;
    uint32_t scalar_start = 0;
    uint32_t scalar_end = 1;
    uint32_t scalar_value = 0;

    constexpr uint32_t window_size_hw = window_h * window_w;
    constexpr uint32_t remaining_elems = window_size_hw % (max_rows_for_reduction - 1);
    constexpr uint32_t interm_reduction_chunks = remaining_elems ? window_size_hw / (max_rows_for_reduction - 1) + 1
                                                                 : window_size_hw / (max_rows_for_reduction - 1);
    // we only need to initialize the in_cb if we will not fill each multibuffering chunk with max_rows worth of data
    constexpr bool need_to_initialize_in_cb = remaining_elems && interm_reduction_chunks <= multi_buffering_factor;
    constexpr uint32_t in_cb_ntiles = in_cb_sz / (TILE_WIDTH * TILE_HEIGHT);  // only use the non-multi buffering size

    // fill the clear cb
    fill_with_val(get_write_ptr(clear_value_cb_id), TILE_HEIGHT * TILE_WIDTH, bf16_init_value);

    constexpr uint32_t bf16_one_u16 = bf16_one_u32 >> 16;
    // initialize buffers
    clear_out_tiles<in_cb_id, clear_value_cb_id>();
    clear_out_tiles<interm_reduction_cb_id, clear_value_cb_id>();
    if constexpr (one_scalar_per_core) {
        fill_with_val(get_write_ptr(in_scalar_cb_id_0), TILE_WIDTH, bf16_scalar >> 16);
    }
    // DPRINT << "READER scalar" << ENDL();
    // tt::data_movement::common::print_bf16_pages(get_read_ptr(in_scalar_cb_id_0), 32, 1);
    if constexpr (is_avg_pool) {
        // for avgpool, we use a one's CB to avoid double division by kernel size for large kernel case.
        fill_with_val(get_write_ptr(in_one_cb_id), TILE_WIDTH, bf16_one_u16);
    }

    // ensure initialization is done before proceeding
    if constexpr (reader_id == 0) {
        cb_push_back(sync_cb_id1, 1);
        if constexpr (split_reader) {
            cb_wait_front(sync_cb_id2, 2);
        }
    } else {
        cb_push_back(sync_cb_id2, 1);
        cb_wait_front(sync_cb_id1, 2);
    }

    const uint32_t in_l1_read_base_addr = get_read_ptr(in_shard_cb_id);
    uint32_t reader_indices_l1_addr = get_read_ptr(in_reader_indices_cb_id);
    volatile tt_l1_ptr uint16_t* reader_indices_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reader_indices_l1_addr);
    uint32_t config_l1_addr;
    volatile tt_l1_ptr uint16_t* config_ptr;

    constexpr uint32_t in_w_padded = in_w + pad_w + ceil_pad_w;

    constexpr uint32_t total_elems_to_reduce = window_h * window_w;
    constexpr bool wide_reduction = in_nblocks_c > 1;
    constexpr uint32_t in_write_inc =
        wide_reduction ? MAX_ELE_PER_REDUCTION : in_nbytes_c;  // in_cb is MAX_ELE_PER_REDUCTION for wide reductions

    if constexpr (!one_scalar_per_core) {
        config_l1_addr = get_read_ptr(config_cb_id);
        config_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);
        scalar_start = config_ptr[3 * scalar_index];
        scalar_value = config_ptr[3 * scalar_index + 1];
        scalar_end = config_ptr[3 * scalar_index + 2];
        scalar_index++;
    }

    // DPRINT << "reader_nindices: " << reader_nindices << ENDL();

    uint32_t out_l1_write_addr = get_write_ptr(out_cb_id);
    for (uint32_t n = 0; n < reader_nindices; ++n) {
        if constexpr (!one_scalar_per_core) {
            // DPRINT << "HIT" << ENDL();
            cb_reserve_back(in_scalar_cb_id, 1);
            while ((n >= scalar_end) && scalar_end != reader_nindices) {
                scalar_start = scalar_end;
                scalar_value = config_ptr[3 * scalar_index + 1];
                scalar_end = config_ptr[3 * scalar_index + 2];
                scalar_index++;
            }
            // We want to fill the scalar CB at most only the fisrt 2 times since the number of pages is 2, only for the
            // intervals [x, y) where y >= x + 3 exactly 2 times and when y < x + 3 only once. When split reader is
            // enabled n takes even or odd values only depennding on the reader id so if the scalar start is even
            // and n is even it will fullfill the first half of the condition n == scalar_start || n
            // == scalar_start + 2. When reader is even and scalar_start is odd or vice versa we will fullfill the
            // second half of the condition n == scalar_start + 1 || n == scalar_start + 3.
            if (n < scalar_end && (n == scalar_start || n == scalar_start + 1)) {
                fill_with_val(get_write_ptr(in_scalar_cb_id), TILE_WIDTH, scalar_value, false);
            }
            cb_push_back(in_scalar_cb_id, 1);
        }

        const uint16_t top_left_local_index = reader_indices_ptr[n];
        const uint64_t in_l1_write_addr_base = get_write_ptr(in_cb_id);
        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            const uint32_t read_bytes = !wide_reduction ? in_nbytes_c
                                        : c_i != in_nblocks_c - 1
                                            ? MAX_ELE_PER_REDUCTION
                                            : (in_c - c_i * TILE_WIDTH * MAX_TILES_PER_REDUCTION) * BYTES_PER_ELEM;
            uint32_t processed_rows = 0;
            fill_with_val(in_l1_write_addr_base, read_bytes / BYTES_PER_ELEM, bf16_init_value);  // reset first row
            for (uint32_t i = 0; i < interm_reduction_chunks; i++) {
                cb_reserve_back(sync_cb_id1, 2);
                for (uint32_t r = 1; r < max_rows_for_reduction; ++r) {
                    uint32_t in_l1_write_addr = in_l1_write_addr_base + r * in_write_inc;
                    if (processed_rows < total_elems_to_reduce) {  // fill with data
                        uint32_t h = processed_rows / window_w;
                        uint32_t w = processed_rows % window_w;
                        const uint32_t stick_offset = top_left_local_index + w + h * in_w_padded;
                        const uint32_t read_offset =
                            in_l1_read_base_addr + (stick_offset * in_nbytes_c + c_i * MAX_ELE_PER_REDUCTION);
                        noc_async_read(get_noc_addr(read_offset), in_l1_write_addr, read_bytes);
                    } else if (is_avg_pool) {  // fill with padding
                        fill_with_val(in_l1_write_addr, read_bytes / BYTES_PER_ELEM, bf16_init_value);
                    }
                    processed_rows++;
                }
                noc_async_read_barrier();
                // DPRINT << "READER chunk i: " << i << ENDL();
                // tt::data_movement::common::print_bf16_pages(in_l1_write_addr_base, in_write_inc / 2, 32);
                cb_push_back(sync_cb_id1, 2);
            }
            cb_wait_front(sync_cb_id2, 2);

            // write the final result to the output buffer
            noc_async_read(get_noc_addr(in_l1_write_addr_base), out_l1_write_addr, read_bytes);  // write the first row
            noc_async_read_barrier();
            // DPRINT << "READER output:" << ENDL();
            // tt::data_movement::common::print_bf16_pages(out_l1_write_addr, read_bytes / 2, 1);
            out_l1_write_addr += read_bytes;

            cb_pop_front(sync_cb_id2, 2);
        }
    }
}  // kernel_main()
