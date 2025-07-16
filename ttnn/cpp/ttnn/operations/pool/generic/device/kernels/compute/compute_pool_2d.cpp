// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/cb_api.h"

#define DEBUG_PRINT 0

#if DEBUG_PRINT == 1
#include "debug/dprint.h"
#include "debug/dprint_pages.h"
#include "debug/dprint_tensix.h"
#endif

namespace NAMESPACE {

void MAIN {
    // NOTE: here it is assumed that in_ntiles_hw == 1. General cases not handled yet. When ntiles_hw > 1 the large
    // kernel is called
    constexpr uint32_t in_ntiles_c = get_compile_time_arg_val(0);
    constexpr uint32_t window_size_hw = get_compile_time_arg_val(1);

    constexpr uint32_t split_reader = get_compile_time_arg_val(2);
    constexpr uint32_t multi_buffering_factor = get_compile_time_arg_val(3);

    constexpr uint32_t nsticks_per_core_by_nblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in_c = get_compile_time_arg_val(5);
    constexpr uint32_t in_nblocks_c = get_compile_time_arg_val(6);
    constexpr uint32_t max_rows_for_reduction = get_compile_time_arg_val(7);

    constexpr uint32_t in_cb_id_0 = get_compile_time_arg_val(8);
    constexpr uint32_t in_cb_id_1 = get_compile_time_arg_val(9);   // for split reader
    constexpr uint32_t ones_cb_id = get_compile_time_arg_val(10);  // cb with all ones for avg pool
    constexpr uint32_t in_scalar_cb_id_0 = get_compile_time_arg_val(11);
    constexpr uint32_t in_scalar_cb_id_1 = get_compile_time_arg_val(12);
    constexpr uint32_t out_cb_id = get_compile_time_arg_val(13);
    constexpr bool one_scalar_per_core = get_compile_time_arg_val(14);
    constexpr uint32_t fp32_scalar = get_compile_time_arg_val(15);
    constexpr uint32_t sync_cb_id = get_compile_time_arg_val(16);

    constexpr bool is_partial_tile = in_c < 32;
    static_assert((!is_partial_tile || (in_c == 16)), "Partial tile must have c_dim 16");
    constexpr uint32_t num_faces_in_input_tile = is_partial_tile ? 1 : max_rows_for_reduction < 32 ? 2 : 4;
    constexpr uint32_t num_faces_in_output_tile = is_partial_tile ? 1 : 2;
    constexpr uint32_t num_out_rows = 1;

    constexpr bool is_avg_pool = REDUCE_OP == PoolType::SUM;
    // average pool with large kernels requires fp32 accumulation so we can only reduce 4 tiles at a time,
    // otherwise we can reduce 8 tiles at a time.
    constexpr bool is_large_kernel = window_size_hw > max_rows_for_reduction;
    constexpr uint32_t MAX_TILES_PER_REDUCTION = (is_avg_pool && is_large_kernel) ? 4 : 8;
    constexpr uint32_t max_tiles_per_iter =
        in_ntiles_c < MAX_TILES_PER_REDUCTION ? in_ntiles_c : MAX_TILES_PER_REDUCTION;
    constexpr uint32_t partial_iter_output_tiles =
        in_ntiles_c % MAX_TILES_PER_REDUCTION == 0 ? max_tiles_per_iter : in_ntiles_c % MAX_TILES_PER_REDUCTION;

    static_assert(REDUCE_OP == PoolType::MAX || REDUCE_OP == PoolType::SUM, "Only supports REDUCE_OP = MAX or Sum");
    constexpr bool neginf_srca_maxpool = (REDUCE_OP == PoolType::MAX) ? true : false;
    constexpr bool zero_srca_avgpool = (REDUCE_OP == PoolType::SUM) ? true : false;

    constexpr uint32_t face_r_dim = window_size_hw < 16 ? window_size_hw : 16;
    tilizeA_B_reduce_init<neginf_srca_maxpool, zero_srca_avgpool>(
        in_cb_id_0, ones_cb_id, max_tiles_per_iter, out_cb_id, num_faces_in_input_tile, face_r_dim);
    pack_untilize_dest_init<max_tiles_per_iter>(out_cb_id, num_out_rows, num_faces_in_output_tile);

    constexpr uint32_t remaining_elems = window_size_hw % max_rows_for_reduction;
    constexpr uint32_t interm_reduction_chunks =
        remaining_elems ? window_size_hw / max_rows_for_reduction + 1 : window_size_hw / max_rows_for_reduction;

    // wait for initialization to complete
    cb_wait_front(ones_cb_id, 1);
    uint32_t multi_buffer_offset_0 = 0;
    uint32_t multi_buffer_offset_1 = 0;

    for (uint32_t n = 0; n < nsticks_per_core_by_nblocks; ++n) {
        const bool reader0 = !(split_reader && (n & 0x1));
        const uint32_t curr_scalar_cb_id = (!reader0 && !one_scalar_per_core) ? in_scalar_cb_id_1 : in_scalar_cb_id_0;
        const uint32_t curr_in_cb_id = !reader0 ? in_cb_id_1 : in_cb_id_0;
        uint32_t fp32_scalar_var = 0;
        if constexpr (!one_scalar_per_core) {
            cb_wait_front(curr_scalar_cb_id, 1);

            // sync PACK and UNPACK
            cb_reserve_back(sync_cb_id, 1);
            cb_push_back(sync_cb_id, 1);
            cb_wait_front(sync_cb_id, 1);
            cb_pop_front(sync_cb_id, 1);

            volatile uint32_t* fp32_scalar_ptr;
            cb_get_tile(curr_scalar_cb_id, 0, &fp32_scalar_ptr);
            fp32_scalar_ptr += reader0 ? multi_buffer_offset_0 : multi_buffer_offset_1;
            uint32_t pointer_cast = (uint32_t)fp32_scalar_ptr;
            fp32_scalar_var = fp32_scalar_ptr[4];  // value from get tile is offset by 4 elements
            // DPRINT << "GET TILE value: " << fp32_scalar_var << " from: " << pointer_cast + 16 << ENDL();
            cb_release_tile(curr_scalar_cb_id);

            if (reader0) {
                multi_buffer_offset_0 += 1;
                if (multi_buffer_offset_0 == multi_buffering_factor) {
                    multi_buffer_offset_0 = 0;
                }
            } else {
                multi_buffer_offset_1 += 1;
                if (multi_buffer_offset_1 == multi_buffering_factor) {
                    multi_buffer_offset_1 = 0;
                }
            }
        }

        for (uint32_t c_i = 0; c_i < in_nblocks_c; c_i++) {
            bool last_c_block = c_i == in_nblocks_c - 1;
            uint32_t tiles_to_reduce = last_c_block ? partial_iter_output_tiles : max_tiles_per_iter;
            tile_regs_acquire();
            for (uint32_t chunk = 0; chunk < interm_reduction_chunks; chunk++) {
                cb_wait_front(curr_in_cb_id, 1);
                unpack_tilizeA_B_block<neginf_srca_maxpool, true, false, zero_srca_avgpool>(
                    curr_in_cb_id,
                    ones_cb_id,
                    max_tiles_per_iter,
                    0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/,
                    num_faces_in_input_tile,
                    face_r_dim);
                for (uint32_t math_tile_idx = 0; math_tile_idx < max_tiles_per_iter; ++math_tile_idx) {
                    reduce_tile_math(math_tile_idx, num_faces_in_input_tile);
                    if (chunk == interm_reduction_chunks - 1) {
                        // 3x3 kernel -> 1/9 = 1038323257
                        // 9x9 kernel -> 1/81 = 1011500424
                        if constexpr (one_scalar_per_core) {
                            mul_unary_tile(math_tile_idx, fp32_scalar);
                        } else {
                            mul_unary_tile(math_tile_idx, fp32_scalar_var);
                        }
                    }
                }
                cb_pop_front(curr_in_cb_id, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            if (last_c_block) {
                pack_untilize_dest<partial_iter_output_tiles>(out_cb_id, 1, 0, num_out_rows, num_faces_in_output_tile);
                cb_push_back(out_cb_id, partial_iter_output_tiles);
            } else {
                pack_untilize_dest<max_tiles_per_iter>(out_cb_id, 1, 0, num_out_rows, num_faces_in_output_tile);
                cb_push_back(out_cb_id, max_tiles_per_iter);
            }
            tile_regs_release();
        }
        if constexpr (!one_scalar_per_core) {
            cb_pop_front(curr_scalar_cb_id, 1);
        }
    }
}

}  // namespace NAMESPACE
