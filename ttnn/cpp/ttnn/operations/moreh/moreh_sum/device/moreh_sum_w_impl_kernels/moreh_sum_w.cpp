// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/matmul.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);

    auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_scaler = tt::CBIndex::c_2;
    constexpr auto cb_mask_w = tt::CBIndex::c_3;
    constexpr auto cb_accum_dst = tt::CBIndex::c_24;
    constexpr auto cb_masked_input = tt::CBIndex::c_25;
    constexpr auto cb_out = tt::CBIndex::c_16;
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    binary_op_init_common(cb_input, cb_scaler, cb_out);

    ckernel::cb_wait_front(cb_scaler, 1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    if (do_mask_w) {
        ckernel::cb_wait_front(cb_mask_w, onetile);
    }

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            // tiles are expected to be coming in in NCHW order (W-contiguous)
            // reducing in W means out[h][0] = sum(w=0..W-1, in[h][w])
            // in this case we just sequentially add to accumulator all the W-tiles in a row
            cb_input = tt::CBIndex::c_0;
            bool is_w_single_tile = (Wt == 1);
            if (!is_w_single_tile) {
                ckernel:: tile_regs_acquire();
                for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                    ckernel::cb_wait_front(cb_input, onetile);
#if defined FP32_DEST_ACC_EN
                    reconfig_data_format(cb_input, cb_scaler);
#endif
                    mm_init_short(cb_input, cb_scaler, false);
                    matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx, false);

                    ckernel::cb_pop_front(cb_input, onetile);
                }
                ckernel:: tile_regs_commit();
                ckernel::cb_reserve_back(cb_accum_dst, onetile);
                ckernel::tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_accum_dst);
#endif
                ckernel:: pack_tile(reduce_dst_idx, cb_accum_dst);
                ckernel::tile_regs_release();
                ckernel::cb_push_back(cb_accum_dst, onetile);
            }

            if (do_mask_w) {
                ckernel:: tile_regs_acquire();
                ckernel::cb_wait_front(cb_input, onetile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_input);
#endif
                copy_tile_to_dst_init_short(cb_input);
                ckernel:: copy_tile(cb_input, 0, reduce_dst_idx);
                ckernel:: copy_tile(cb_mask_w, 0, mask_dst_idx);
                mask_tile_init();
                mask_tile(reduce_dst_idx, mask_dst_idx);
                ckernel:: tile_regs_commit();

                ckernel::cb_reserve_back(cb_masked_input, onetile);
                ckernel::tile_regs_wait();
#if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_masked_input);
#endif
                ckernel:: pack_tile(reduce_dst_idx, cb_masked_input);
                ckernel::tile_regs_release();
                ckernel::cb_push_back(cb_masked_input, onetile);

                ckernel::cb_pop_front(cb_input, onetile);
                cb_input = cb_masked_input;
            }

            ckernel:: tile_regs_acquire();
            ckernel::cb_wait_front(cb_input, onetile);
            if (!is_w_single_tile) {
#if defined FP32_DEST_ACC_EN
                reconfig_data_format_srca(cb_accum_dst);
#endif
                ckernel::cb_wait_front(cb_accum_dst, onetile);
                copy_tile_to_dst_init_short(cb_accum_dst);
                ckernel:: copy_tile(cb_accum_dst, 0, reduce_dst_idx);
            }

#if defined FP32_DEST_ACC_EN
            reconfig_data_format(cb_input, cb_scaler);
#endif
            mm_init_short(cb_input, cb_scaler, false);
            matmul_tiles(cb_input, cb_scaler, 0, 0, reduce_dst_idx, false);
            ckernel:: tile_regs_commit();

            ckernel::cb_reserve_back(cb_out, onetile);
            ckernel::tile_regs_wait();
#if defined FP32_DEST_ACC_EN
            pack_reconfig_data_format(cb_out);
#endif
            ckernel:: pack_tile(reduce_dst_idx, cb_out);
            ckernel::tile_regs_release();
            ckernel::cb_push_back(cb_out, onetile);

            ckernel::cb_pop_front(cb_input, onetile);
            if (!is_w_single_tile) {
                ckernel::cb_pop_front(cb_accum_dst, onetile);
            }
        }
    }

    if (do_mask_w) {
        ckernel::cb_pop_front(cb_mask_w, onetile);
    }
    ckernel::cb_pop_front(cb_scaler, onetile);
}
}  // namespace NAMESPACE
