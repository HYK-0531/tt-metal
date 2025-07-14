// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t cb_in = tt::CBIndex::c_0;
    uint32_t cb_out = tt::CBIndex::c_1;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_1);

    unary_op_init_common(cb_in, cb_out);

    tile_regs_acquire();
    cb_wait_front(cb_in, 1);
    copy_tile_to_dst_init_short(cb_in);
    copy_tile(cb_in, 0, 0);
    cb_pop_front(cb_in, 1);
    tile_regs_commit();

    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
}  // namespace NAMESPACE
