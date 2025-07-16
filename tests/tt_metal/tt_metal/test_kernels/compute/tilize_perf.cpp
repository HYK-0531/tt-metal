// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"

namespace NAMESPACE {

inline void bar_sync() {
    volatile std::uint32_t* base_address = (std::uint32_t*)MEM_LLK_DEBUG_BASE;

    UNPACK((base_address[1] = 1));
    MATH((base_address[2] = 2));
    PACK((base_address[3] = 3));
    while (base_address[1] != 1) {
        asm("nop");
    }
    while (base_address[2] != 2) {
        asm("nop");
    }
    while (base_address[3] != 3) {
        asm("nop");
    }
    UNPACK((base_address[5] = 5));
    MATH((base_address[6] = 6));
    PACK((base_address[7] = 7));
    while (base_address[5] != 5) {
        asm("nop");
    }
    while (base_address[6] != 6) {
        asm("nop");
    }
    while (base_address[7] != 7) {
        asm("nop");
    }
    UNPACK((base_address[1] = 0));
    MATH((base_address[2] = 0));
    PACK((base_address[3] = 0));
    while (base_address[1] != 0) {
        asm("nop");
    }
    while (base_address[2] != 0) {
        asm("nop");
    }
    while (base_address[3] != 0) {
        asm("nop");
    }
    UNPACK((base_address[5] = 0));
    MATH((base_address[6] = 0));
    PACK((base_address[7] = 0));
}

void MAIN {
    uint32_t ct_dim = get_compile_time_arg_val(0);

    uint32_t loop_factor = 1024;

    uint64_t start = 0;
    uint64_t end = 0;

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);

#ifndef FAST_TILIZE
    tilize_init(tt::CBIndex::c_0, ct_dim, tt::CBIndex::c_16);
#else
    fast_tilize_init(tt::CBIndex::c_0, ct_dim, tt::CBIndex::c_16);
#endif

    cb_wait_front(tt::CBIndex::c_0, ct_dim);
    cb_reserve_back(tt::CBIndex::c_16, ct_dim);

    bar_sync();
    start = read_wall_clock();
    for (uint32_t b = 0; b < loop_factor; ++b) {
#ifndef FAST_TILIZE
        tilize_block(tt::CBIndex::c_0, ct_dim, tt::CBIndex::c_16);
#else
        fast_tilize_block(tt::CBIndex::c_0, ct_dim, tt::CBIndex::c_16);
#endif
    }
    tensix_sync();
    end = read_wall_clock();

    DPRINT << start << " " << end << ENDL();

    cb_pop_front(tt::CBIndex::c_0, ct_dim);
    cb_push_back(tt::CBIndex::c_16, ct_dim);

#ifndef FAST_TILIZE
    tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
#else
    fast_tilize_uninit(tt::CBIndex::c_0, tt::CBIndex::c_16);
#endif
}
}  // namespace NAMESPACE
