// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>

#include "dataflow_api.h"
#include "dataflow_api_addrgen.h"
#include "debug/dprint.h"
#include "hostdevcommon/kernel_structs.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr bool in0_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr bool in1_is_dram = get_compile_time_arg_val(3) == 1;

#ifdef IN0_SHARDED
    cb_reserve_back(cb_id_in0, num_tiles);
    cb_push_back(cb_id_in0, num_tiles);
#else
    const InterleavedAddrGenFast<in0_is_dram> s0 = {
        .bank_base_address = src0_addr,
        .page_size = get_tile_size(cb_id_in0),
        .data_format = get_dataformat(cb_id_in0),
    };
#endif
#ifdef IN1_SHARDED
    cb_reserve_back(cb_id_in1, num_tiles);
    cb_push_back(cb_id_in1, num_tiles);
#else
    const InterleavedAddrGenFast<in1_is_dram> s1 = {
        .bank_base_address = src1_addr,
        .page_size = get_tile_size(cb_id_in1),
        .data_format = get_dataformat(cb_id_in1),
    };
#endif

#if !(defined IN0_SHARDED && defined IN1_SHARDED)
    // won't work if only one input is sharded!
    uint32_t src0_start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t src1_start_tile_id = get_arg_val<uint32_t>(3);

    uint32_t src0_tile_id = src0_start_tile_id;
    uint32_t src1_tile_id = src1_start_tile_id;
    constexpr uint32_t onetile = 1;
    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_reserve_back(cb_id_in0, onetile);
        cb_reserve_back(cb_id_in1, onetile);

        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        noc_async_read_tile(src0_tile_id, s0, l1_write_addr_in0);
        noc_async_read_tile(src1_tile_id, s1, l1_write_addr_in1);
        noc_async_read_barrier();

        cb_push_back(cb_id_in0, onetile);
        cb_push_back(cb_id_in1, onetile);

        src0_tile_id += onetile;
        src1_tile_id += onetile;
    }
#endif
}
