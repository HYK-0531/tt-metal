// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t tiles_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_in = tt::CBIndex::c_0;

    const uint32_t input_tile_size = get_tile_size(cb_in);
    const DataFormat input_data_format = get_dataformat(cb_in);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = src_addr, .page_size = input_tile_size, .data_format = input_data_format};

    cb_reserve_back(cb_in, 1);
    const uint32_t cb_input_addr = get_write_ptr(cb_in);
    noc_async_read_tile(tiles_offset, s, cb_input_addr);
    noc_async_read_barrier();
    cb_push_back(cb_in, 1);
}
