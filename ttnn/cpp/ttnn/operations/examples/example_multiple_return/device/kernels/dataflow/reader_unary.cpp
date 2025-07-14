// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr1 = get_arg_val<uint32_t>(0);
    uint32_t src_addr2 = get_arg_val<uint32_t>(1);
    const uint32_t tiles_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in1 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in2 = tt::CBIndex::c_1;

    const uint32_t input_tile_size = get_tile_size(cb_in1);
    const DataFormat input_data_format = get_dataformat(cb_in1);
    const InterleavedAddrGenFast<true> s1 = {
        .bank_base_address = src_addr1, .page_size = input_tile_size, .data_format = input_data_format};

    const InterleavedAddrGenFast<true> s2 = {
        .bank_base_address = src_addr2, .page_size = input_tile_size, .data_format = input_data_format};

    cb_reserve_back(cb_in1, 1);
    {
        const uint32_t cb_input_addr = get_write_ptr(cb_in1);
        noc_async_read_tile(tiles_offset, s1, cb_input_addr);
        noc_async_read_barrier();
    }
    cb_push_back(cb_in1, 1);

    cb_reserve_back(cb_in2, 1);
    {
        const uint32_t cb_input_addr = get_write_ptr(cb_in2);
        noc_async_read_tile(tiles_offset, s2, cb_input_addr);
        noc_async_read_barrier();
    }
    cb_push_back(cb_in2, 1);
}