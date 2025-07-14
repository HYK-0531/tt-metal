// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t tiles_offset = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = tt::CBIndex::c_2;

    const uint32_t input_tile_size = get_tile_size(cb_out);
    const DataFormat input_data_format = get_dataformat(cb_out);
    const InterleavedAddrGenFast<true> s = {
        .bank_base_address = dst_addr, .page_size = input_tile_size, .data_format = input_data_format};

    cb_wait_front(cb_out, 1);
    const auto l1_read_addr = get_read_ptr(cb_out);
    noc_async_write_tile(tiles_offset, s, l1_read_addr);
    noc_async_write_barrier();
    cb_pop_front(cb_out, 1);
}