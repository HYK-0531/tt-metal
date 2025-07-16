
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t end_id = start_id + num_tiles;

    constexpr bool output_is_dram = get_compile_time_arg_val(0) == 1;

    const uint32_t output_addr = get_common_arg_val<uint32_t>(0);

    constexpr uint32_t cb_output = 2;
    constexpr uint32_t onetile = 1;

    constexpr uint32_t output_base_cta = 1;
    constexpr uint32_t output_base_crta = 1;
    auto output_args = make_tensor_accessor_args<output_base_cta, output_base_crta>();

    // single-tile ublocks
    const uint32_t cb_output_tile_size = get_tile_size(cb_output);
    const DataFormat data_format = get_dataformat(cb_output);

    auto output_tensor_accessor = make_tensor_accessor_from_args(output_args, output_addr, cb_output_tile_size);

    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_output, onetile);
        uint32_t cb_output_read_ptr = get_read_ptr(cb_output);
        auto noc_addr = output_tensor_accessor.get_noc_addr(i);
        noc_async_write(cb_output_read_ptr, noc_addr, cb_output_tile_size);
        noc_async_write_barrier();
        cb_pop_front(cb_output, onetile);
    }
}
