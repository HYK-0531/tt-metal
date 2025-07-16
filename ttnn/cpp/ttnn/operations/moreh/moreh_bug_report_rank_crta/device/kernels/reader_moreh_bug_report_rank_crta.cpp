#include "accessor/tensor_accessor.h"
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t start_id = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);
    uint32_t end_id = start_id + num_tiles;

    constexpr uint32_t input_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t other_is_dram = get_compile_time_arg_val(1);

    const uint32_t input_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t other_addr = get_common_arg_val<uint32_t>(1);

    constexpr uint32_t input_base_cta = 2;
    constexpr uint32_t input_base_crta = 2;
    auto input_args = make_tensor_accessor_args<input_base_cta, input_base_crta>();

    constexpr uint32_t other_base_cta = input_base_cta + input_args.compile_time_args_skip();
    constexpr uint32_t other_base_crta = input_base_crta + input_args.runtime_args_skip();
    auto other_args = make_tensor_accessor_args<other_base_cta, other_base_crta>();

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_other = 1;
    constexpr uint32_t onetile = 1;
    const uint32_t cb_input_tile_size = get_tile_size(cb_input);
    const DataFormat cb_input_data_format = get_dataformat(cb_input);

    const uint32_t cb_other_tile_size = get_tile_size(cb_other);
    const DataFormat cb_other_data_format = get_dataformat(cb_other);

    auto input_tensor_accessor = make_tensor_accessor_from_args(input_args, input_addr, cb_input_tile_size);
    auto other_tensor_accessor = make_tensor_accessor_from_args(other_args, other_addr, cb_other_tile_size);

    // DPRINT << "Start id: " << U32(start_id) << ENDL();
    // DPRINT << "End id: " << U32(end_id) << ENDL();
    // DPRINT << "cb_input_tile_size: " << U32(cb_input_tile_size) << ENDL();
    // DPRINT << "cb_other_tile_size: " << U32(cb_other_tile_size) << ENDL();

    constexpr uint32_t one_tile = 1;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_input, onetile);
        cb_reserve_back(cb_other, onetile);

        uint32_t cb_input_write_ptr = get_write_ptr(cb_input);
        auto input_noc_addr = input_tensor_accessor.get_noc_addr(i);
        noc_async_read_tile(i, input_tensor_accessor, cb_input_write_ptr);

        uint32_t cb_other_write_ptr = get_write_ptr(cb_other);
        auto other_noc_addr = other_tensor_accessor.get_noc_addr(i);
        noc_async_read_tile(i, other_tensor_accessor, cb_other_write_ptr);

        noc_async_read_barrier();
        cb_push_back(cb_input, one_tile);
        cb_push_back(cb_other, one_tile);
    }
}
