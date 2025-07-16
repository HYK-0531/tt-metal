
#include "ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "debug/dprint.h"

inline void print_cb(const uint32_t cb_id, uint32_t tile_id, uint32_t precision = 5, bool print_untilized = false) {
    // print_untilized only has effect in compute kernels, BR and NC (reader and
    // writer) does not have any effects
    for (int32_t r = 0; r < 32; ++r) {
        SliceRange sr = SliceRange{
            .h0 = static_cast<uint8_t>(r), .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = 32, .ws = 1};
        // Print BR (DMVK) (noc0) (reader)
        DPRINT_DATA0({
            DPRINT << SETPRECISION(precision) << (uint)r << " --READ--cb" << U32(cb_id) << "-- "
                   << TileSlice(cb_id, tile_id, sr, true, print_untilized) << ENDL();
        });
        // Print NC (DMVK) (noc1) (writer)
        DPRINT_DATA1({
            DPRINT << SETPRECISION(precision) << (uint)r << " --READ--cb" << U32(cb_id) << "-- "
                   << TileSlice(cb_id, tile_id, sr, true, print_untilized) << ENDL();
        });
        // Print TR0 (CK) (compute)
        DPRINT_UNPACK({
            DPRINT << SETPRECISION(precision) << (uint)r << " --READ--cb" << U32(cb_id) << "-- "
                   << TileSlice(cb_id, tile_id, sr, true, print_untilized) << ENDL();
        });
    }
}

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_input = 0;
    constexpr uint32_t cb_other = 1;
    constexpr uint32_t cb_output = 2;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_input, cb_other, cb_output);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();
        cb_wait_front(cb_input, onetile);
        cb_wait_front(cb_other, onetile);
        // print_cb(cb_input, 0, 5, true);

        add_tiles_init_with_dt(cb_input, cb_other);
        add_tiles(cb_input, cb_other, 0, 0, 0);

        cb_pop_front(cb_input, onetile);
        cb_pop_front(cb_other, onetile);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_output, onetile);
        pack_tile_with_dt(0, cb_output);
        cb_push_back(cb_output, onetile);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
