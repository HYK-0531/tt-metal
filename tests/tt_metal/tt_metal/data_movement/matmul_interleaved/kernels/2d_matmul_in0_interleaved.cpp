#include <stdint.h>

#include "dataflow_api.h"
#include "risc_common.h"
#include "test_config.h"
#include "in0_sender_kernel.h"
#include "in0_receiver_kernel.h"
#include "in1_receiver_kernel.h"
#include "in1_sender_kernel.h"

using namespace TestConfig;

void kernel_main() {
    constexpr uint32_t origin_x_coord = get_compile_time_arg_val(0);
    constexpr uint32_t origin_y_coord = get_compile_time_arg_val(1);
    constexpr uint32_t start_x = get_compile_time_arg_val(2);
    constexpr uint32_t start_y = get_compile_time_arg_val(3);
    constexpr uint32_t end_x = get_compile_time_arg_val(4);
    constexpr uint32_t end_y = get_compile_time_arg_val(5);
    constexpr uint32_t g_mhartid = get_compile_time_arg_val(6);
    constexpr uint32_t test_id = get_compile_time_arg_val(7);

    uint32_t noc_id_reg = NOC_CMD_BUF_READ_REG(0, 0, NOC_CFG(NOC_ID_LOGICAL));
    uint32_t phy_x_coord = noc_id_reg & NOC_NODE_ID_MASK;
    uint32_t phy_y_coord = (noc_id_reg >> NOC_ADDR_NODE_ID_BITS) & NOC_NODE_ID_MASK;

    DeviceTimestampedData("Test id", test_id);

    if (g_mhartid == 0) {
        if (phy_x_coord - origin_x_coord == 0) {
            in0_sender_run(
                origin_x_coord, origin_y_coord, phy_x_coord, phy_y_coord, start_x, start_y, end_x, end_y, g_mhartid);
        } else {
            in0_receiver_run(
                origin_x_coord, origin_y_coord, phy_x_coord, phy_y_coord, start_x, start_y, end_x, end_y, g_mhartid);
        }
    } else if ((g_mhartid == 1)) {
        if (phy_y_coord - origin_y_coord == 0) {
            in1_sender_run(
                origin_x_coord, origin_y_coord, phy_x_coord, phy_y_coord, start_x, start_y, end_x, end_y, g_mhartid);
        } else {
            in1_receiver_run(
                origin_x_coord, origin_y_coord, phy_x_coord, phy_y_coord, start_x, start_y, end_x, end_y, g_mhartid);
        }
    }
}
