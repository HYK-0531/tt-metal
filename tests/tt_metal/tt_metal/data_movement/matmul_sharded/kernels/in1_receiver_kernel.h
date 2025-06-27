#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in1_receiver_run(
    const uint32_t origin_x_coord,
    const uint32_t origin_y_coord,
    const uint32_t phy_x_coord,
    const uint32_t phy_y_coord,
    const uint32_t start_x,
    const uint32_t start_y,
    const uint32_t end_x,
    const uint32_t end_y,
    const uint32_t mhartid) {
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in1_mcast_receiver_semaphore_addr_ptr = (uint32_t*)(in1_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready  to
    // receive the mcast
    volatile uint32_t* in1_mcast_sender_semaphore_addr_ptr = (uint32_t*)(in1_mcast_sender_semaphore_addr);
    // Semaphore with valid value, used for multicasting
    volatile uint32_t* in1_mcast_sender_semaphore_valid_addr_ptr = (uint32_t*)(in1_mcast_sender_semaphore_valid_addr);
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in1_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    in1_mcast_sender_semaphore_addr_ptr[0] = 0;

    uint64_t in1_mcast_sender_semaphore_noc_addr =
        get_noc_addr(phy_x_coord, origin_y_coord, in1_mcast_sender_semaphore_addr);

    {
        // DeviceZoneScopedN("RISCV1");
        for (uint32_t h = 0; h < num_subblocks_k_dim; h += subblock_k_dim) {
            // Set in1 semaphore value to INVALID
            noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, 0);

            // Atomic increment source core counter
            noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

            // wait on in1 semaphore value to become VALID (set by mcast sender after it multicasts data)
            // noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, 1);
        }
    }
}
