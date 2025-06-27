#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in0_sender_receiver_run(
    const uint32_t origin_x_coord,
    const uint32_t origin_y_coord,
    const uint32_t phy_x_coord,
    const uint32_t phy_y_coord,
    const uint32_t start_x,
    const uint32_t start_y,
    const uint32_t end_x,
    const uint32_t end_y,
    const uint32_t mhartid) {
    uint32_t sender_id = phy_x_coord - origin_x_coord;
    /* Semaphore setup */
    // Set ur local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    volatile uint32_t* in0_mcast_receiver_semaphore_addr_ptr = (uint32_t*)(in0_mcast_receiver_semaphore_addr);
    // local address that will be atomically incremented by mcast receivers, to know when all receivers are ready  to
    // receive the mcast
    volatile uint32_t* in0_mcast_sender_semaphore_addr_ptr = (uint32_t*)(in0_mcast_sender_semaphore_addr);
    // Semaphore with valid value, used for multicasting
    volatile uint32_t* in0_mcast_sender_semaphore_valid_addr_ptr = (uint32_t*)(in0_mcast_sender_semaphore_valid_addr);
    // Set up local VALID value, to be mcasted to destinations flag address after the data has been mcasted
    in0_mcast_sender_semaphore_valid_addr_ptr[0] =
        1;  // Load const 1 to be used as semaphore valid value sent from sender to receivers
    in0_mcast_sender_semaphore_addr_ptr[0] = 0;

    /* Pre calculate receiver semaphore multicast address*/
    const uint64_t in0_multicast_data_noc = get_noc_multicast_addr(start_x, start_y, end_x, start_y, 0);
    uint64_t in0_mcast_receiver_semaphore_noc_addr =
        in0_multicast_data_noc | (uint64_t)in0_mcast_receiver_semaphore_addr;
    /* Semaphore setup end */
    uint64_t remote_sender_noc_addrs[num_of_dests_x];

    for (uint32_t i = 0; i < num_of_dests_x; i++) {
        uint32_t cur_x = i % (end_x + 1);
        // uint32_t cur_y = i / (end_x+1);
        remote_sender_noc_addrs[i] = get_noc_addr(origin_x_coord + cur_x, phy_y_coord, in0_mcast_sender_semaphore_addr);
    }

    noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, 1);

    /* Reserve output circular buffer */
    uint32_t in0_tensor_current_inner_dim_block_start_addr = 0x80000;

    DeviceTimestampedData("Number of transactions", num_blocks_k_dim);
    DeviceTimestampedData("Transaction size in bytes", in0_block_size_bytes);

    // Main loop for the blocks
    {
        DeviceZoneScopedN("RISCV0");
        uint32_t current_t6 = 0;
        for (uint32_t block = 0; block < num_blocks_k_dim; block++) {
            noc_semaphore_set(in0_mcast_receiver_semaphore_addr_ptr, 0);

            uint32_t block_id = block / num_subblocks_k_dim;
            /* in the sending block mode*/
            if (block_id == sender_id) {
                uint32_t in0_tensor_local_l1_write_addr = 0x80000;
                uint32_t in0_tensor_read_addr = in0_tensor_current_inner_dim_block_start_addr;
                in0_tensor_current_inner_dim_block_start_addr += in0_block_size_bytes;

                // while(*in0_mcast_sender_semaphore_addr_ptr!=num_of_dests_x){}

                // noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, num_of_dests_x);
                noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

                uint64_t in0_multicast_data_addr =
                    in0_multicast_data_noc | in0_tensor_local_l1_write_addr;  // This is address where we multicast data
                noc_async_write_multicast(
                    in0_tensor_read_addr, in0_multicast_data_addr, in0_block_size_bytes, num_of_dests_x - 1);
                // This needs snoop bit enabled
                noc_semaphore_set_multicast_loopback_src(
                    in0_mcast_sender_semaphore_valid_addr, in0_mcast_receiver_semaphore_noc_addr, num_of_dests_x);
            } else /* In the receiving block mode*/
            {
                uint64_t in0_mcast_sender_semaphore_noc_addr = remote_sender_noc_addrs[block_id];
                // Atomic increment source core counter
                // Snoop bit enabled
                noc_semaphore_inc(in0_mcast_sender_semaphore_noc_addr, 1);
            }
            // noc_semaphore_wait(in0_mcast_receiver_semaphore_addr_ptr, 1);
        }
    }
}
