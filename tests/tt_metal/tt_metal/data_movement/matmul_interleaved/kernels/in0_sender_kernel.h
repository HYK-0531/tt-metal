#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in0_sender_run(
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

    const uint64_t in0_multicast_data_noc = noc_index == 0
                                                ? get_noc_multicast_addr(start_x, phy_y_coord, end_x, phy_y_coord, 0)
                                                : get_noc_multicast_addr(end_x, phy_y_coord, start_x, phy_y_coord, 0);
    uint64_t in0_mcast_receiver_semaphore_noc_addr =
        in0_multicast_data_noc | (uint64_t)in0_mcast_receiver_semaphore_addr;

    // in0 handler4
    uint32_t in0_read_address = 0x80000;
    uint32_t in0_write_address = 0x80000;
    uint64_t in0_multicast_start_address =
        in0_multicast_data_noc | in0_write_address;  // This is address where we multicast data

    const uint32_t num_of_transactions = num_subblocks_k_dim / subblock_k_dim;
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", in0_block_size_bytes);

    {
        DeviceZoneScopedN("RISCV0");
        for (uint32_t h = 0; h < num_subblocks_k_dim; h += subblock_k_dim) {
            noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr, num_of_dests_x - 1);
            noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);
            noc_async_write_multicast(
                in0_read_address, in0_multicast_start_address, in0_block_size_bytes, num_of_dests_x);

            // This needs snoop bit enabled
            noc_semaphore_set_multicast(
                in0_mcast_sender_semaphore_valid_addr, in0_mcast_receiver_semaphore_noc_addr, num_of_dests_x);

            in0_read_address += in0_block_size_bytes;
            in0_multicast_start_address += in0_block_size_bytes;
        }
    }
}
