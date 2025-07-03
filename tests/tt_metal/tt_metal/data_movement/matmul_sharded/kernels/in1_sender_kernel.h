#pragma once

#include "test_config.h"
#include "dataflow_api.h"
#include "risc_common.h"

using namespace TestConfig;

void in1_sender_run(
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

    const uint64_t in1_multicast_data_noc = noc_index == 0
                                                ? get_noc_multicast_addr(phy_x_coord, start_y, phy_x_coord, end_y, 0)
                                                : get_noc_multicast_addr(phy_x_coord, end_y, phy_x_coord, start_y, 0);
    uint64_t in1_mcast_receiver_semaphore_noc_addr =
        in1_multicast_data_noc | (uint64_t)in1_mcast_receiver_semaphore_addr;

    // in1 handler4
    uint64_t l1_write_addr_in1 = 0x80000;
    // copy start address of block, to be used for mcasting
    uint64_t in1_start_address = l1_write_addr_in1;
    uint64_t in1_multicast_start_address =
        in1_multicast_data_noc | in1_start_address;  // This is address where we multicast data

    // Copy in1 block into CB, as the default kernel
    uint32_t sender_id = phy_y_coord - origin_y_coord;
    uint32_t in1_tensor_start_tile_id = sender_id * subblock_c_dim * num_subblocks_c_dim;

    const uint32_t num_of_transactions = num_subblocks_k_dim / subblock_k_dim;
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", in1_block_size_bytes);

    {
        DeviceZoneScopedN("RISCV1");
        for (uint32_t h = 0; h < num_subblocks_k_dim; h += subblock_k_dim) {
            uint32_t in1_tensor_tile_id = in1_tensor_start_tile_id;
            for (uint32_t w = 0; w < num_subblocks_c_dim; w += subblock_c_dim) {
                l1_write_addr_in1 += tile_size;
                in1_tensor_tile_id += subblock_c_dim;
            }
            in1_tensor_start_tile_id += in1_tensor_stride_h_tiles;

            noc_semaphore_wait(in1_mcast_sender_semaphore_addr_ptr, num_of_dests_y - 1);
            noc_semaphore_set(in1_mcast_sender_semaphore_addr_ptr, 0);

            noc_async_write_multicast(
                in1_start_address, in1_multicast_start_address, in1_block_size_bytes, num_of_dests_y);

            // This needs snoop bit enabled
            noc_semaphore_set_multicast(
                in1_mcast_sender_semaphore_valid_addr, in1_mcast_receiver_semaphore_noc_addr, num_of_dests_y);
        }
    }
}
