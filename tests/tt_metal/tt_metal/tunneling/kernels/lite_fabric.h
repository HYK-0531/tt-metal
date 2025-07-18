// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tunneling {

// STREAM REGISTER ASSIGNMENT
// senders update this stream
constexpr uint32_t to_receiver_0_pkts_sent_id = 0;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_acked_id = 2;
// receivers updates the reg on this stream
constexpr uint32_t to_sender_0_pkts_completed_id = 7;

constexpr uint32_t NUM_SENDER_CHANNELS = 1;
constexpr uint32_t NUM_RECEIVER_CHANNELS = 1;

constexpr size_t sender_channel_base_id = 0;
constexpr size_t receiver_channel_base_id = NUM_SENDER_CHANNELS;

constexpr uint8_t NUM_TRANSACTION_IDS = 4;

constexpr std::array<size_t, NUM_SENDER_CHANNELS> SENDER_NUM_BUFFERS_ARRAY = {8};

constexpr std::array<size_t, NUM_RECEIVER_CHANNELS> RECEIVER_NUM_BUFFERS_ARRAY = {16};

constexpr size_t VC0_RECEIVER_CHANNEL = 0;

enum LiteFabricState : uint8_t {
    MMIO_ETH_INIT_NEIGHBOUR = 0,
    LOCAL_HANDSHAKE = 1,
    NON_MMIO_ETH_INIT_LOCAL_ETHS = 2,
    NEIGHBOUR_HANDSHAKE = 3,
    DONE_HANDSHAKE = 4,
    READY_FOR_PACKETS = 5,
    TERMINATED = 6,
    UNKNOWN = 7,
};

struct host_lite_fabric_interface_t {
    volatile uint8_t sender_host_write_index = 0;
    volatile uint8_t sender_fabric_read_index = 0;
    volatile uint8_t pad[6] = {0};
} __attribute__((packed));

// need to put this at a 16B aligned address
struct lite_fabric_config_t {
    volatile uint32_t binary_address = 0;     // only used by eths that set up kernels
    volatile uint32_t binary_size_bytes = 0;  // only used by eths that set up kernels
    volatile uint32_t eth_chans_mask = 0;
    volatile uint32_t num_local_eths = 0;
    volatile uint32_t primary_local_handshake = 0;      // where subordinate eths will signal
    volatile uint32_t subordinate_local_handshake = 0;  // where primary eth will signal to subordinates
    volatile uint8_t primary_eth_core_x = 0;
    volatile uint8_t primary_eth_core_y = 0;
    volatile LiteFabricState init_state = LiteFabricState::UNKNOWN;
    volatile uint8_t multi_eth_cores_setup = 1;  // test mode only
    volatile LiteFabricState state = LiteFabricState::UNKNOWN;
    volatile uint8_t pad0[3] = {0};
    volatile uint32_t local_neighbour_handshake = 0;  // this needs to be 16B aligned
    volatile uint32_t pad1[3] = {0};
    volatile uint32_t remote_neighbour_handshake = 0;  // this needs to be 16B aligned
    // non-zero means terminate. Host will write 1 to mmio eth cores to signal termination. mmio eth cores will
    // increment and send to neigbour which in turn does the same all mmio and non-mmio eth cores transition to
    // TERMINATED once termination_signal == 3
    volatile uint32_t termination_signal = 0;
    volatile host_lite_fabric_interface_t
        host_interface;  // this is used by the host to communicate with the lite fabric
} __attribute__((packed));

static_assert(sizeof(lite_fabric_config_t) % 16 == 0, "lite_fabric_config_t must be 16B aligned");
static_assert(
    offsetof(lite_fabric_config_t, local_neighbour_handshake) % 16 == 0,
    "local_neighbour_handshake must be 16B aligned");
static_assert(
    offsetof(lite_fabric_config_t, remote_neighbour_handshake) % 16 == 0,
    "remote_neighbour_handshake must be 16B aligned");

#if defined(KERNEL_BUILD) || defined(FW_BUILD)

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/pause.h"
#include "eth_chan_noc_mapping.h"

FORCE_INLINE void do_init_and_handshake_sequence(uint32_t lite_fabric_config_addr) {
    volatile tunneling::lite_fabric_config_t* lite_fabric_config =
        reinterpret_cast<volatile tunneling::lite_fabric_config_t*>(lite_fabric_config_addr);

    DPRINT << "multi_eth_cores_setup " << (uint32_t)lite_fabric_config->multi_eth_cores_setup << ENDL();

    uint32_t launch_msg_addr = (uint32_t)&(((mailboxes_t*)MEM_AERISC_MAILBOX_BASE)->launch);
    constexpr uint32_t launch_and_go_msg_size_bytes =
        ((sizeof(launch_msg_t) * launch_msg_buffer_num_entries) + sizeof(go_msg_t) + 15) & ~0xF;
    static_assert(launch_and_go_msg_size_bytes % 16 == 0, "Launch and go msg size must be multiple of 16 bytes");
    uint32_t go_msg_addr = launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries);

    constexpr uint32_t total_num_eths = sizeof(eth_chan_to_noc_xy[0]) / sizeof(eth_chan_to_noc_xy[0][0]);
    uint32_t exclude_eth_chan = get_absolute_logical_y();
    uint32_t rt_arg_base_addr = get_arg_addr(0);  // using rt arg space as scratch area for kernel metadata

    // local_neighbour_handshake_addr is where this eth core will check that its neighbour has completed its handshake
    // remote_neighbour_handshake_addr is where this eth core will write value before sending it over link to
    // local_neighbour_handshake_addr
    uint32_t local_neighbour_handshake_addr =
        lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, local_neighbour_handshake);
    uint32_t remote_neighbour_handshake_addr =
        lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, remote_neighbour_handshake);

    DPRINT << "local_neighbour_handshake_addr: " << HEX() << local_neighbour_handshake_addr
           << " remote_neighbour_handshake_addr: " << HEX() << remote_neighbour_handshake_addr << DEC() << ENDL();

    lite_fabric_config->remote_neighbour_handshake = 0xFEEDE145;

    // capture the initial state because it will be changed when sending it to neighbouring eths/subordinate eths
    tunneling::LiteFabricState initial_state = lite_fabric_config->init_state;
    tunneling::LiteFabricState state = initial_state;
    while (state != tunneling::LiteFabricState::DONE_HANDSHAKE) {
        invalidate_l1_cache();
        switch (state) {
            case tunneling::LiteFabricState::MMIO_ETH_INIT_NEIGHBOUR: {
                // first send the rt args and config to the remote core
                // clobber the initial state arg with the state that remote eth core should come up in
                lite_fabric_config->init_state = lite_fabric_config->multi_eth_cores_setup
                                                     ? tunneling::LiteFabricState::NON_MMIO_ETH_INIT_LOCAL_ETHS
                                                     : tunneling::LiteFabricState::NEIGHBOUR_HANDSHAKE;
                internal_::eth_send_packet<false>(
                    0,
                    lite_fabric_config_addr >> 4,
                    lite_fabric_config_addr >> 4,
                    sizeof(tunneling::lite_fabric_config_t) >> 4);
                DPRINT << "Sent lite_fabric_config to " << HEX() << lite_fabric_config_addr << DEC() << ENDL();

                internal_::eth_send_packet<false>(
                    0, rt_arg_base_addr >> 4, rt_arg_base_addr >> 4, 1024 >> 4);  // just send all rt args
                DPRINT << "Sent runtime args to " << HEX() << rt_arg_base_addr << DEC() << ENDL();

                // send the kernel binary
                internal_::eth_send_packet<false>(
                    0,
                    lite_fabric_config->binary_address >> 4,
                    lite_fabric_config->binary_address >> 4,
                    lite_fabric_config->binary_size_bytes >> 4);
                DPRINT << "Sent binary to " << HEX() << lite_fabric_config->binary_address << DEC() << " size is "
                       << lite_fabric_config->binary_size_bytes << ENDL();

                // send launch and go message
                internal_::eth_send_packet<false>(
                    0, launch_msg_addr >> 4, launch_msg_addr >> 4, launch_and_go_msg_size_bytes >> 4);
                DPRINT << "Sent launch/go msg to 0x" << HEX() << launch_msg_addr << DEC() << " of size "
                       << launch_and_go_msg_size_bytes << " go msg addr " << HEX()
                       << (launch_msg_addr + (sizeof(launch_msg_t) * launch_msg_buffer_num_entries)) << DEC() << ENDL();

                state = lite_fabric_config->multi_eth_cores_setup ? tunneling::LiteFabricState::LOCAL_HANDSHAKE
                                                                  : tunneling::LiteFabricState::NEIGHBOUR_HANDSHAKE;
                lite_fabric_config->state = state;
                DPRINT << "going to next state: " << (uint32_t)state << ENDL();
                break;
            }
            case tunneling::LiteFabricState::LOCAL_HANDSHAKE: {
                // go over the eth chan header and do case 0 for all cores using noc writes
                // set additional rt args in the kernel that are the x-y of this core

                uint32_t primary_local_handshake_addr =
                    lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, primary_local_handshake);
                uint32_t subordinate_local_handshake_addr =
                    lite_fabric_config_addr + offsetof(tunneling::lite_fabric_config_t, subordinate_local_handshake);

                DPRINT << "primary_local_handshake_addr: " << HEX() << primary_local_handshake_addr
                       << " subordinate_local_handshake_addr: " << HEX() << subordinate_local_handshake_addr << DEC()
                       << ENDL();

                if (initial_state == tunneling::LiteFabricState::MMIO_ETH_INIT_NEIGHBOUR or
                    initial_state == tunneling::LiteFabricState::NON_MMIO_ETH_INIT_LOCAL_ETHS) {
                    uint32_t remaining_cores = lite_fabric_config->eth_chans_mask;
                    for (uint32_t i = 0; i < total_num_eths; i++) {
                        if (remaining_cores == 0) {
                            break;
                        }
                        if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                            uint64_t dest_handshake_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], subordinate_local_handshake_addr);
                            noc_semaphore_inc(dest_handshake_addr, 1);
                            remaining_cores &= ~(0x1 << i);
                        }
                    }

                    while (lite_fabric_config->primary_local_handshake != lite_fabric_config->num_local_eths - 1) {
                        // wait for the subordinate eth cores to send us a handshake signal
                        invalidate_l1_cache();
                    }

                } else {
                    noc_semaphore_inc(
                        get_noc_addr(
                            lite_fabric_config->primary_eth_core_x,
                            lite_fabric_config->primary_eth_core_y,
                            primary_local_handshake_addr),
                        1);
                    while (lite_fabric_config->subordinate_local_handshake != 1) {
                        // wait for the primary eth core
                        invalidate_l1_cache();
                    }
                }

                state = tunneling::LiteFabricState::NEIGHBOUR_HANDSHAKE;
                lite_fabric_config->state = state;
                break;
            }
            case tunneling::LiteFabricState::NON_MMIO_ETH_INIT_LOCAL_ETHS: {
                // update primary_eth_core_x and primary_eth_core_y with this core's virtual x and y
                lite_fabric_config->init_state = tunneling::LiteFabricState::LOCAL_HANDSHAKE;
                lite_fabric_config->primary_local_handshake = 0;
                lite_fabric_config->subordinate_local_handshake = 0;

                if (lite_fabric_config->multi_eth_cores_setup) {
                    uint32_t remaining_cores = lite_fabric_config->eth_chans_mask;
                    for (uint32_t i = 0; i < total_num_eths; i++) {
                        if (remaining_cores == 0) {
                            break;
                        }
                        if ((remaining_cores & (0x1 << i)) && (exclude_eth_chan != i)) {  // exclude_eth_chan is self
                            uint64_t dest_config_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], lite_fabric_config_addr);
                            uint64_t dest_rt_args_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], rt_arg_base_addr);
                            uint64_t dest_binary_addr = get_noc_addr_helper(
                                eth_chan_to_noc_xy[noc_index][i], lite_fabric_config->binary_address);
                            uint64_t dest_launch_and_go_addr =
                                get_noc_addr_helper(eth_chan_to_noc_xy[noc_index][i], launch_msg_addr);
                            noc_async_write(
                                lite_fabric_config_addr, dest_config_addr, sizeof(tunneling::lite_fabric_config_t));
                            noc_async_write(rt_arg_base_addr, dest_rt_args_addr, 1024);
                            noc_async_write(
                                lite_fabric_config->binary_address,
                                dest_binary_addr,
                                lite_fabric_config->binary_size_bytes);
                            noc_async_write(launch_msg_addr, dest_launch_and_go_addr, launch_and_go_msg_size_bytes);
                            remaining_cores &= ~(0x1 << i);
                        }
                    }
                    noc_async_write_barrier();
                }

                state = tunneling::LiteFabricState::LOCAL_HANDSHAKE;
                lite_fabric_config->state = state;
                break;
            }
            case tunneling::LiteFabricState::NEIGHBOUR_HANDSHAKE: {
                // we can only come into this state if our local handshakes are done or we are initializing one core

                internal_::eth_send_packet<false>(
                    0, remote_neighbour_handshake_addr >> 4, local_neighbour_handshake_addr >> 4, 16 >> 4);

                if (lite_fabric_config->local_neighbour_handshake == 0xFEEDE145) {
                    DPRINT << "done with handshaking" << ENDL();
                    state = tunneling::LiteFabricState::DONE_HANDSHAKE;
                }

                break;
            }
            default: ASSERT(false);
        }
    }

    lite_fabric_config->state = LiteFabricState::DONE_HANDSHAKE;
    DPRINT << "done init" << ENDL();
}

#endif

}  // namespace tunneling
