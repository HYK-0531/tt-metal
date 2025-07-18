// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "debug/pause.h"
#include "eth_chan_noc_mapping.h"
#include "lite_fabric.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/1d_fabric_transaction_id_tracker.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_stream_regs.hpp"

// taken from fabric_erisc_datamover.cpp ... commonize!
// Forward‐declare the Impl primary template:
template <template <uint8_t> class ChannelType, auto& BufferSizes, typename Seq>
struct ChannelPointersTupleImpl;

// Provide the specialization that actually holds the tuple and `get<>`:
template <template <uint8_t> class ChannelType, auto& BufferSizes, size_t... Is>
struct ChannelPointersTupleImpl<ChannelType, BufferSizes, std::index_sequence<Is...>> {
    std::tuple<ChannelType<BufferSizes[Is]>...> channel_ptrs;

    template <size_t I>
    constexpr auto& get() {
        return std::get<I>(channel_ptrs);
    }
};

// Simplify the “builder” so that make() returns the Impl<…> directly:
template <template <uint8_t> class ChannelType, auto& BufferSizes>
struct ChannelPointersTuple {
    static constexpr size_t N = std::size(BufferSizes);

    static constexpr auto make() {
        return ChannelPointersTupleImpl<ChannelType, BufferSizes, std::make_index_sequence<N>>{};
    }
};

/*
 * Tracks receiver channel pointers (from sender side)
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct OutboundReceiverChannelPointers {
    uint32_t num_free_slots = RECEIVER_NUM_BUFFERS;
    tt::tt_fabric::BufferIndex remote_receiver_buffer_index{0};
    size_t cached_next_buffer_slot_addr = 0;

    FORCE_INLINE bool has_space_for_packet() const { return num_free_slots; }
};

/*
 * Tracks receiver channel pointers (from receiver side). Must call reset() before using.
 */
template <uint8_t RECEIVER_NUM_BUFFERS>
struct ReceiverChannelPointers {
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_sent_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> wr_flush_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> ack_counter;
    tt::tt_fabric::ChannelCounter<RECEIVER_NUM_BUFFERS> completion_counter;
    std::array<uint8_t, RECEIVER_NUM_BUFFERS> src_chan_ids;

    FORCE_INLINE void set_src_chan_id(tt::tt_fabric::BufferIndex buffer_index, uint8_t src_chan_id) {
        src_chan_ids[buffer_index.get()] = src_chan_id;
    }

    FORCE_INLINE uint8_t get_src_chan_id(tt::tt_fabric::BufferIndex buffer_index) const {
        return src_chan_ids[buffer_index.get()];
    }

    FORCE_INLINE void reset() {
        wr_sent_counter.reset();
        wr_flush_counter.reset();
        ack_counter.reset();
        completion_counter.reset();
    }
};

FORCE_INLINE void send_next_data(
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& sender_buffer_channel,
    volatile tunneling::host_lite_fabric_interface_t& host_interface,
    OutboundReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& receiver_buffer_channel,
    bool on_mmio_chip) {
    auto& remote_receiver_buffer_index = outbound_to_receiver_channel_pointers.remote_receiver_buffer_index;
    auto& remote_receiver_num_free_slots = outbound_to_receiver_channel_pointers.num_free_slots;
    constexpr uint32_t sender_txq_id = 0;
    uint32_t src_addr = sender_buffer_channel.get_cached_next_buffer_slot_addr();

    volatile auto* pkt_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(src_addr);
    size_t payload_size_bytes = pkt_header->get_payload_size_including_header();
    uint32_t dest_addr = receiver_buffer_channel.get_cached_next_buffer_slot_addr();
    DPRINT << "S: Sent packet from " << HEX() << src_addr << " to dest_addr: " << HEX() << dest_addr << DEC() << ENDL();
    pkt_header->src_ch_id = 0;

    while (internal_::eth_txq_is_busy(sender_txq_id));
    internal_::eth_send_packet_bytes_unsafe(sender_txq_id, src_addr, dest_addr, payload_size_bytes);

    host_interface.sender_fabric_read_index =
        tt::tt_fabric::wrap_increment<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>(host_interface.sender_fabric_read_index);

    remote_receiver_buffer_index = tt::tt_fabric::BufferIndex{
        tt::tt_fabric::wrap_increment<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>(remote_receiver_buffer_index.get())};
    receiver_buffer_channel.set_cached_next_buffer_slot_addr(
        receiver_buffer_channel.get_buffer_address(remote_receiver_buffer_index));
    sender_buffer_channel.set_cached_next_buffer_slot_addr(sender_buffer_channel.get_buffer_address(
        tt::tt_fabric::BufferIndex{(uint8_t)host_interface.sender_fabric_read_index}));
    remote_receiver_num_free_slots--;
    // update the remote reg
    static constexpr uint32_t packets_to_forward = 1;
    while (internal_::eth_txq_is_busy(sender_txq_id));
    remote_update_ptr_val<tunneling::to_receiver_0_pkts_sent_id, sender_txq_id>(packets_to_forward);
}

FORCE_INLINE void run_sender_channel_step(
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& local_sender_channel,
    volatile tunneling::host_lite_fabric_interface_t& host_interface,
    OutboundReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& outbound_to_receiver_channel_pointers,
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& remote_receiver_channel,
    bool on_mmio_chip) {
    bool receiver_has_space_for_packet = outbound_to_receiver_channel_pointers.has_space_for_packet();
    bool has_unsent_packet = host_interface.sender_host_write_index != host_interface.sender_fabric_read_index;
    bool can_send = receiver_has_space_for_packet && has_unsent_packet;

    DPRINT << "S: host write index: " << (uint32_t)host_interface.sender_host_write_index
           << ", fabric read index: " << (uint32_t)host_interface.sender_fabric_read_index << ENDL();
    DPRINT << "S: Receiver has space for packet: " << (uint32_t)receiver_has_space_for_packet
           << ", has unsent packet: " << (uint32_t)has_unsent_packet << ", can send: " << (uint32_t)can_send << ENDL();

    if (can_send) {
        send_next_data(
            local_sender_channel,
            host_interface,
            outbound_to_receiver_channel_pointers,
            remote_receiver_channel,
            on_mmio_chip);
    }

    // Process COMPLETIONs from receiver
    int32_t completions_since_last_check = get_ptr_val(tunneling::to_sender_0_pkts_completed_id);
    if (completions_since_last_check) {
        outbound_to_receiver_channel_pointers.num_free_slots += completions_since_last_check;
        increment_local_update_ptr_val(tunneling::to_sender_0_pkts_completed_id, -completions_since_last_check);
    }
}

__attribute__((optimize("jump-tables"))) FORCE_INLINE void service_fabric_request(
    tt_l1_ptr PACKET_HEADER_TYPE* const packet_start,
    uint16_t payload_size_bytes,
    uint32_t transaction_id,
    volatile tunneling::lite_fabric_config_t& lite_fabric_config,
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& sender_buffer_channel,
    bool on_mmio_chip) {
    const auto& header = *packet_start;
    uint32_t payload_start_address = reinterpret_cast<size_t>(packet_start) + sizeof(PACKET_HEADER_TYPE);

    tt::tt_fabric::NocSendType noc_send_type = header.noc_send_type;
    if (noc_send_type > tt::tt_fabric::NocSendType::NOC_SEND_TYPE_LAST) {
        __builtin_unreachable();
    }
    switch (noc_send_type) {
        case tt::tt_fabric::NocSendType::NOC_UNICAST_WRITE: {
            const auto dest_address = header.command_fields.unicast_write.noc_address;
            DPRINT << "R: NOC_UNICAST_WRITE dest_address: " << HEX() << dest_address
                   << " payload start address: " << HEX() << payload_start_address << DEC() << ENDL();
            noc_async_write_one_packet_with_trid<false, false>(
                payload_start_address,
                dest_address,
                payload_size_bytes,
                transaction_id,
                tt::tt_fabric::local_chip_data_cmd_buf,
                tt::tt_fabric::edm_to_local_chip_noc,
                tt::tt_fabric::forward_and_local_write_noc_vc);
        } break;

        case tt::tt_fabric::NocSendType::NOC_READ: {
            if (!on_mmio_chip) {
                volatile tunneling::host_lite_fabric_interface_t& host_interface = lite_fabric_config.host_interface;

                const auto src_address =
                    header.command_fields.unicast_write.noc_address;  // didn't rename command_field for read
                uint32_t dst_address = sender_buffer_channel.get_cached_next_buffer_slot_addr();
                uint32_t payload_dst_address = dst_address + sizeof(PACKET_HEADER_TYPE);

                DPRINT << "R: NOC_READ src_address: " << HEX() << src_address << " dst_address: " << payload_dst_address
                       << DEC() << ENDL();
                // copy the header to the dst_address and then offset
                tt_l1_ptr PACKET_HEADER_TYPE* copy_packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(dst_address);
                *copy_packet_header = header;

                noc_async_read(
                    src_address, payload_dst_address, payload_size_bytes, tt::tt_fabric::edm_to_local_chip_noc);
                noc_async_read_barrier(tt::tt_fabric::edm_to_local_chip_noc);

                // update the write pointer in the sender channel... not really host interface but allows next iteration
                // to run
                host_interface.sender_host_write_index =
                    tt::tt_fabric::wrap_increment<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>(
                        host_interface.sender_host_write_index);
            }

        } break;

        default: {
            ASSERT(false);
        } break;
    };
}

// MUST CHECK !is_eth_txq_busy() before calling
FORCE_INLINE void receiver_send_completion_ack(uint8_t src_id) {
    while (internal_::eth_txq_is_busy(receiver_txq_id));
    remote_update_ptr_val<receiver_txq_id>(tunneling::to_sender_0_pkts_completed_id, 1);
}

FORCE_INLINE void run_receiver_channel_step(
    tt::tt_fabric::EthChannelBuffer<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& remote_receiver_channel,
    ReceiverChannelPointers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>& receiver_channel_pointers,
    WriteTransactionIdTracker<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0], tunneling::NUM_TRANSACTION_IDS, 0>&
        receiver_channel_trid_tracker,
    volatile tunneling::lite_fabric_config_t& lite_fabric_config,
    tt::tt_fabric::EthChannelBuffer<tunneling::SENDER_NUM_BUFFERS_ARRAY[0]>& local_sender_channel,
    bool on_mmio_chip) {
    auto pkts_received_since_last_check = get_ptr_val<tunneling::to_receiver_0_pkts_sent_id>();
    auto& wr_sent_counter = receiver_channel_pointers.wr_sent_counter;
    bool unwritten_packets = pkts_received_since_last_check != 0;
    volatile tunneling::host_lite_fabric_interface_t& host_interface = lite_fabric_config.host_interface;

    DPRINT << "R: Has unwritten packets: " << (uint32_t)unwritten_packets
           << ", pkts received since last check: " << pkts_received_since_last_check << ENDL();

    if (unwritten_packets) {
        invalidate_l1_cache();
        auto receiver_buffer_index = wr_sent_counter.get_buffer_index();
        tt_l1_ptr PACKET_HEADER_TYPE* packet_header = const_cast<PACKET_HEADER_TYPE*>(
            remote_receiver_channel.template get_packet_header<PACKET_HEADER_TYPE>(receiver_buffer_index));

        DPRINT << "R: rcvr buffer index " << (uint32_t)receiver_buffer_index << " from addr " << HEX()
               << (uint32_t)(remote_receiver_channel.get_buffer_address(receiver_buffer_index)) << DEC() << ENDL();

        receiver_channel_pointers.set_src_chan_id(receiver_buffer_index, packet_header->src_ch_id);

        uint8_t trid = receiver_channel_trid_tracker.update_buffer_slot_to_next_trid_and_advance_trid_counter(
            receiver_buffer_index);
        // lite fabric tunnel depth is 1 so any fabric cmds being sent here will be writes to/reads from this chip
        DPRINT << "R: pkt header " << (uint32_t)packet_header->payload_size_bytes << " noc addr " << HEX()
               << (uint64_t)packet_header->command_fields.unicast_write.noc_address << DEC() << ENDL();
        service_fabric_request(
            packet_header,
            packet_header->payload_size_bytes,
            trid,
            lite_fabric_config,
            local_sender_channel,
            on_mmio_chip);

        wr_sent_counter.increment();
        // decrement the to_receiver_0_pkts_sent_id stream register by 1 since current packet has been processed.
        increment_local_update_ptr_val<tunneling::to_receiver_0_pkts_sent_id>(-1);
    }

    // flush and completion are fused, so we only need to update one of the counters
    // update completion since other parts of the code check against completion
    auto& completion_counter = receiver_channel_pointers.completion_counter;
    // Currently unclear if it's better to loop here or not...
    bool unflushed_writes = !completion_counter.is_caught_up_to(wr_sent_counter);
    auto receiver_buffer_index = completion_counter.get_buffer_index();
    bool next_trid_flushed = receiver_channel_trid_tracker.transaction_flushed(receiver_buffer_index);
    bool can_send_completion = unflushed_writes && next_trid_flushed;
    if (on_mmio_chip) {
        can_send_completion = can_send_completion &&
                              (((host_interface.receiver_fabric_write_index + 1) %
                                ::tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]) != host_interface.receiver_host_read_index);
    }
    // if constexpr (!ETH_TXQ_SPIN_WAIT_RECEIVER_SEND_COMPLETION_ACK) {
    //     can_send_completion = can_send_completion && !internal_::eth_txq_is_busy(DEFAULT_ETH_TXQ);
    // }
    if (can_send_completion) {
        receiver_send_completion_ack(receiver_channel_pointers.get_src_chan_id(receiver_buffer_index));
        receiver_channel_trid_tracker.clear_trid_at_buffer_slot(receiver_buffer_index);
        completion_counter.increment();
        if (on_mmio_chip) {
            host_interface.receiver_fabric_write_index =
                tt::tt_fabric::wrap_increment<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0]>(
                    host_interface.receiver_fabric_write_index);
        }
    }
}

void kernel_main() {
    size_t arg_idx = 0;
    const uint32_t lite_fabric_config_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_local_sender_0_channel_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_local_sender_channel_0_connection_info_addr = get_arg_val<uint32_t>(arg_idx++);
    const size_t lf_remote_receiver_0_channel_buffer_address = get_arg_val<uint32_t>(arg_idx++);
    const bool on_mmio_chip = get_arg_val<uint32_t>(arg_idx++) == 1;

    DPRINT << "Is mmio chip " << (uint32_t)on_mmio_chip << ENDL();

    const size_t lf_local_sender_channel_0_connection_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    auto lf_sender0_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_arg_val<uint32_t>(arg_idx++));

    constexpr uint32_t channel_buffer_size = 4096 + sizeof(PACKET_HEADER_TYPE);
    static_assert(channel_buffer_size == 4128, "Expected channel buffer size to be 4128B");

    volatile tunneling::lite_fabric_config_t* lite_fabric_config =
        reinterpret_cast<volatile tunneling::lite_fabric_config_t*>(lite_fabric_config_addr);

    volatile tunneling::host_lite_fabric_interface_t& host_interface = lite_fabric_config->host_interface;

    // One send buffer and one receiver buffer
    init_ptr_val<tunneling::to_receiver_0_pkts_sent_id>(0);
    init_ptr_val<tunneling::to_sender_0_pkts_acked_id>(0);
    init_ptr_val<tunneling::to_sender_0_pkts_completed_id>(0);

    auto remote_receiver_channels = tt::tt_fabric::EthChannelBuffers<tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<tunneling::NUM_RECEIVER_CHANNELS>{});

    auto local_sender_channels = tt::tt_fabric::EthChannelBuffers<tunneling::SENDER_NUM_BUFFERS_ARRAY>::make(
        std::make_index_sequence<tunneling::NUM_SENDER_CHANNELS>{});

    const std::array<size_t, MAX_NUM_SENDER_CHANNELS>& local_sender_buffer_addresses = {
        lf_local_sender_0_channel_address};
    const std::array<size_t, tunneling::NUM_RECEIVER_CHANNELS>& remote_receiver_buffer_addresses = {
        lf_remote_receiver_0_channel_buffer_address};

    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_flow_control_semaphores = {
        reinterpret_cast<size_t>(lf_sender0_worker_semaphore_ptr)};
    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_connection_live_semaphore_addresses = {
        lf_local_sender_channel_0_connection_semaphore_addr};

    // use same addr space for host to lite fabric edm connection
    std::array<size_t, tunneling::NUM_SENDER_CHANNELS> local_sender_connection_info_addresses = {
        lf_local_sender_channel_0_connection_info_addr};

    // initialize the remote receiver channel buffers
    remote_receiver_channels.init(
        remote_receiver_buffer_addresses.data(),
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE),
        receiver_channel_base_id);

    // initialize the local sender channel worker interfaces
    local_sender_channels.init(
        local_sender_buffer_addresses.data(),
        channel_buffer_size,
        sizeof(PACKET_HEADER_TYPE),
        tunneling::sender_channel_base_id);

    WriteTransactionIdTracker<tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0], tunneling::NUM_TRANSACTION_IDS, 0>
        receiver_channel_0_trid_tracker;

    auto outbound_to_receiver_channel_pointers =
        ChannelPointersTuple<OutboundReceiverChannelPointers, tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto outbound_to_receiver_channel_pointer_ch0 = outbound_to_receiver_channel_pointers.template get<0>();

    auto receiver_channel_pointers =
        ChannelPointersTuple<ReceiverChannelPointers, tunneling::RECEIVER_NUM_BUFFERS_ARRAY>::make();
    auto receiver_channel_pointers_ch0 = receiver_channel_pointers.template get<0>();
    receiver_channel_pointers_ch0.reset();

    // ------------------------ Done all the initializations ------------------------

    // ------------------------ Do local and neighbour handshake ------------------------
    tunneling::do_init_and_handshake_sequence(lite_fabric_config_addr);
    // ------------------------ Done local and neighbour handshake ------------------------

    lite_fabric_config->state = tunneling::LiteFabricState::READY_FOR_PACKETS;

    // ------------------------ Main loop ------------------------
    // Host adds packets to MMIO eth sender channels which are then forwarded to remote receiver channel on connected
    // chip's eth core Non-MMIO eth cores will only use their sender channel to send back read response packets, when
    // MMIO eth core sees read in the remote receiver channel, it won't service the request but host will read it
    while (lite_fabric_config->termination_signal == 0) {
        invalidate_l1_cache();

        run_sender_channel_step(
            local_sender_channels.template get<0>(),
            host_interface,
            outbound_to_receiver_channel_pointer_ch0,
            remote_receiver_channels.template get<0>(),
            on_mmio_chip);

        run_receiver_channel_step(
            remote_receiver_channels.template get<0>(),
            receiver_channel_pointers_ch0,
            receiver_channel_0_trid_tracker,
            *lite_fabric_config,
            local_sender_channels.template get<0>(),
            on_mmio_chip);
    }

    DPRINT << "Got the termination signal " << ENDL();

    // Lite fabric doesn't necessarily need to use transaction ids
    receiver_channel_0_trid_tracker.all_buffer_slot_transactions_acked();

    // Re-init the noc counters as the noc api used is not incrementing them
    ncrisc_noc_counters_init();

    noc_async_write_barrier();
    noc_async_atomic_barrier();

    // Host sets termination_signal to 1 which exits main loop for MMIO eth cores.
    // MMIO eth cores will increment this and send to connected neighbour which will also exit the main loop and
    // increment + send config back.
    lite_fabric_config->termination_signal++;
    internal_::eth_send_packet<false>(
        0, lite_fabric_config_addr >> 4, lite_fabric_config_addr >> 4, sizeof(tunneling::lite_fabric_config_t) >> 4);

    while (lite_fabric_config->termination_signal != 3) {
        invalidate_l1_cache();
    }
    lite_fabric_config->state = tunneling::LiteFabricState::TERMINATED;
}
