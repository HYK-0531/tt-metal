// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <stdint.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/device_pool.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/fabric_edm_packet_header.hpp>
#include <array>
#include <cstddef>
#include <magic_enum/magic_enum.hpp>

#include "device_fixture.hpp"
#include "utils.hpp"
#include "test_lite_fabric_utils.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt_stl/span.hpp>

namespace tt::tt_metal {

TEST_F(DeviceFixture, MmioEthCoreRunLiteFabricWritesSingleEthCore) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    if (devices_.size() != 2) {
        GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    }

    ::tunneling::clear_eth_l1(devices_);

    ::tunneling::MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    ::tunneling::LiteFabricAddrs lite_fabric_addrs;
    auto mmio_program = create_lite_fabric_program(
        desc, ::tunneling::TestConfig{.init_all_eth_cores = false, .init_handshake_only = false}, lite_fabric_addrs);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());

    auto dest_device_id = std::get<0>(tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
        {desc.mmio_device->id(), desc.mmio_eth.value()}));
    tt::tt_metal::IDevice* dest_device = nullptr;
    for (auto device : devices_) {
        if (device->id() == dest_device_id) {
            dest_device = device;
            break;
        }
    }
    ASSERT_TRUE(dest_device != nullptr);

    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program, /*don't wait until this finishes*/ false);

    uint32_t lite_fabric_config_addr = lite_fabric_addrs.lite_fabric_config_addr;
    uint32_t host_interface_addr =
        lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, host_interface);
    uint32_t state_addr = lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, state);

    // Wait until handshake/init sequence is complete
    ::tunneling::wait_on_state(
        lite_fabric_config_addr,
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        ::tunneling::LiteFabricState::READY_FOR_PACKETS);

    // Send Fabric commands
    std::unordered_map<CoreCoord, std::vector<uint32_t>>
        dst_cores_and_expected_data;  // used to readback data for validation
    std::vector<uint32_t> zero_vec(4096 / sizeof(uint32_t), 0);
    auto num_packets = dest_device->compute_with_storage_grid_size().y;
    uint32_t dest_addr = dest_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (uint32_t packet_index = 0; packet_index < num_packets; packet_index++) {
        CoreCoord dst_core(0, packet_index);
        CoreCoord virtual_dst_core = dest_device->virtual_core_from_logical_core(dst_core, CoreType::WORKER);
        uint64_t dst_noc_addr =
            (uint64_t(virtual_dst_core.y) << (36 + 6)) | (uint64_t(virtual_dst_core.x) << 36) | (uint64_t)dest_addr;

        // Clear L1 for test purposes
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            zero_vec.data(),
            zero_vec.size() * sizeof(uint32_t),
            tt_cxy_pair(dest_device->id(), virtual_dst_core),
            dest_addr);

        // Wait until there is space to write in the sender buffer
        ::tunneling::host_lite_fabric_interface_t host_interface;
        do {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                (void*)&host_interface,
                sizeof(::tunneling::host_lite_fabric_interface_t),
                tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
                host_interface_addr);

        } while ((host_interface.sender_host_write_index + 1) % ::tunneling::SENDER_NUM_BUFFERS_ARRAY[0] ==
                 host_interface.sender_fabric_read_index);

        // Header and payload set up
        PACKET_HEADER_TYPE packet_header;
        packet_header.to_chip_unicast(1);
        packet_header.to_noc_unicast_write(tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, 4096);

        dst_cores_and_expected_data[virtual_dst_core] =
            create_random_vector_of_bfloat16(4096, 100, std::chrono::system_clock::now().time_since_epoch().count());

        // Get write address for packet header and payload based on buffer index given each buffer slot is 4096 payload
        // + 32B header
        uint32_t write_address =
            lite_fabric_addrs.base_sender_channel_addr + (host_interface.sender_host_write_index * 4128);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            (void*)&packet_header,
            sizeof(PACKET_HEADER_TYPE),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            write_address);
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            dst_cores_and_expected_data.at(virtual_dst_core).data(),
            dst_cores_and_expected_data.at(virtual_dst_core).size() * sizeof(uint32_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            write_address + sizeof(PACKET_HEADER_TYPE));

        host_interface.sender_host_write_index =
            (host_interface.sender_host_write_index + 1) % ::tunneling::SENDER_NUM_BUFFERS_ARRAY[0];
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            (void*)&host_interface.sender_host_write_index,
            sizeof(uint8_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            host_interface_addr + offsetof(::tunneling::host_lite_fabric_interface_t, sender_host_write_index));
    }

    // Wait for all packets to be sent
    ::tunneling::host_lite_fabric_interface_t host_interface;
    do {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            (void*)&host_interface,
            sizeof(::tunneling::host_lite_fabric_interface_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            host_interface_addr);

    } while (host_interface.sender_host_write_index != host_interface.sender_fabric_read_index);

    // Send the termination signal to the virtual eth core
    uint32_t termination_signal_addr =
        lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, termination_signal);
    uint32_t termination_val = 1;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &termination_val,
        sizeof(uint32_t),
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        termination_signal_addr);
    ::tunneling::wait_on_state(
        lite_fabric_config_addr,
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        ::tunneling::LiteFabricState::TERMINATED);

    tt_metal::detail::WaitProgramDone(desc.mmio_device, mmio_program);

    // for validation check all the dest addresses of the packets and make sure they are what we expect
    std::vector<uint32_t> readback_payload(4096 / sizeof(uint32_t));
    for (const auto& [virtual_core, expected_data] : dst_cores_and_expected_data) {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            readback_payload.data(),
            readback_payload.size() * sizeof(uint32_t),
            tt_cxy_pair(dest_device->id(), virtual_core),
            dest_addr);
        EXPECT_EQ(readback_payload, expected_data)
            << "Payload written to core " << virtual_core.str() << " does not match expected data";
    }
}

TEST_F(DeviceFixture, MmioEthCoreRunLiteFabricReadsSingleEthCore) {
    if (arch_ == tt::ARCH::WORMHOLE_B0) {
        GTEST_SKIP() << "Skipping test for Wormhole B0, as it does not support tunneling yet";
    }
    if (devices_.size() != 2) {
        GTEST_SKIP() << "Only expect to be initializing 1 eth device per MMIO chip. Test should ";
    }

    ::tunneling::clear_eth_l1(devices_);

    ::tunneling::MmmioAndEthDeviceDesc desc;
    get_mmio_device_and_eth_device_to_init(devices_, desc);

    ::tunneling::LiteFabricAddrs lite_fabric_addrs;
    auto mmio_program = create_lite_fabric_program(
        desc, ::tunneling::TestConfig{.init_all_eth_cores = false, .init_handshake_only = false}, lite_fabric_addrs);

    auto virtual_eth_core = desc.mmio_device->ethernet_core_from_logical_core(desc.mmio_eth.value());

    std::cout << "Virtual eth core: " << virtual_eth_core.str() << std::endl;

    auto dest_device_id = std::get<0>(tt::tt_metal::MetalContext::instance().get_cluster().get_connected_ethernet_core(
        {desc.mmio_device->id(), desc.mmio_eth.value()}));
    tt::tt_metal::IDevice* dest_device = nullptr;
    for (auto device : devices_) {
        if (device->id() == dest_device_id) {
            dest_device = device;
            break;
        }
    }
    ASSERT_TRUE(dest_device != nullptr);

    tt_metal::detail::LaunchProgram(desc.mmio_device, mmio_program, /*don't wait until this finishes*/ false);

    uint32_t lite_fabric_config_addr = lite_fabric_addrs.lite_fabric_config_addr;
    uint32_t host_interface_addr =
        lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, host_interface);
    uint32_t state_addr = lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, state);

    // Wait until handshake/init sequence is complete
    ::tunneling::wait_on_state(
        lite_fabric_config_addr,
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        ::tunneling::LiteFabricState::READY_FOR_PACKETS);

    // Send Fabric commands
    auto num_packets = 1;  // dest_device->compute_with_storage_grid_size().y;
    ASSERT_TRUE(num_packets < ::tunneling::RECEIVER_NUM_BUFFERS_ARRAY[0])
        << "This test reads data all at once, need to make sure we don't stall due to not clearing receiver buffer "
           "slots";

    std::vector<std::vector<uint32_t>> all_expected_data(num_packets, std::vector<uint32_t>(1024));
    uint32_t dest_addr = dest_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    for (uint32_t packet_index = 0; packet_index < num_packets; packet_index++) {
        CoreCoord dst_core(0, packet_index);
        CoreCoord virtual_dst_core = dest_device->virtual_core_from_logical_core(dst_core, CoreType::WORKER);
        uint64_t dst_noc_addr =
            (uint64_t(virtual_dst_core.y) << (36 + 6)) | (uint64_t(virtual_dst_core.x) << 36) | (uint64_t)dest_addr;

        // Write random data to the destination core's L1 memory
        std::iota(all_expected_data[packet_index].begin(), all_expected_data[packet_index].end(), 0);
        std::cout << "Writing data to core " << virtual_dst_core.str() << " at address " << std::hex << dest_addr
                  << std::dec << std::endl;

        all_expected_data[packet_index] =
            create_random_vector_of_bfloat16(4096, 100, std::chrono::system_clock::now().time_since_epoch().count());
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            all_expected_data[packet_index].data(),
            all_expected_data[packet_index].size() * sizeof(uint32_t),
            tt_cxy_pair(dest_device->id(), virtual_dst_core),
            dest_addr);

        // Wait until there is space to write in the sender buffer
        ::tunneling::host_lite_fabric_interface_t host_interface;
        do {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                (void*)&host_interface,
                sizeof(::tunneling::host_lite_fabric_interface_t),
                tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
                host_interface_addr);

        } while ((host_interface.sender_host_write_index + 1) % ::tunneling::SENDER_NUM_BUFFERS_ARRAY[0] ==
                 host_interface.sender_fabric_read_index);

        // Header and payload set up
        PACKET_HEADER_TYPE packet_header;
        packet_header.to_chip_unicast(1);
        packet_header.to_noc_read(tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, 4096);

        // Get write address for packet header and payload based on buffer index given each buffer slot is 4096 payload
        // + 32B header
        uint32_t write_address =
            lite_fabric_addrs.base_sender_channel_addr + (host_interface.sender_host_write_index * 4128);
        std::cout << "Writing packet header to " << std::hex << write_address << std::dec << std::endl;
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            (void*)&packet_header,
            sizeof(PACKET_HEADER_TYPE),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            write_address);

        host_interface.sender_host_write_index =
            (host_interface.sender_host_write_index + 1) % ::tunneling::SENDER_NUM_BUFFERS_ARRAY[0];
        tt::tt_metal::MetalContext::instance().get_cluster().write_core(
            (void*)&host_interface.sender_host_write_index,
            sizeof(uint8_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            host_interface_addr + offsetof(::tunneling::host_lite_fabric_interface_t, sender_host_write_index));
    }

    // Wait for all packets to be sent
    ::tunneling::host_lite_fabric_interface_t host_interface;
    do {
        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            (void*)&host_interface,
            sizeof(::tunneling::host_lite_fabric_interface_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            host_interface_addr);

    } while (host_interface.sender_host_write_index != host_interface.sender_fabric_read_index);

    std::vector<uint32_t> readback_payload(4096 / sizeof(uint32_t));
    for (const auto& expected_data : all_expected_data) {
        // Wait until there is data to be read
        ::tunneling::host_lite_fabric_interface_t host_interface;
        do {
            tt::tt_metal::MetalContext::instance().get_cluster().read_core(
                (void*)&host_interface,
                sizeof(::tunneling::host_lite_fabric_interface_t),
                tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
                host_interface_addr);

        } while (host_interface.receiver_host_read_index == host_interface.receiver_fabric_write_index);

        std::cout << "rcvr host read index: " << (uint32_t)host_interface.receiver_host_read_index << std::endl;
        std::cout << "rcvr fabric write index: " << (uint32_t)host_interface.receiver_fabric_write_index << std::endl;

        // Get read address for packet header and payload based on buffer index given each buffer slot is 4096 payload
        // + 32B header
        uint32_t read_address = lite_fabric_addrs.base_receiver_channel_addr +
                                (host_interface.receiver_host_read_index * 4128) + sizeof(PACKET_HEADER_TYPE);
        std::cout << "Reading packet header from " << std::hex << read_address << std::dec << std::endl;

        tt::tt_metal::MetalContext::instance().get_cluster().read_core(
            readback_payload.data(),
            readback_payload.size() * sizeof(uint32_t),
            tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
            read_address);

        EXPECT_EQ(readback_payload, expected_data);
    }

    // Send the termination signal to the virtual eth core
    uint32_t termination_signal_addr =
        lite_fabric_config_addr + offsetof(::tunneling::lite_fabric_config_t, termination_signal);
    uint32_t termination_val = 1;
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        &termination_val,
        sizeof(uint32_t),
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        termination_signal_addr);
    ::tunneling::wait_on_state(
        lite_fabric_config_addr,
        tt_cxy_pair(desc.mmio_device->id(), virtual_eth_core),
        ::tunneling::LiteFabricState::TERMINATED);

    tt_metal::detail::WaitProgramDone(desc.mmio_device, mmio_program);
}

}  // namespace tt::tt_metal
