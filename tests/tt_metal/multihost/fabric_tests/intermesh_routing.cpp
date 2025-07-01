// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <gtest/gtest.h>
#include <stdint.h>
#include <cstddef>
#include <cstdlib>
#include <vector>

#include "multihost_fabric_fixtures.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/fabric.hpp>

#include <random>
#include <algorithm> 

namespace tt::tt_fabric {
namespace fabric_router_tests::multihost {

// ========= Data-Movement Tests for 2 Host, 1 T3K bringup machine  =========

// TEST_F(InterMesh2x4Fabric2DFixture, RandomizedInterMeshUnicast) {
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::RandomizedInterMeshUnicast(this);
//     }
// }

// TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshEastMulticast) {
//     std::vector<FabricNodeId> mcast_req_nodes = {
//         FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 2)};
//     std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 0)};
//     std::vector<McastRoutingInfo> routing_info = {
//         McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1}};
//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 3)}, {FabricNodeId(MeshId{1}, 1)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
//     }
// }

// TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshSouthMulticast) {
//     std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 0), FabricNodeId(MeshId{0}, 1)};
//     std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 1)};
//     std::vector<McastRoutingInfo> routing_info = {
//         McastRoutingInfo{.mcast_dir = RoutingDirection::S, .num_mcast_hops = 1}};
//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 2)}, {FabricNodeId(MeshId{1}, 3)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
//     }
// }

// TEST_F(InterMesh2x4Fabric2DFixture, MultiMeshNorthMulticast) {
//     std::vector<FabricNodeId> mcast_req_nodes = {FabricNodeId(MeshId{0}, 3), FabricNodeId(MeshId{0}, 3)};
//     std::vector<FabricNodeId> mcast_start_nodes = {FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 3)};
//     std::vector<McastRoutingInfo> routing_info = {
//         McastRoutingInfo{.mcast_dir = RoutingDirection::N, .num_mcast_hops = 1}};
//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 0)}, {FabricNodeId(MeshId{1}, 1)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 2], mcast_start_nodes[i % 2], routing_info, mcast_group_node_ids[i % 2]);
//     }
// }

// ========= Data-Movement Tests for 2 Loudboxes with Intermesh Connections  =========

// TEST_F(InterMeshDual2x4Fabric2DFixture, RandomizedInterMeshUnicast) {
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::RandomizedInterMeshUnicast(this);
//     }
// }

// TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_Multicast) {
//     std::vector<FabricNodeId> mcast_req_nodes = {
//         FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
//     std::vector<FabricNodeId> mcast_start_nodes = {
//         FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
//     std::vector<McastRoutingInfo> routing_info = {
//         McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
//         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2)},
//         {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)},
//         {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6)},
//         {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info, mcast_group_node_ids[i % 4]);
//     }
// }

// TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_MultiHopMulticast) {
//     std::vector<FabricNodeId> mcast_req_nodes = {
//         FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
//     std::vector<FabricNodeId> mcast_start_nodes = {
//         FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6)};
//     std::vector<std::vector<McastRoutingInfo>> routing_info = {
//         {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2},
//          McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
//         {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
//          McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
//         {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 2},
//          McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}},
//         {McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
//          McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 2}},
//     };

//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2), FabricNodeId(MeshId{1}, 3)},
//         {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)},
//         {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 7)},
//         {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info[i % 4], mcast_group_node_ids[i % 4]);
//     }
// }

// TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMesh_EW_MulticastWithTurns) {
//     std::vector<FabricNodeId> mcast_req_nodes = {
//         FabricNodeId(MeshId{0}, 1), FabricNodeId(MeshId{0}, 2), FabricNodeId(MeshId{0}, 5), FabricNodeId(MeshId{0}, 6)};
//     std::vector<FabricNodeId> mcast_start_nodes = {
//         FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 6), FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 2)};
//     std::vector<McastRoutingInfo> routing_info = {
//         McastRoutingInfo{.mcast_dir = RoutingDirection::E, .num_mcast_hops = 1},
//         McastRoutingInfo{.mcast_dir = RoutingDirection::W, .num_mcast_hops = 1}};
//     std::vector<std::vector<FabricNodeId>> mcast_group_node_ids = {
//         {FabricNodeId(MeshId{1}, 4), FabricNodeId(MeshId{1}, 6)},
//         {FabricNodeId(MeshId{1}, 5), FabricNodeId(MeshId{1}, 7)},
//         {FabricNodeId(MeshId{1}, 0), FabricNodeId(MeshId{1}, 2)},
//         {FabricNodeId(MeshId{1}, 1), FabricNodeId(MeshId{1}, 3)}};
//     for (uint32_t i = 0; i < 500; i++) {
//         multihost_utils::InterMeshLineMcast(
//             this, mcast_req_nodes[i % 4], mcast_start_nodes[i % 4], routing_info, mcast_group_node_ids[i % 4]);
//     }
// }

void test_single_connection(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device_,
    std::size_t socket_fifo_size,
    std::size_t page_size,
    std::size_t data_size) {

    using namespace tt::tt_metal::distributed::multihost;
    using namespace tt::tt_metal::distributed;
    using namespace tt_metal;

    const auto sender_logical_coord = CoreCoord(0, 0);
    const auto recv_logical_coord = CoreCoord(0, 0);

    auto l1_alignment = MetalContext::instance().hal().get_alignment(HalMemType::L1);
    auto fabric_max_packet_size = tt_fabric::get_tt_fabric_max_payload_size_bytes();
    auto packet_header_size_bytes = tt_fabric::get_tt_fabric_packet_header_size_bytes();

    SocketConnection socket_connection = {
        .sender_core = {MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {MeshCoordinate(0, 0), recv_logical_coord}
    };

    SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = tt_metal::BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    SocketConfig socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
        .sender_rank = Rank{0},
        .receiver_rank = Rank{1}
    };

    auto socket = MeshSocket(mesh_device_, socket_config);

    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    std::vector<uint32_t> src_vec(data_size / sizeof(uint32_t));
    uint32_t seed = 0;
    if (*(distributed_context->rank()) == 0) {
        seed = std::chrono::steady_clock::now().time_since_epoch().count();
        distributed_context->send(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        std::cout << "Sender using seed: " << seed << std::endl;
    } else {
        distributed_context->recv(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&seed), sizeof(seed)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange seed over tag 0
        );
        std::cout << "Recv using seed: " << seed << std::endl;
    }
    std::mt19937 gen(seed);
    std::uniform_int_distribution<uint32_t> dis(0, UINT32_MAX);
    std::generate(src_vec.begin(), src_vec.end(), [&]() { return dis(gen); });

    const auto reserved_packet_header_CB_index = tt::CB::c_in0;
    for (int i = 0; i < 10; i++) {
        if (*(distributed_context->rank()) == 0) {
            auto sender_fabric_node_id = mesh_device_->get_device_fabric_node_id(MeshCoordinate(0, 0));
            auto recv_fabric_node_id = socket.get_fabric_node_id(SocketEndpoint::RECEIVER, MeshCoordinate(0, 0));

            auto sender_data_shard_params =
                ShardSpecBuffer(CoreRangeSet(sender_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

            const DeviceLocalBufferConfig sender_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(sender_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};
            const ReplicatedBufferConfig buffer_config{.size = data_size};

            auto sender_data_buffer = MeshBuffer::create(buffer_config, sender_device_local_config, mesh_device_.get());

            WriteShard(mesh_device_->mesh_command_queue(), sender_data_buffer, src_vec, MeshCoordinate(0, 0));

            tt::tt_metal::CircularBufferConfig sender_cb_reserved_packet_header_config =
                tt::tt_metal::CircularBufferConfig(
                    2 * packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::UInt32}})
                    .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
        
            auto sender_program = CreateProgram();
            auto sender_kernel = CreateKernel(
                sender_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_sender.cpp",
                sender_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args =
                        {static_cast<uint32_t>(socket.get_config_buffer()->address()),
                            static_cast<uint32_t>(sender_data_buffer->address()),
                            static_cast<uint32_t>(page_size),
                            static_cast<uint32_t>(data_size)},
                    .defines = {{"FABRIC_MAX_PACKET_SIZE", std::to_string(fabric_max_packet_size)}}});
        
            auto sender_packet_header_CB_handle =
                CreateCircularBuffer(sender_program, sender_logical_coord, sender_cb_reserved_packet_header_config);
        
            std::vector<uint32_t> sender_rtas;
            tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, recv_fabric_node_id, 0, sender_program, {sender_logical_coord}, sender_rtas);
        
            tt_metal::SetRuntimeArgs(sender_program, sender_kernel, sender_logical_coord, sender_rtas);

            auto sender_mesh_workload = CreateMeshWorkload();
            MeshCoordinateRange devices = MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0));
        
            AddProgramToMeshWorkload(sender_mesh_workload, std::move(sender_program), devices);
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), sender_mesh_workload, false);
            Finish(mesh_device_->mesh_command_queue());
        } else {
            auto recv_fabric_node_id = mesh_device_->get_device_fabric_node_id(MeshCoordinate(0, 0));
            auto sender_fabric_node_id = socket.get_fabric_node_id(SocketEndpoint::SENDER, MeshCoordinate(0, 0));

            auto recv_virtual_coord = mesh_device_->worker_core_from_logical_core(recv_logical_coord);
            auto recv_data_shard_params =
                ShardSpecBuffer(CoreRangeSet(recv_logical_coord), {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {1, 1});

            const DeviceLocalBufferConfig recv_device_local_config{
                .page_size = data_size,
                .buffer_type = BufferType::L1,
                .sharding_args = BufferShardingArgs(recv_data_shard_params, TensorMemoryLayout::HEIGHT_SHARDED),
                .bottom_up = false};

            const ReplicatedBufferConfig buffer_config{.size = data_size};

            auto recv_data_buffer = MeshBuffer::create(buffer_config, recv_device_local_config, mesh_device_.get());

            tt::tt_metal::CircularBufferConfig recv_cb_packet_header_config =
                tt::tt_metal::CircularBufferConfig(
                    packet_header_size_bytes, {{reserved_packet_header_CB_index, tt::DataFormat::UInt32}})
                    .set_page_size(tt::CB::c_in0, packet_header_size_bytes);

            auto recv_program = CreateProgram();
            KernelHandle recv_kernel = CreateKernel(
                recv_program,
                "tests/tt_metal/tt_metal/test_kernels/misc/socket/fabric_receiver_worker.cpp",
                recv_logical_coord,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::RISCV_0_default,
                    .compile_args = {
                        static_cast<uint32_t>(socket.get_config_buffer()->address()),
                        static_cast<uint32_t>(reserved_packet_header_CB_index),
                        static_cast<uint32_t>(page_size),
                        static_cast<uint32_t>(data_size),
                        static_cast<uint32_t>(recv_virtual_coord.x),
                        static_cast<uint32_t>(recv_virtual_coord.y),
                        static_cast<uint32_t>(recv_data_buffer->address())}});

            auto recv_packet_header_CB_handle =
                CreateCircularBuffer(recv_program, recv_logical_coord, recv_cb_packet_header_config);

            std::vector<uint32_t> recv_rtas;
            tt_fabric::append_fabric_connection_rt_args(
                recv_fabric_node_id, sender_fabric_node_id, 0, recv_program, {recv_logical_coord}, recv_rtas);
            tt_metal::SetRuntimeArgs(recv_program, recv_kernel, recv_logical_coord, recv_rtas);

            auto recv_mesh_workload = CreateMeshWorkload();
            MeshCoordinateRange devices = MeshCoordinateRange(MeshCoordinate(0, 0), MeshCoordinate(0, 0));
            AddProgramToMeshWorkload(recv_mesh_workload, std::move(recv_program), devices);
            EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), recv_mesh_workload, false);
            std::vector<uint32_t> recv_data_readback;
            ReadShard(mesh_device_->mesh_command_queue(), recv_data_readback, recv_data_buffer, MeshCoordinate(0, 0));
            EXPECT_EQ(src_vec, recv_data_readback);
        }
        for (int i = 0; i < src_vec.size(); i++) {
            src_vec[i]++;
        }
    }
}

TEST_F(InterMeshDual2x4Fabric2DFixture, MultiMeshSingleConnection) {
    test_single_connection(mesh_device_, 1024, 64, 1024);
    test_single_connection(mesh_device_, 1024, 64, 2048);
    test_single_connection(mesh_device_, 4096, 1088, 9792);
}

}  // namespace fabric_router_tests::multihost
}  // namespace tt::tt_fabric
