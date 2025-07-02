// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"

#include "tt_metal/multihost/fabric_tests/multihost_fabric_fixtures.hpp"
#include <tt-metalium/mesh_socket.hpp>
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_metal {

std::array<TensorSpec, 2> tensor_specs = {
    TensorSpec(
        ttnn::Shape({3, 2, 32, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
    TensorSpec(
        ttnn::Shape({3, 2, 32, 128}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::UINT32,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
            tt::tt_metal::MemoryConfig(tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM))),
};

class MeshDeviceDual2x4SendRecvFixture : public tt::tt_fabric::fabric_router_tests::MeshDeviceDual2x4Fixture,
                                         public testing::WithParamInterface<TensorSpec> {};

TEST_P(MeshDeviceDual2x4SendRecvFixture, SendRecvAsync) {
    auto tensor_spec = GetParam();

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    uint32_t socket_fifo_size = 10 * 1024;
    auto mesh_shape = mesh_device_->shape();
    std::vector<distributed::SocketConnection> forward_socket_connections;
    forward_socket_connections.reserve(mesh_shape.mesh_size());
    std::vector<distributed::SocketConnection> backward_socket_connections;
    backward_socket_connections.reserve(mesh_shape.mesh_size());
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
        forward_socket_connections.push_back({
            .sender_core = {coord, sender_logical_coord},
            .receiver_core = {coord, recv_logical_coord},
        });
        backward_socket_connections.push_back({
            .sender_core = {coord, sender_logical_coord},
            .receiver_core = {coord, recv_logical_coord},
        });
    }

    distributed::SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    distributed::SocketConfig forward_socket_config = {
        .socket_connection_config = forward_socket_connections,
        .socket_mem_config = socket_mem_config,
        .sender_rank = distributed::multihost::Rank{0},
        .receiver_rank = distributed::multihost::Rank{1}};
    distributed::SocketConfig backward_socket_config = {
        .socket_connection_config = backward_socket_connections,
        .socket_mem_config = socket_mem_config,
        .sender_rank = distributed::multihost::Rank{1},
        .receiver_rank = distributed::multihost::Rank{0}};
    auto forward_socket = distributed::MeshSocket(mesh_device_, forward_socket_config);
    auto backward_socket = distributed::MeshSocket(mesh_device_, backward_socket_config);

    auto distributed_context = tt_metal::distributed::multihost::DistributedContext::get_current_world();

    const auto& input_shape = tensor_spec.logical_shape();
    const auto& memory_config = tensor_spec.memory_config();
    uint32_t num_elems = input_shape.volume();
    auto layout = tensor_spec.layout();
    auto dtype = tensor_spec.data_type();
    if (*(distributed_context->rank()) == 0) {
        const Tensor input_tensor =
            ttnn::distributed::distribute_tensor(
                ttnn::experimental::view(ttnn::arange(0, num_elems, 1, dtype), input_shape).to_layout(layout),
                *ttnn::distributed::replicate_tensor_to_mesh_mapper(*mesh_device_),
                *mesh_device_)
                .to_device(mesh_device_.get(), memory_config);
        ttnn::experimental::send_async(input_tensor, forward_socket);
        distributed::Synchronize(mesh_device_.get(), std::nullopt);
        auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
        auto input_data = ttnn::distributed::aggregate_tensor(input_tensor, *composer).to_vector<uint32_t>();
        // Send test results to the receiver host
        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(uint32_t)),
            tt::tt_metal::distributed::multihost::Rank{1},  // send to receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        auto output_tensor = tt::tt_metal::allocate_tensor_on_mesh(
            TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
            mesh_device_.get());
        ttnn::experimental::recv_async(output_tensor, backward_socket);
        distributed::Synchronize(mesh_device_.get(), std::nullopt);
        auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<uint32_t>();
        std::vector<uint32_t> inc_output_data(output_data.size());
        distributed_context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(uint32_t)),
            tt::tt_metal::distributed::multihost::Rank{1},  // recv from receiver host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        EXPECT_EQ(output_data, inc_output_data);
    } else {
        auto output_tensor = tt::tt_metal::allocate_tensor_on_mesh(
            TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
            mesh_device_.get());
        ttnn::experimental::recv_async(output_tensor, forward_socket);
        distributed::Synchronize(mesh_device_.get(), std::nullopt);
        auto composer = ttnn::distributed::concat_mesh_to_tensor_composer(*mesh_device_, /*dim=*/0);
        auto output_data = ttnn::distributed::aggregate_tensor(output_tensor, *composer).to_vector<uint32_t>();
        std::vector<uint32_t> input_data(output_data.size());
        distributed_context->recv(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(input_data.data()), input_data.size() * sizeof(uint32_t)),
            tt::tt_metal::distributed::multihost::Rank{0},  // recv from sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
        EXPECT_EQ(input_data, output_data);
        auto inc_output_tensor = ttnn::add(output_tensor, 1);
        ttnn::experimental::send_async(inc_output_tensor, backward_socket);
        distributed::Synchronize(mesh_device_.get(), std::nullopt);
        auto inc_output_data = ttnn::distributed::aggregate_tensor(inc_output_tensor, *composer).to_vector<uint32_t>();
        distributed_context->send(
            tt::stl::Span<std::byte>(
                reinterpret_cast<std::byte*>(inc_output_data.data()), inc_output_data.size() * sizeof(uint32_t)),
            tt::tt_metal::distributed::multihost::Rank{0},  // send to sender host
            tt::tt_metal::distributed::multihost::Tag{0}    // exchange test results over tag 0
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    MeshDeviceDual2x4SendRecvTests, MeshDeviceDual2x4SendRecvFixture, ::testing::ValuesIn(tensor_specs));

}  // namespace tt::tt_metal
