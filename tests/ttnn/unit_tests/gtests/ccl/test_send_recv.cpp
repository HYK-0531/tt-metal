// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gtest/gtest.h"

#include "ttnn/operations/experimental/ccl/send_async/send_async.hpp"
#include "ttnn/operations/experimental/ccl/recv_async/recv_async.hpp"
#include "ttnn/operations/experimental/reshape/view.hpp"
#include "ttnn/operations/creation.hpp"

#include "tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include <tt-metalium/mesh_socket.hpp>

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

class T3K2DFabricSendRecvFixture : public T3000MeshDevice2DFabricFixture,
                                   public testing::WithParamInterface<TensorSpec> {};

TEST_P(T3K2DFabricSendRecvFixture, SendRecvAsync) {
    auto tensor_spec = GetParam();
    auto mesh_device = get_mesh_device();
    auto md0 = mesh_device->create_submesh(distributed::MeshShape(1, 1), distributed::MeshCoordinate(0, 1));
    auto md1 = mesh_device->create_submesh(distributed::MeshShape(1, 1), distributed::MeshCoordinate(0, 2));

    auto sender_logical_coord = CoreCoord(0, 0);
    auto recv_logical_coord = CoreCoord(0, 1);
    uint32_t socket_fifo_size = 10 * 1024;
    distributed::SocketConnection socket_connection = {
        .sender_core = {distributed::MeshCoordinate(0, 0), sender_logical_coord},
        .receiver_core = {distributed::MeshCoordinate(0, 0), recv_logical_coord},
    };

    distributed::SocketMemoryConfig socket_mem_config = {
        .socket_storage_type = BufferType::L1,
        .fifo_size = socket_fifo_size,
    };

    distributed::SocketConfig socket_config = {
        .socket_connection_config = {socket_connection},
        .socket_mem_config = socket_mem_config,
    };
    auto [send_socket, recv_socket] = distributed::MeshSocket::create_socket_pair(md0, md1, socket_config);

    const auto& input_shape = tensor_spec.logical_shape();
    const auto& memory_config = tensor_spec.memory_config();
    uint32_t num_elems = input_shape.volume();
    auto layout = tensor_spec.layout();
    auto dtype = tensor_spec.data_type();
    Tensor input_tensor =
        ttnn::distributed::aggregate_as_tensor(
            {ttnn::experimental::view(ttnn::arange(0, num_elems, 1, dtype), input_shape).to_layout(layout)},
            ReplicateTensor{})
            .to_device(md0.get(), memory_config);
    auto output_tensor = tt::tt_metal::allocate_tensor_on_mesh(
        TensorSpec(input_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(layout), memory_config)),
        md1.get());

    ttnn::experimental::send_async(input_tensor, send_socket);
    ttnn::experimental::recv_async(output_tensor, recv_socket);

    distributed::Synchronize(md0.get(), std::nullopt);
    distributed::Synchronize(md1.get(), std::nullopt);

    auto input_data = input_tensor.to_vector<uint32_t>();
    auto output_data = output_tensor.to_vector<uint32_t>();
    EXPECT_EQ(input_data, output_data);
    for (uint32_t i = 0; i < num_elems; ++i) {
        EXPECT_EQ(input_data[i], output_data[i]);
        if (input_data[i] != output_data[i]) {
            break;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(T3K2DFabricSendRecvTests, T3K2DFabricSendRecvFixture, ::testing::ValuesIn(tensor_specs));

}  // namespace tt::tt_metal
