// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_fabric_2d_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/device/all_to_all_dispatch_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <limits>

namespace ttnn::operations::experimental::ccl {

// Reuse the detail functions from all_to_all_dispatch directly

AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::cached_mesh_workload_t
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    auto mesh_device = tensor_args.input_tensor.mesh_device();

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::shared_variables_t>
AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto output_tensor = tensor_return_value;
    auto num_links = operation_attributes.num_links;
    auto cluster_axis = operation_attributes.cluster_axis;

    // Convert ttnn::ccl::Topology to tt::tt_fabric::Topology
    auto topology = operation_attributes.topology == ttnn::ccl::Topology::Ring ? tt::tt_fabric::Topology::Ring
                                                                               : tt::tt_fabric::Topology::Linear;

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    auto src_device = mesh_device->get_device(mesh_coordinate);
    auto src_physical_device_id = src_device->id();

    auto fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(src_device->id());
    uint32_t src_mesh_id = *fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    log_debug(
        tt::LogOp,
        "\nCreating all_gather_fabric_2d program for mesh coordinate: ({}, {}) with physical device id: {} mesh id: {} "
        "chip id: {} cluster_axis: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_device->id(),
        src_mesh_id,
        src_chip_id,
        cluster_axis);

    // Use the cluster_axis instead of optional axis for 2D fabric all gather
    const auto [neighbors, directions] = ttnn::operations::ccl::detail::get_neighbors(
        mesh_view, mesh_coordinate, topology, std::optional<uint32_t>(cluster_axis));

    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    uint32_t num_devices = mesh_view.num_devices();
    uint32_t gather_devices = cluster_axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    uint32_t hidden_size = input_shape[-1];

    auto input_page_size = input_tensor.buffer()->page_size();
    auto output_page_size = output_tensor.buffer()->page_size();
    auto input_pages = input_tensor.buffer()->num_pages();
    auto output_pages = output_tensor.buffer()->num_pages();

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());

    constexpr uint32_t buffering_factor = 2;
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    uint32_t output_tensor_cb_id = tt::CBIndex::c_1;
    uint32_t packet_header_cb_id = tt::CBIndex::c_2;

    uint32_t aligned_input_page_size = input_tensor.buffer()->aligned_page_size();
    uint32_t aligned_output_page_size = output_tensor.buffer()->aligned_page_size();

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * aligned_input_page_size, {{input_tensor_cb_id, input_data_format}})
            .set_page_size(input_tensor_cb_id, aligned_input_page_size);

    tt::tt_metal::CircularBufferConfig cb_output_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * aligned_output_page_size, {{output_tensor_cb_id, output_data_format}})
            .set_page_size(output_tensor_cb_id, aligned_output_page_size);

    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, packet_header_size_bytes);

    auto subdevice_core_range_set = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        operation_attributes.subdevice_id.value_or(tt::tt_metal::SubDeviceId{0}));

    auto subdevice_cores = corerange_to_cores(subdevice_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    std::vector<CoreCoord> sender_cores;
    for (uint32_t i = 0; i < num_links; i++) {
        sender_cores.push_back(subdevice_cores.at(i));
    }

    auto sender_core = sender_cores.at(0);

    // Create circular buffers
    auto input_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_input_tensor_config);
    auto output_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_output_tensor_config);
    auto packet_header_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, packet_header_cb_config);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto device = mesh_device->get_device(coord);
        auto fabric_node_id = tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id());
        dest_mesh_id.push_back(*fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)fabric_node_id.chip_id);
    }

    log_debug(tt::LogOp, "dest_chip_id: {}", ttnn::operations::ccl::detail::stringify_vector(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ttnn::operations::ccl::detail::stringify_vector(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ttnn::operations::ccl::detail::stringify_array(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor.buffer()->is_dram(),
        output_tensor.buffer()->is_dram(),
        input_tensor_cb_id,
        output_tensor_cb_id,
        packet_header_cb_id,
        input_pages,
        output_pages,
        (uint32_t)input_page_size,
        (uint32_t)output_page_size,
        gather_devices,
        hidden_size,
        num_links,
        topology == tt::tt_fabric::Topology::Ring ? 1u : 0u,
        src_mesh_id,
        src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        aligned_input_page_size,
        aligned_output_page_size,
        (uint32_t)fabric_max_packet_size,
    };

    const auto& writer_compile_time_args = reader_compile_time_args;

    std::map<std::string, std::string> reader_defines = {
        {"CLUSTER_AXIS", std::to_string(cluster_axis)},
    };

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", ttnn::operations::ccl::detail::stringify_vector(dest_chip_id)},
        {"DEST_MESH_ID", ttnn::operations::ccl::detail::stringify_vector(dest_mesh_id)},
        {"DIRECTIONS", ttnn::operations::ccl::detail::stringify_array(directions)},
        {"CLUSTER_AXIS", std::to_string(cluster_axis)},
    };

    // For now, return empty program as requested
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
    };

    std::vector<uint32_t> writer_runtime_args = {
        input_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
    };

    // Add fabric connections for neighbors
    for (auto& neighbor : neighbors) {
        auto neighbor_coordinate = mesh_view.find_device(neighbor->id());
        uint32_t link_id = ttnn::operations::ccl::detail::select_link(
            mesh_view, mesh_coordinate, neighbor_coordinate, num_links, topology);
        log_debug(
            tt::LogOp,
            "Connection between ({}, {}) and ({}, {}) will choose link_id: {}",
            mesh_coordinate[0],
            mesh_coordinate[1],
            neighbor_coordinate[0],
            neighbor_coordinate[1],
            link_id);
        tt::tt_fabric::append_fabric_connection_rt_args(
            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(src_physical_device_id),
            tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(neighbor->id()),
            link_id,
            program,
            sender_core,
            writer_runtime_args);
    }

    // Set runtime args (empty for now since we don't have actual kernels)
    // tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_cores.at(0), reader_runtime_args);
    // tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_cores.at(0), writer_runtime_args);

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id, .writer_kernel_id = writer_kernel_id, .core = sender_cores.at(0)}};
}

void AllGatherFabric2DDeviceOperation::AllGatherFabric2DAdd::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Override runtime arguments for dynamic tensor addresses
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        auto& reader_kernel_id = shared_variables.reader_kernel_id;
        auto& writer_kernel_id = shared_variables.writer_kernel_id;
        auto& core = shared_variables.core;

        // When we have actual kernels, we can set the runtime args like this:
        // auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        // auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        // reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
        // reader_runtime_args.at(1) = tensor_return_value.buffer()->address();
        // writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
        // writer_runtime_args.at(1) = tensor_return_value.buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl
