// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "dataflow_api.h"
#include "fabric_host_interface.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/api/tt-metalium/fabric_edm_packet_header.hpp"

namespace ccl_routing_utils {

template <typename... T>
inline constexpr bool always_false_v = false;

struct line_unicast_route_info_t {
    uint32_t dst_mesh_id;
    union {
        uint32_t dst_chip_id;
        uint32_t distance_in_hops;
    };
};

struct line_multicast_route_info_t {
    eth_chan_directions routing_direction;
    uint32_t dst_mesh_id;
    union {
        uint32_t dst_chip_id;
        uint32_t start_distance_in_hops;
    };
    uint32_t range_hops;
};

inline constexpr uint32_t num_line_unicast_args = 2;
inline constexpr uint32_t num_line_multicast_args = 4;

template <uint32_t arg_idx>
constexpr line_unicast_route_info_t get_line_unicast_route_info_from_args() {
    return {.dst_mesh_id = get_compile_time_arg_val(arg_idx), .dst_chip_id = get_compile_time_arg_val(arg_idx + 1)};
}

template <uint32_t arg_idx>
constexpr line_multicast_route_info_t get_line_multicast_route_info_from_args() {
    return {
        .routing_direction = static_cast<eth_chan_directions>(get_compile_time_arg_val(arg_idx)),
        .dst_mesh_id = get_compile_time_arg_val(arg_idx + 1),
        .dst_chip_id = get_compile_time_arg_val(arg_idx + 2),
        .range_hops = get_compile_time_arg_val(arg_idx + 3)};
}

// dst_chip_id is the hop count for 1D routing, and the chip ID for 2D routing
// dst_mesh_id is the mesh ID of the destination chip (ignored for 1D routing)
template <typename packet_header_t>
FORCE_INLINE void fabric_set_line_unicast_route(
    volatile tt_l1_ptr packet_header_t* fabric_header_addr, const line_unicast_route_info_t& route_info) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::MeshPacketHeader>) {
        fabric_set_unicast_route(
            (packet_header_t*)fabric_header_addr,
            eth_chan_directions::COUNT,  // Ignored
            0,                           // Ignored
            route_info.dst_chip_id,
            route_info.dst_mesh_id,
            0  // Ignored
        );
    } else if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::LowLatencyPacketHeader>) {
        fabric_header_addr->to_chip_unicast(static_cast<uint8_t>(route_info.distance_in_hops));
    } else {
        static_assert(
            always_false_v<packet_header_t>, "Unsupported packet header type passed to fabric_set_unicast_route");
    }
}

// dst_chip_id is the start hop count for 1D routing, and the chip ID for 2D routing
// dst_mesh_id is the mesh ID of the destination chip (ignored for 1D routing)
// routing_direction is the direction of the multicast, and is ignored for 1D routing
// num_hops is the number of hops for the multicast in the specified direction
template <typename packet_header_t>
FORCE_INLINE void fabric_set_line_multicast_route(
    volatile tt_l1_ptr packet_header_t* fabric_header_addr, const line_multicast_route_info_t& route_info) {
    if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::MeshPacketHeader>) {
        if (route_info.routing_direction == eth_chan_directions::EAST) {
            fabric_set_mcast_route(
                (packet_header_t*)fabric_header_addr,
                route_info.dst_chip_id,
                route_info.dst_mesh_id,
                route_info.range_hops,
                0,
                0,
                0);
        } else if (route_info.routing_direction == eth_chan_directions::WEST) {
            fabric_set_mcast_route(
                (packet_header_t*)fabric_header_addr,
                route_info.dst_chip_id,
                route_info.dst_mesh_id,
                0,
                route_info.range_hops,
                0,
                0);
        } else if (route_info.routing_direction == eth_chan_directions::NORTH) {
            fabric_set_mcast_route(
                (packet_header_t*)fabric_header_addr,
                route_info.dst_chip_id,
                route_info.dst_mesh_id,
                0,
                0,
                route_info.range_hops,
                0);
        } else if (route_info.routing_direction == eth_chan_directions::SOUTH) {
            fabric_set_mcast_route(
                (packet_header_t*)fabric_header_addr,
                route_info.dst_chip_id,
                route_info.dst_mesh_id,
                0,
                0,
                0,
                route_info.range_hops);
        } else {
            ASSERT(0);
        }
    } else if constexpr (std::is_same_v<packet_header_t, tt::tt_fabric::LowLatencyPacketHeader>) {
        fabric_header_addr->to_chip_multicast(MulticastRoutingCommandHeader{
            static_cast<uint8_t>(route_info.start_distance_in_hops), static_cast<uint8_t>(route_info.range_hops)});
    } else {
        static_assert(
            always_false_v<packet_header_t>, "Unsupported packet header type passed to fabric_set_line_mcast_route");
    }
}

}  // namespace ccl_routing_utils
