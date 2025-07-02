// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Utility functions and data structures related to tt-metal kernel profiler's noc tracing feature

#include <nlohmann/json_fwd.hpp>
#include <string>
#include <tuple>
#include <optional>
#include <map>
#include <vector>
#include <utility>
#include <nlohmann/json.hpp>

#include "tt_cluster.hpp"
#include "fabric/fabric_host_utils.hpp"
#include "tt_metal.hpp"

namespace tt {

namespace tt_metal {

// precomputes the mapping between EDM router physical coordinate locations and their associated fabric channel IDs
class FabricRoutingLookup {
public:
    // both of these are keyed by physical chip id!
    using EthCoreToChannelMap = std::map<std::tuple<chip_id_t, CoreCoord>, tt::tt_fabric::chan_id_t>;

    // Default constructor for cases where lookup is not built (e.g., non-1D fabric)
    FabricRoutingLookup() = default;

    // lookup Eth Channel ID given a physical chip id and physical EDM router core coordinate
    std::optional<tt::tt_fabric::chan_id_t> getRouterEthCoreToChannelLookup(
        chip_id_t phys_chip_id, CoreCoord eth_router_phys_core_coord) const {
        std::optional<tt::tt_fabric::chan_id_t> eth_chan = std::nullopt;
        try {
            const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(phys_chip_id);
            CoreCoord eth_router_logical_core_coord = soc_desc.translate_coord_to(
                tt::umd::CoreCoord(eth_router_phys_core_coord, CoreType::ETH, CoordSystem::PHYSICAL),
                CoordSystem::LOGICAL);
            eth_chan = soc_desc.logical_eth_core_to_chan_map.at(eth_router_logical_core_coord);
        } catch (const std::exception& e) {
            log_error(
                tt::LogMetal,
                "Failed to translate physical coordinate {},{} to logical",
                eth_router_phys_core_coord.x,
                eth_router_phys_core_coord.y);
        }
        return eth_chan;
    }

    void RecordForwardingChannelPair(
        const tt_fabric::FabricNodeId& fabric_node_id, CoreCoord eth_logical_core1, CoreCoord eth_logical_core2) {
        const auto& control_plane = tt::tt_metal::MetalContext::instance().get_control_plane();
        auto physical_chip_id = control_plane.get_physical_chip_id_from_fabric_node_id(fabric_node_id);
        const auto& soc_desc = tt::tt_metal::MetalContext::instance().get_cluster().get_soc_desc(physical_chip_id);
        ethernet_channel_t eth_chan1 = soc_desc.logical_eth_core_to_chan_map.at(eth_logical_core1);
        ethernet_channel_t eth_chan2 = soc_desc.logical_eth_core_to_chan_map.at(eth_logical_core2);
        forwarding_channel_pairs[physical_chip_id].push_back({eth_chan1, eth_chan2});
    }

    std::unordered_map<chip_id_t, std::vector<std::pair<int, int>>> forwarding_channel_pairs;
};

inline void dumpClusterCoordinatesAsJson(const std::filesystem::path& filepath) {
    Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    nlohmann::ordered_json cluster_json;
    cluster_json["physical_chip_to_eth_coord"] = nlohmann::ordered_json();
    for (auto& [chip_id, eth_core] : cluster.get_user_chip_ethernet_coordinates()) {
        eth_coord_t eth_coord = eth_core;
        auto& entry = cluster_json["physical_chip_to_eth_coord"][std::to_string(chip_id)];
        entry["rack"] = eth_coord.rack;
        entry["shelf"] = eth_coord.shelf;
        entry["x"] = eth_coord.x;
        entry["y"] = eth_coord.y;
    }

    std::ofstream cluster_json_ofs(filepath);
    if (cluster_json_ofs.is_open()) {
        cluster_json_ofs << cluster_json.dump(2);
    } else {
        log_error(tt::LogMetal, "Failed to open file '{}' for dumping cluster coordinate map", filepath.string());
    }
}

inline void dumpRoutingInfo(const std::filesystem::path& filepath, FabricRoutingLookup& routing_lookup) {
    nlohmann::ordered_json topology_json;

    const Cluster& cluster = tt::tt_metal::MetalContext::instance().get_cluster();

    topology_json["forwarding_channel_pairs"] = nlohmann::ordered_json::array();
    for (auto [physical_chip_id, device_forwarding_channel_pairs] : routing_lookup.forwarding_channel_pairs) {
        for (auto forwarding_channel_pair : device_forwarding_channel_pairs) {
            topology_json["forwarding_channel_pairs"].push_back(
                {{"device_id", physical_chip_id},
                 {"channels", {forwarding_channel_pair.first, forwarding_channel_pair.second}}});
        }
    }

    std::unordered_map<chip_id_t, std::unordered_map<ethernet_channel_t, std::tuple<chip_id_t, ethernet_channel_t>>>
        ethernet_connections = cluster.get_ethernet_connections();
    topology_json["inter_device_channel_pairs"] = nlohmann::ordered_json::array();
    for (auto [physical_chip_id, local_channel_map] : ethernet_connections) {
        for (auto [local_eth_channel, remote_chip_and_channel] : local_channel_map) {
            auto [remote_chip_id, remote_eth_channel] = remote_chip_and_channel;
            topology_json["inter_device_channel_pairs"].push_back(
                {{"local_device_id", physical_chip_id},
                 {"local_eth_channel", local_eth_channel},
                 {"remote_device_id", remote_chip_id},
                 {"remote_eth_channel", remote_eth_channel}});
        }
    }

    topology_json["eth_chan_to_coord"] = nlohmann::ordered_json::object();
    auto physical_chip_id = *(cluster.get_cluster_desc()->get_all_chips().begin());
    for (int j = 0; j < cluster.get_soc_desc(physical_chip_id).get_num_eth_channels(); j++) {
        tt::umd::CoreCoord edm_eth_core =
            cluster.get_soc_desc(physical_chip_id).get_eth_core_for_channel(j, CoordSystem::PHYSICAL);
        topology_json["eth_chan_to_coord"][std::to_string(j)] = {edm_eth_core.x, edm_eth_core.y};
    }

    std::ofstream topology_json_ofs(filepath);
    if (topology_json_ofs.is_open()) {
        topology_json_ofs << topology_json.dump(2);
    } else {
        log_error(tt::LogMetal, "Failed to open file '{}' for dumping topology", filepath.string());
    }
}

// determines the implied unicast/multicast start distance and range in tt_fabric::LowLatencyRoutingFields
inline std::tuple<int, int> get_low_latency_routing_start_distance_and_range(uint32_t llrf_value) {
    using LLRF = tt::tt_fabric::LowLatencyRoutingFields;

    uint32_t value = llrf_value;
    uint32_t hops = 0;
    while (value) {
        value >>= tt::tt_fabric::LowLatencyRoutingFields::FIELD_WIDTH;
        hops++;
    }
    return hops;
}

// determines the implied unicast hop count in tt_fabric::RoutingFields
inline int get_routing_hops(uint8_t routing_fields_value) {
    return tt::tt_fabric::RoutingFields::HOP_DISTANCE_MASK & routing_fields_value;
}

}  // namespace tt_metal
}  // namespace tt
