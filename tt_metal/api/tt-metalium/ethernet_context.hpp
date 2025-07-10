// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <unordered_set>

#include <tt_stl/span.hpp>
#include <tt-metalium/routing_table_generator.hpp>
#include <tt-metalium/fabric_host_interface.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/multi_mesh_types.hpp>
#include "core_coord.hpp"

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

class FabricContext;

/*
* The following is copied from tt_cluster.hpp
* No need to duplicate in both places
*/
enum class EthRouterMode : uint32_t {
    IDLE = 0,
    BI_DIR_TUNNELING = 1,
    FABRIC_ROUTER = 2,
};

/*
* EthernetContext is a singleton class
* used to hold all Ethernet context
* and provides accessor functions for
* ethernet related functionality
* Most of the functionality is migrated from
* tt_cluster.
*/
class EthernetContext {

public:
    // Configures ethernet cores for fabric routers depending on whether fabric is enabled
    void configure_ethernet_cores_for_fabric_routers(
        tt_metal::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes = std::nullopt);

    EthernetContext(tt::umd::Cluster *driver, tt_ClusterDescriptor *clusterdesc) {
        driver_ = driver;
        clusterdesc_ = clusterdesc;
    }

    // Returns whether `logical_core` has an eth link to a core on a connected chip
    // Cores that connect to another cluster will show up as connected
    bool is_ethernet_link_up(chip_id_t chip_id, const CoreCoord& logical_core) const;    

private:
    tt::umd::Cluster *driver_;
    tt_ClusterDescriptor *clusterdesc_;
    void reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes);
    // Releases all reserved ethernet cores for fabric routers
    void release_ethernet_cores_for_fabric_routers();

    void disable_ethernet_cores_with_retrain();
    // Mapping of each devices' ethernet routing mode
    std::unordered_map<chip_id_t, std::unordered_map<CoreCoord, EthRouterMode>> device_eth_routing_info_;

    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<CoreCoord>>> ethernet_sockets_;
};

}  // namespace tt::tt_fabric
