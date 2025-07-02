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
    void reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes);
    void release_ethernet_cores_for_fabric_routers();
    void configure_ethernet_cores_for_fabric_routers();
    std::vector<CoreCoord> get_fabric_ethernet_routers_between_src_and_dest(chip_id_t src_id, chip_id_t dst_id) const;
    bool is_ethernet_link_up(chip_id_t chip_id, const CoreCoord& logical_core) const;
    void set_internal_routing_info_for_ethernet_cores(
        bool enable_internal_routing, const std::vector<chip_id_t>& target_mmio_devices = {}) const;
    
    bool is_external_cable(chip_id_t physical_chip_id, CoreCoord eth_core) const;
        const std::unordered_set<CoreCoord>& get_eth_cores_with_frequent_retraining(chip_id_t chip_id) const {
        return this->frequent_retrain_cores_.at(chip_id);
    }

    const std::unordered_map<CoreCoord, EthRouterMode>& get_eth_routing_info(chip_id_t chip_id) const {
        return this->device_eth_routing_info_.at(chip_id);
    }

private:
    // Mapping of each devices' ethernet routing mode
    std::unordered_map<chip_id_t, std::unordered_map<CoreCoord, EthRouterMode>> device_eth_routing_info_;

    std::unordered_map<chip_id_t, std::unordered_map<chip_id_t, std::vector<CoreCoord>>> ethernet_sockets_;
};

}  // namespace tt::tt_fabric
