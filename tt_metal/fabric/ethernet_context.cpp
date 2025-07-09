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
#include <tt-metalium/ethernet_context.hpp>
#include "core_coord.hpp"

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

void EthernetContext::configure_ethernet_cores_for_fabric_routers(
    tt_metal::FabricConfig fabric_config, std::optional<uint8_t> num_routing_planes) {
    if (fabric_config != tt_metal::FabricConfig::DISABLED) {
        TT_FATAL(num_routing_planes.has_value(), "num_routing_planes should be set for reserving cores for fabric");
        TT_FATAL(num_routing_planes.value() > 0, "Expected non-zero num_routing_planes for reserving cores for fabric");
        this->reserve_ethernet_cores_for_fabric_routers(num_routing_planes.value());
    } else {
        if (num_routing_planes.has_value()) {
            log_warning(
                tt::LogMetal,
                "Got num_routing_planes while releasing fabric cores, ignoring it and releasing all reserved cores");
        }
        this->release_ethernet_cores_for_fabric_routers();
    }
}

void EthernetContext::reserve_ethernet_cores_for_fabric_routers(uint8_t num_routing_planes) {
    if (num_routing_planes == std::numeric_limits<uint8_t>::max()) {
        // default behavior, reserve whatever cores are available
        for (const auto& [chip_id, eth_cores] : this->device_eth_routing_info_) {
            for (const auto& [eth_core, mode] : eth_cores) {
                if (mode == EthRouterMode::IDLE) {
                    this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::FABRIC_ROUTER;
                }
            }
        }

        // Update sockets to reflect fabric routing
        this->ethernet_sockets_.clear();
        return;
    }

    // to reserve specified number of cores, ensure that the same are avaialble on connected chip id as well
    for (const auto& chip_id : this->driver_->get_target_device_ids()) {
        const auto& connected_chips_and_cores = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [connnected_chip_id, cores] : connected_chips_and_cores) {
            const uint8_t num_cores_to_reserve = std::min(num_routing_planes, static_cast<uint8_t>(cores.size()));
            uint8_t num_reserved_cores = 0;
            for (auto i = 0; i < cores.size(); i++) {
                if (num_reserved_cores == num_cores_to_reserve) {
                    break;
                }

                const auto eth_core = cores[i];
                const auto connected_core =
                    std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core)));
                if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::FABRIC_ROUTER) {
                    // already reserved for fabric, potenially by the connected chip id
                    num_reserved_cores++;
                    continue;
                }

                if (this->device_eth_routing_info_[chip_id][eth_core] == EthRouterMode::IDLE &&
                    this->device_eth_routing_info_.at(connnected_chip_id).at(connected_core) == EthRouterMode::IDLE) {
                    this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::FABRIC_ROUTER;
                    this->device_eth_routing_info_[connnected_chip_id][connected_core] = EthRouterMode::FABRIC_ROUTER;
                    num_reserved_cores++;
                }
            }

            TT_FATAL(
                num_reserved_cores == num_cores_to_reserve,
                "Unable to reserve {} routing planes b/w chip {} and {} for fabric, reserved only {}",
                num_cores_to_reserve,
                chip_id,
                connnected_chip_id,
                num_reserved_cores);
        }
    }

    // re-init sockets to reflect fabric routing
    this->ethernet_sockets_.clear();
    this->initialize_ethernet_sockets();
}

void EthernetContext::release_ethernet_cores_for_fabric_routers() {
    for (const auto& [chip_id, eth_cores] : this->device_eth_routing_info_) {
        for (const auto& [eth_core, mode] : eth_cores) {
            if (mode == EthRouterMode::FABRIC_ROUTER) {
                this->device_eth_routing_info_[chip_id][eth_core] = EthRouterMode::IDLE;
            }
        }
    }
    // TODO: We should just cache restore
    this->initialize_ethernet_sockets();
}

// initialize ethernet configuration
void EthernetContext::initialize_ethernet_sockets() {
    for (const auto& chip_id : this->driver_->get_target_device_ids()) {
        if (this->ethernet_sockets_.find(chip_id) == this->ethernet_sockets_.end()) {
            this->ethernet_sockets_.insert({chip_id, {}});
        }
        for (const auto &[connected_chip_id, eth_cores] :
             this->get_ethernet_cores_grouped_by_connected_chips(chip_id)) {
            if (this->ethernet_sockets_.at(chip_id).find(connected_chip_id) ==
                this->ethernet_sockets_.at(chip_id).end()) {
                this->ethernet_sockets_.at(chip_id).insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.find(connected_chip_id) == this->ethernet_sockets_.end()) {
                this->ethernet_sockets_.insert({connected_chip_id, {}});
            }
            if (this->ethernet_sockets_.at(connected_chip_id).find(chip_id) ==
                this->ethernet_sockets_.at(connected_chip_id).end()) {
                this->ethernet_sockets_.at(connected_chip_id).insert({chip_id, {}});
            } else {
                continue;
            }
            for (const auto &eth_core : eth_cores) {
                if (this->device_eth_routing_info_.at(chip_id).at(eth_core) == EthRouterMode::IDLE) {
                    this->ethernet_sockets_.at(chip_id).at(connected_chip_id).emplace_back(eth_core);
                    this->ethernet_sockets_.at(connected_chip_id)
                        .at(chip_id)
                        .emplace_back(
                            std::get<1>(this->get_connected_ethernet_core(std::make_tuple(chip_id, eth_core))));
                }
            }
        }
    }
}

void EthernetContext::disable_ethernet_cores_with_retrain() {
    std::vector<uint32_t> read_vec;
    const auto& chips = this->driver_->get_target_device_ids();
    for (const auto& chip_id : chips) {
        if (this->frequent_retrain_cores_.find(chip_id) == this->frequent_retrain_cores_.end()) {
            this->frequent_retrain_cores_.insert({chip_id, {}});
        }
        const auto& connected_chips = this->get_ethernet_cores_grouped_by_connected_chips(chip_id);
        for (const auto& [other_chip_id, eth_cores] : connected_chips) {
            for (const auto& eth_core : eth_cores) {
                if (rtoptions_.get_skip_eth_cores_with_retrain() and
                    this->cluster_desc_->get_board_type(chip_id) == BoardType::UBB) {
                    tt_cxy_pair virtual_eth_core(
                        chip_id, get_virtual_coordinate_from_logical_coordinates(chip_id, eth_core, CoreType::ETH));
                    auto retrain_count_addr = hal_.get_dev_addr(
                        tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH,
                        tt::tt_metal::HalL1MemAddrType::RETRAIN_COUNT);
                    this->read_core(read_vec, sizeof(uint32_t), virtual_eth_core, retrain_count_addr);
                    if (read_vec[0] != 0) {
                        log_warning(
                            LogDevice,
                            "Disabling active eth core {} due to retraining (count={})",
                            virtual_eth_core.str(),
                            read_vec[0]);
                        this->frequent_retrain_cores_[chip_id].insert(eth_core);
                    }
                }
            }
        }
    }
}
