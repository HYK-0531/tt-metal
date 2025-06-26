// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <filesystem>
#include <memory>
#include <vector>

#include "fabric_fixture.hpp"
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed_context.hpp>

namespace tt::tt_fabric {
namespace multi_host_tests {

TEST(MultiHost, TestDualGalaxyControlPlaneInit) {
    const std::filesystem::path dual_galaxy_mesh_graph_desc_path =
        std::filesystem::path(tt::tt_metal::MetalContext::instance().rtoptions().get_root_dir()) /
        "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.yaml";
    auto control_plane = std::make_unique<ControlPlane>(dual_galaxy_mesh_graph_desc_path.string());

    for (int i= 0; i < 32; ++i) {
        const auto& active_eth_cores = control_plane->get_active_ethernet_cores(i, false);
    std::cout << " active eth cores size: " << active_eth_cores.size() << std::endl;
    for (auto& core : active_eth_cores) {
      std::cout << " active eth core: " << core.str() << std::endl;
    }
    }
   // control_plane->configure_routing_tables_for_fabric_ethernet_channels(
   //     tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
   //     tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
}

TEST(MultiHost, TestDualGalaxyFabricSanity) {
    tt::tt_metal::MetalContext::instance().set_fabric_config(
        tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC,
        tt::tt_fabric::FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE);
    tt::tt_metal::MetalContext::instance().initialize_fabric_config();
}

}  // namespace multi_host_tests
}  // namespace tt::tt_fabric
