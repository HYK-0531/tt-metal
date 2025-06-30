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

#include <map>
#include <unordered_map>
#include <memory>
#include <vector>

namespace tt::tt_fabric {

class FabricContext;

/*
* EthernetContext is a singleton class
* used to hold all Ethernet context
* and provides accessor functions for
* ethernet related functionality
* Most of the functionality is migrated from
* tt_cluster and ControlPlane
*/
class EthernetContext {

};

}  // namespace tt::tt_fabric
