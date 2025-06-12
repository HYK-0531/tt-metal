// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mesh_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <stdint.h>

namespace tt {
namespace tt_metal {
namespace distributed {
template <typename T>
class MeshContainer;
}  // namespace distributed
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {

// PhysicalMeshCoordinate is a 2D-coordinate in the physical mesh as defined by the Fabric layer.
// MeshCoordinate[0] is the mesh_id and MeshCoordinate[1] is the physical_device_id.
class PhysicalMeshCoordinate {
public:
    using chip_id_t = uint32_t;
    using MeshId = tt::tt_fabric::MeshId;
    PhysicalMeshCoordinate() = delete;
    //PhysicalMeshCoordinate(chip_id_t mesh_id, chip_id_t chip_id) : mesh_id_(MeshId(mesh_id)), chip_id_(chip_id) {}
    PhysicalMeshCoordinate(chip_id_t mesh_id, std::optional<chip_id_t> chip_id)
        : mesh_id_(MeshId(mesh_id)), chip_id_(chip_id) {}
    MeshId mesh_id() const { return mesh_id_; }
    chip_id_t chip_id() const { 
        TT_ASSERT(chip_id_.has_value(), "Chip id is not set");
        return chip_id_.value();
    }
    bool is_valid() const { return chip_id_.has_value(); }

private:
    MeshId mesh_id_{0};
    std::optional<chip_id_t> chip_id_;
};

// Returns a map of all physical mesh coordinates in the system.
const MeshContainer<PhysicalMeshCoordinate>& get_system_mesh_coordinate_translation_map();

}  // namespace tt::tt_metal::distributed
