// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gtest/gtest.h"
#include <vector>

#include "impl/context/metal_context.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tests/tt_metal/tt_metal/common/multi_device_fixture.hpp"
#include "intermesh_routing_test_utils.hpp"
#include <tt-metalium/distributed.hpp>

namespace tt::tt_fabric {
namespace fabric_router_tests {
namespace {

template <typename Fixture>
void validate_system_configs(Fixture* fixture) {
    TT_FATAL(
        tt::tt_metal::MetalContext::instance().get_control_plane().system_has_intermesh_links(),
        "Multi-Host Routing tests require ethernet links to a remote host.");
    TT_FATAL(
        *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) > 1,
        "Multi-Host Routing tests require multiple hosts in the system");
}

}  // namespace

// Base fixture for Inter-Mesh Routing Fabric 2D tests.
class InterMeshRoutingFabric2DFixture : public BaseFabricFixture {
public:
    // This test fixture closes/opens devices on each test
    static void SetUpTestSuite() {}
    static void TearDownTestSuite() {}
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a supported system.";
        }

        validate_system_configs(this);
        this->DoSetUpTestSuite(tt::tt_fabric::FabricConfig::FABRIC_2D_DYNAMIC);
    }

    void TearDown() override {
        if (system_supported()) {
            BaseFabricFixture::DoTearDownTestSuite();
        }
    }

    virtual tt_metal::distributed::MeshShape get_mesh_shape() = 0;
    virtual uint32_t get_num_procs() = 0;

    bool system_supported() {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto mesh_shape = this->get_mesh_shape();
        auto num_chips_in_mesh = mesh_shape[0] * mesh_shape[1];
        return *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) == this->get_num_procs() &&
               cluster.user_exposed_chip_ids().size() == num_chips_in_mesh;
    }
};

// Base fixture for Multi-Host MeshDevice tests relying on Inter-Mesh Routing.
class MultiMeshDeviceFabricFixture : public tt::tt_metal::GenericMeshDevice2DFabricFixture {
public:
    void SetUp() override {
        if (not system_supported()) {
            GTEST_SKIP() << "Skipping since this is not a supported system.";
        }
        validate_system_configs(this);
        tt::tt_metal::GenericMeshDevice2DFabricFixture::SetUp();
    }

    void TearDown() override {
        if (system_supported()) {
            tt::tt_metal::GenericMeshDevice2DFabricFixture::TearDown();
        }
    }

    virtual tt_metal::distributed::MeshShape get_mesh_shape() = 0;
    virtual uint32_t get_num_procs() = 0;

    bool system_supported() {
        const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
        auto mesh_shape = this->get_mesh_shape();
        auto num_chips_in_mesh = mesh_shape[0] * mesh_shape[1];
        return *(tt::tt_metal::MetalContext::instance().get_distributed_context().size()) == this->get_num_procs() &&
               cluster.user_exposed_chip_ids().size() == num_chips_in_mesh;
    }
};

// Generic Fixture for Split 2x2 T3K systems using Fabric
template <typename Fixture>
class Split2x2FabricFixture : public Fixture {
public:
    tt_metal::distributed::MeshShape get_mesh_shape() { return tt_metal::distributed::MeshShape{2, 2}; }
    uint32_t get_num_procs() { return 2; }
};

// Generic Fixture for Split 1x2 T3K systems using Fabric
template <typename Fixture>
class Split1x2FabricFixture : public Fixture {
public:
    tt_metal::distributed::MeshShape get_mesh_shape() { return tt_metal::distributed::MeshShape{1, 2}; }
    uint32_t get_num_procs() { return 4; }
};

// Generic Fixture for Dual 2x2 T3K systems using Fabric
template <typename Fixture>
class Dual2x2FabricFixture : public Fixture {
public:
    tt_metal::distributed::MeshShape get_mesh_shape() { return tt_metal::distributed::MeshShape{2, 2}; }
    uint32_t get_num_procs() { return 2; }
};

// Generic Fixture for Dual T3K systems using Fabric
template <typename Fixture>
class Dual2x4FabricFixture : public Fixture {
public:
    tt_metal::distributed::MeshShape get_mesh_shape() { return tt_metal::distributed::MeshShape(2, 4); }
    uint32_t get_num_procs() { return 2; }
};

// Generic Fixture for Nano-Exabox systems using Fabric
template <typename Fixture>
class NanoExaboxFabricFixture : public Fixture {
public:
    tt_metal::distributed::MeshShape get_mesh_shape() { return tt_metal::distributed::MeshShape(2, 4); }
    uint32_t get_num_procs() { return 5; }
};

// Dedicated Fabric and Distributed Test Fixtures fir Multi-Host + Multi-Mesh Tests
using IntermeshSplit2x2FabricFixture = Split2x2FabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceSplit2x2Fixture = Split2x2FabricFixture<MultiMeshDeviceFabricFixture>;

using InterMeshSplit1x2FabricFixture = Split1x2FabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceSplit1x2Fixture = Split1x2FabricFixture<MultiMeshDeviceFabricFixture>;

using IntermeshDual2x2FabricFixture = Dual2x2FabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceDual2x2Fixture = Dual2x2FabricFixture<MultiMeshDeviceFabricFixture>;

using InterMeshDual2x4FabricFixture = Dual2x4FabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceDual2x4Fixture = Dual2x4FabricFixture<MultiMeshDeviceFabricFixture>;

using IntermeshNanoExaboxFabricFixture = NanoExaboxFabricFixture<InterMeshRoutingFabric2DFixture>;
using MeshDeviceNanoExaboxFixture = NanoExaboxFabricFixture<MultiMeshDeviceFabricFixture>;

}  // namespace fabric_router_tests
}  // namespace tt::tt_fabric
