// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tt-metalium/distributed_context.hpp"

#if defined(OPEN_MPI)
#include "mpi_distributed_context.hpp"
#else
#include "single_host_context.hpp"
#endif

namespace tt::tt_metal::distributed::multihost {

#if defined(OPEN_MPI)
using ContextImpl = MPIContext;
#else
using ContextImpl = SingleHostContext;
#endif

/* -------------------- factory for generic interface --------------------- */
void DistributedContext::create(int argc, char** argv) { ContextImpl::create(argc, argv); }

const ContextPtr& DistributedContext::get_current_world() { return ContextImpl::get_current_world(); }

void DistributedContext::set_current_world(const ContextPtr& ctx) { ContextImpl::set_current_world(ctx); }

bool DistributedContext::is_initialized() { return ContextImpl::is_initialized(); }

DistributedContextId DistributedContext::id() const { return id_; }

/* -------------------- DistributedContext ID generation --------------------- */
DistributedContextId DistributedContext::unique_distributed_context_id() {
    // This function is used to generate a unique ID for each DistributedContext instance.
    // It allows tracking which contexts are in use, and can be used for creating context specific resources.
    // This function is not thread-safe.
    static std::size_t next_id = 0;
    return DistributedContextId(next_id++);
}

}  // namespace tt::tt_metal::distributed::multihost
