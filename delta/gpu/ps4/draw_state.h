#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * How Liverpool register state becomes one renderer draw.
 *
 * The registers say almost nothing directly: a draw's vertex streams, constant
 * buffers and textures live behind descriptors the shaders load out of their
 * user-data SGPRs, so producing a DrawInfo is as much recovering what the
 * shaders are about to read (gpu/gcn resource tracking) as reading registers.
 * The packet walker only says which draw packet arrived.
 */

#include <cstdint>

#include "gpu/ps4/liverpool.h"
#include "gpu/rhi/command.h"
#include "gpu/rhi/renderer.h"

namespace gpu::ps4 {

// The parts of a draw that are not in the register file: the packet itself plus
// the index/instance state the IT_* packets before it set up.
struct DrawPacket {
  uint32_t op = 0;
  const uint32_t* body = nullptr;
  uint32_t count = 0;
  uint32_t index_type = 0;     // IT_INDEX_TYPE: 0 = 16-bit, 1 = 32-bit
  uint64_t index_base = 0;     // IT_INDEX_BASE (DRAW_INDEX_2 carries its own)
  uint64_t indirect_base = 0;  // IT_SET_BASE(1): where indirect args live
  uint32_t num_instances = 1;  // IT_NUM_INSTANCES
  uint32_t frame = 0;          // presented frames + 1, for the frame gates
};

// Fill `d` from the packet and the live registers. False when nothing resolved
// a vertex source: a draw whose shaders or descriptors we could not follow is
// dropped rather than rendered from garbage.
bool BuildDrawInfo(rhi::Renderer& renderer,
                   const Regs& regs,
                   const DrawPacket& packet,
                   rhi::DrawInfo& d);

}  // namespace gpu::ps4
