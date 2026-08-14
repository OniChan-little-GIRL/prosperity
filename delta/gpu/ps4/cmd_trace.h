#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The DELTA_GPU_* instrumentation of the PS4 command stream: one function per
 * knob, each taking the state it reports on and each a no-op unless its knob is
 * set. What a probe remembers between calls (line caps, histograms, the
 * register-write bookkeeping nothing else reads) lives behind these functions
 * rather than in the code being observed.
 *
 * The frame debugger (DELTA_GPU_CAPTURE, ../DEBUGGER.md) is the richer tool.
 * These answer questions that span a whole run rather than one frame.
 */

#include <cstdint>

#include "gpu/gcn/gcn_resource.h"
#include "gpu/gcn/gcn_translate.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/rhi/command.h"

namespace gpu::ps4 {

// Why a draw did or did not reach the recompiled-shader path. A draw that
// renders nothing looks the same either way from the outside.
enum class RecompStatus {
  kOk,
  kDisabled,      // DELTA_GPU_RECOMP=0
  kSkipped,       // DELTA_GPU_SKIPSH names this VS or PS
  kBadAddress,    // VS/PS program address outside the guest window
  kRejected,      // the translator declined the shader pair
  kBadAttrs,      // an attribute's V# did not resolve
  kAttrBindings,  // more distinct vertex buffers than bindings
  kAttrOffset,    // an attribute lies outside its binding's record
};
const char* RecompStatusName(RecompStatus status);

// --- register file ---------------------------------------------------------

// `packet_base` is the packet's register base, zero for a type-0 write. Keeps
// the CB_SHADER_MASK / CB_TARGET_MASK write counts (a zero mask reads the same
// whether it was cleared or never written) and the per-register source pointer
// DELTA_GPU_REGSRC reports.
void NoteRegisterWrites(uint32_t first_reg,
                        const uint32_t* values,
                        uint32_t count,
                        uint32_t packet_base);

// --- draws -----------------------------------------------------------------

// DELTA_GPU_ROOT_WPROT_*: watch the guest memory a shader resource root lives
// in. Armed once, on the first draw whose PS matches the address or code hash.
void MaybeArmRootWriteWatch(const Regs& regs, uint64_t ps_addr, uint32_t frame);
// Installed by the composition root (SetWriteWatchCallback); the watch above is
// its only user.
using WriteWatchFn = void (*)(uintptr_t, size_t, unsigned, bool, bool);
void SetWriteWatch(WriteWatchFn watch);

// DELTA_GPU_REGSRC_PS: the user-data SGPRs of one pixel shader with the packet
// word each was written from.
void TraceRegisterSources(const Regs& regs, uint64_t ps_addr, uint32_t frame);

// DELTA_GPU_TRACE: the first pixel shader that samples a texture, with its
// user data and what TrackTextures made of it.
void TraceFirstTexturedPs(const Regs& regs, uint64_t ps_addr);

// DELTA_GPU_DRAWPKT: which opcode each draw arrives as, and how each index
// packet was decoded. A count that never leaves the packet shows up downstream
// as "too few vertices", which points at the wrong layer.
void TraceDrawOpcode(uint32_t op, uint32_t prim_type, uint32_t auto_count);
void TraceIndexBuffer(uint64_t index_base,
                      uint32_t index_count,
                      uint32_t index_type,
                      bool accepted);
void TraceIndirectArgs(uint32_t op,
                       uint64_t args_addr,
                       uint64_t indirect_base,
                       uint32_t offset,
                       bool mapped,
                       const uint32_t args[3]);
void TraceIndexOffsetArgs(uint32_t max_size,
                          uint32_t index_offset,
                          uint32_t index_count,
                          uint64_t index_base,
                          uint64_t resolved_base,
                          uint32_t index_type,
                          bool accepted);

// DELTA_GPU_NOMRT: draws whose colour targets do not add up. A write mask
// enabling a target the CB registers never name renders into nothing, and a gap
// at slot 0 drops the draw downstream; both show up only as a black target.
void TraceMrtSlotZeroGap(const rhi::DrawInfo& d,
                         uint32_t target_mask,
                         uint64_t vs_addr,
                         uint64_t ps_addr);
void TraceNoMrtBound(const Regs& regs,
                     const rhi::DrawInfo& d,
                     uint32_t target_mask,
                     uint64_t vs_addr,
                     uint64_t ps_addr);

// DELTA_GPU_DBTRACE: the depth/stencil registers of every depth-testing draw.
// `z_info` is the value the draw path acted on, which DELTA_GPU_NODEPTH forces
// to zero.
void TraceDepthState(const Regs& regs, uint32_t z_info);

// DELTA_GPU_DBWATCH: does an address ever appear in a DB base register? A
// surface read as depth that nothing writes is either CPU-filled or a plane the
// depth path never registers; only the registers tell those apart.
void TraceDepthBaseWatch(const Regs& regs);

// DELTA_GPU_TEXTRACK_FRAME: whether TrackTextures should run in tracing mode
// for this draw.
bool ShouldTraceTextureTracking(uint32_t frame, uint64_t ps_addr);

// DELTA_GPU_TEXFMT: the format/tiling of sampled textures, to pin a scrambled
// draw to an encoding we mishandle.
void TraceTextureFormat(const gcn::TImage& tex, const rhi::DrawInfo& d);

// DELTA_GPU_PSINCNTL: the VS parameter export each PS input slot reads.
void TracePsInputCntl(const Regs& regs,
                      uint64_t ps_addr,
                      uint32_t ps_input_ena,
                      const uint32_t* ps_in_cntl);

// DELTA_GPU_SHRELOC: attribute a recompile-cache miss to the key field that
// changed since this VS/PS content pair was last compiled. The distinct-value
// tallies say whether a churning field is bounded (the cache converges) or
// unbounded (it misses forever).
void TraceShaderCacheMiss(uint64_t vs_addr,
                          uint64_t ps_addr,
                          uint64_t vs_hash,
                          uint64_t ps_hash,
                          uint64_t fetch_hash,
                          uint32_t ps_input_ena,
                          uint32_t tex_3d_mask,
                          uint32_t tex_1d_mask);

// DELTA_GPU_RAWBUF: the V# behind each buffer a recompiled stage reads by hand.
void TraceRawBuffer(const char* stage,
                    uint64_t vs_addr,
                    const gcn::ShaderBuffer& buffer,
                    const gcn::VBuffer& resolved,
                    uint64_t bytes,
                    bool accepted,
                    const rhi::DrawInfo& d);

// DELTA_GPU_BLITDUMP: the full state of the first few draws into a
// scanout-sized target, with the PS listing and its resolved constants.
void TraceBlitDraw(const Regs& regs,
                   const rhi::DrawInfo& d,
                   uint64_t vs_addr,
                   uint64_t ps_addr);

// DELTA_GPU_DRAWLIST: one line per draw before any gating, so draws dropped for
// unresolved vertex data or shaders are visible.
void TraceDrawList(const Regs& regs,
                   const rhi::DrawInfo& d,
                   uint64_t vs_addr,
                   uint64_t ps_addr,
                   uint64_t fetch_addr,
                   RecompStatus status);

// DELTA_GPU_MASKTRACE: the channels a draw writes are (CB_TARGET_MASK &
// CB_SHADER_MASK) nibble n. We read the first to decide whether a target is
// bound and never read the second, so this puts both next to CB_COLOR_CONTROL
// and the recompiler's own export mask.
void TraceColorMasks(const Regs& regs,
                     const rhi::DrawInfo& d,
                     uint64_t ps_addr,
                     RecompStatus status);

// DELTA_GPU_SPRITEDUMP: transform, vertices and textures of the first few
// textured draws: degenerate MVP vs UV=0 vs blend.
void TraceSpriteDraw(const rhi::DrawInfo& d);

// DELTA_GPU_VATTRDUMP: the raw first-vertex bytes per attribute of small
// multi-attribute draws, to separate "the colours in guest memory are zero"
// from "the fetch path zeroes them".
void TraceVertexAttrs(const rhi::DrawInfo& d);

// DELTA_GPU_GEOMDUMP: high-index (world geometry) draws, with their vertices
// projected through the resolved transform to say whether the geometry is on
// screen at all.
void TraceWorldGeometry(const Regs& regs, const rhi::DrawInfo& d);

// DELTA_GPU_TRACE: the register state behind a draw, plus a one-time probe of
// the shader binaries, their user data and the descriptor tables they point at.
void TraceDrawRegisters(const Regs& regs,
                        uint32_t op,
                        const uint32_t* body,
                        uint32_t count);

// --- compute ---------------------------------------------------------------

// DELTA_GPU_CSDUMP: the listing and user data of each distinct compute shader.
void TraceComputeShader(const Regs& regs,
                        uint64_t cs_addr,
                        const uint32_t groups[3],
                        const uint32_t threads[3],
                        uint32_t user_sgpr,
                        uint32_t tgid_enable,
                        uint32_t lds_dwords);

// DELTA_GPU_CSDROPS: dispatches refused before they could reach a capture,
// where one the guest issued and we dropped looks like one never issued.
void TraceDroppedDispatch(uint64_t cs_addr,
                          const uint32_t groups[3],
                          const uint32_t threads[3],
                          const char* reason);

// DELTA_GPU_CSRES=1 traces the first dispatch of each distinct shader; =<addr>
// every dispatch of that one shader, since a streaming copy runs thousands of
// times with a different destination and the first says nothing about the rest.
bool ShouldTraceCsResources(uint64_t cs_addr);
// DELTA_GPU_CSWATCH=<addr>: also report any resource whose range covers it.
bool CsWatchCovers(uint64_t base, uint64_t size);

void TraceCsUnresolved(uint64_t cs_addr, const gcn::CsResource& res);
// DELTA_GPU_EUDFAIL: raw code of an unresolvable CS, once, so its descriptor
// chain can be decoded offline.
void TraceCsCode(uint64_t cs_addr);
void TraceCsZeroFill(uint64_t cs_addr,
                     uint32_t binding,
                     const gcn::TImage& image,
                     const uint32_t* descriptor);
void TraceCsUnsupportedImage(uint64_t cs_addr,
                             uint32_t binding,
                             const gcn::TImage& image,
                             const uint32_t* descriptor);
void TraceCsWindowedBuffer(uint64_t cs_addr,
                           uint32_t binding,
                           uint64_t declared_size,
                           uint64_t mapped_size);
void TraceCsResource(uint64_t cs_addr,
                     const gcn::CsResource& res,
                     uint64_t base,
                     uint64_t size,
                     uint64_t guest_size,
                     bool image_staging,
                     const gcn::TImage& image,
                     uint32_t elem_bytes,
                     uint32_t stage_elem_bytes);
void TraceCsInvalidRange(uint64_t cs_addr,
                         uint32_t binding,
                         uint64_t base,
                         uint64_t guest_size);
void TraceCsDispatch(uint64_t cs_addr, bool executed, uint32_t num_resources);

// --- the packet stream -----------------------------------------------------

// Every type-3 packet, for the opcode histogram and the per-opcode time
// DELTA_GPU_DCBSTAT reports.
void NotePacket(uint32_t op);
void NotePacketCost(uint32_t op, uint64_t ns);
// The timing costs two clock reads per packet, so the walk asks first.
bool WantPacketCost();
void DumpOpcodeHistogram();

// DELTA_GPU_CETRACE: every constant-engine RAM packet and whether it was
// applied. A descriptor table the CE publishes and we drop reads back as zeros,
// which looks exactly like a resolver bug.
void TraceConstRam(const char* what,
                   uint32_t ce_offset,
                   uint32_t num_dwords,
                   uint64_t addr,
                   const char* verdict,
                   uint32_t first_dword);

// DELTA_GPU_CCBHIST: what the constant command buffers are made of. The
// histogram is reported after the walk it is labelled with, not before.
void NoteCcbPacket(uint32_t op);
void TraceCcbSubmit(uint32_t size_bytes, uint32_t words);
void TraceCcbHistogram(uint32_t words);

// DELTA_GPU_WAITTRACE: how many WAIT_REG_MEM polls were satisfied and how many
// timed out. A timeout means the word polled is one we write but had not yet.
void TraceWaitRegMem(bool timed_out);

// DELTA_GPU_DMATRACE: CP DMA packets by class. Shader prefetches (src == dst)
// outnumber real transfers by thousands to one, so they are counted rather than
// printed.
void TraceDmaData(uint32_t control,
                  uint32_t command,
                  uint32_t src_sel,
                  uint32_t dst_sel,
                  uint64_t src,
                  uint64_t dst,
                  uint32_t bytes,
                  bool copied);

// DELTA_GPU_ADDRWATCH=<addr>: name the packet that writes it. `max_lines` is
// this packet class's own budget, not a share of a common one.
void TraceAddrWatch(const char* packet,
                    uint64_t dst,
                    uint64_t bytes,
                    uint32_t first_dword,
                    uint32_t max_lines);

// DELTA_GPU_EOPTRACE: every completion label this command processor writes.
void TraceLabelWrite(const char* packet,
                     uint64_t addr,
                     uint32_t data_sel,
                     uint64_t value);
// EOS carries no DATA_SEL, so it reports one field fewer.
void TraceEosLabel(uint64_t addr, uint32_t value);
void TraceDataWrite(uint64_t addr, uint32_t dwords, uint32_t first_dword);

// DELTA_GPU_OPTRACE: PM4 opcodes nothing handles. Register state set through a
// packet we ignore reads back zero, exactly as if the guest never set it.
void TraceUnhandledOpcode(uint32_t op, uint32_t count);

// DELTA_GPU_IBTRACE: chained indirect buffers, followed or skipped.
void TraceIndirectBuffer(uint32_t position,
                         uint32_t depth,
                         uint32_t words,
                         bool followed);

// DELTA_GPU_COUNTERTRACE: the CE/DE counter packets.
void TraceCounter(const char* packet, uint64_t value);
void TraceWaitOnCeCounter(uint32_t wanted, uint64_t ce_counter);

// A type-1 header, which is a genuine desync. Reported under DELTA_GPU_DESYNC,
// or unconditionally while a submission is being dumped packet by packet.
void TraceDesync(uint32_t position,
                 uint32_t words,
                 uint32_t type,
                 uint32_t hdr,
                 bool force);

// --- submissions -----------------------------------------------------------

// Whether this submission should be dumped packet by packet: the first large
// (real rendering) command buffer under DELTA_GPU_TRACE.
bool ShouldDumpDcb(uint32_t size_bytes);
void TraceDcbPacket(uint32_t position, uint32_t op, uint32_t count);
void TraceSubmit(const void* dcb,
                 uint32_t size_bytes,
                 uint32_t words,
                 uint64_t submit_number,
                 uint64_t draws_so_far);
// After a dumped walk: how far it got, plus a brute scan for draw opcodes the
// walker may have desynced past.
void TraceDcbWalkResult(const uint32_t* dcb,
                        uint32_t words,
                        uint32_t words_walked);
// DELTA_GPU_DCBSTAT: what the command stream is made of and which handler owns
// the time. DELTA_GPU_OPHIST: the cumulative histogram, once, deep into a run.
void TraceDcbStat(uint32_t words);
void MaybeDumpOpcodeHistogram(uint32_t words_walked, uint32_t words);

}  // namespace gpu::ps4
