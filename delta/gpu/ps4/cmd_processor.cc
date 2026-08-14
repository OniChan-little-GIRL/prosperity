/*
 * PS4Delta : PS4 emulation and research project
 *
 * The PM4 packet stream: the DE and CE walks and the state they carry. See
 * cmd_processor.h.
 */

#include "gpu/ps4/cmd_processor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>

#include <utl/mem.h>
#include <utl/options.h>

#include "gpu/gcn/gcn_decode.h"
#include "gpu/gcn/gcn_resource.h"
#include "gpu/ps4/cmd_trace.h"
#include "gpu/ps4/compute_dispatch.h"
#include "gpu/ps4/draw_state.h"
#include "gpu/ps4/guest_address.h"
#include "gpu/ps4/liverpool.h"
#include "gpu/ps4/pm4.h"
#include "gpu/rhi/renderer.h"

namespace {
DELTA_OPTION(bool, kCeOn, "DELTA_GPU_CE", true);
DELTA_OPTION(bool, kNoCopy, "DELTA_GPU_NODMACOPY", false);
}  // namespace

namespace gpu::rhi {
// Declared in rhi/command.h; the frame-time overlay reports them.
uint64_t g_ns_dcb = 0;
uint64_t g_ns_dcb_lock = 0;
uint32_t g_dcb_n = 0;
}  // namespace gpu::rhi

namespace gpu::ps4 {
namespace {

// The register file is the state of one GPU: two submit threads walking it
// concurrently would interleave one draw's registers with another's.
std::mutex g_mutex;
// Persistent across submits: Gnm programs a register once and relies on it
// holding for every later submission.
Regs g_regs;
std::atomic<uint64_t> g_total_submits{0};
std::atomic<uint64_t> g_total_draws{0};
bool g_renderer_started = false;
bool g_frame_active = false;
uint32_t g_presented_frames = 0;

// Latched by the IT_* packets that precede a draw and consumed by it.
struct IndexState {
  uint32_t type = 0;  // VGT_DMA_INDEX_TYPE[1:0]: 0 = 16-bit, 1 = 32-bit
  uint64_t base = 0;  // IT_INDEX_BASE (DRAW_INDEX_2 carries its own)
  uint64_t indirect_base = 0;  // IT_SET_BASE(1): where indirect args live
  uint32_t num_instances = 1;  // IT_NUM_INSTANCES, for the following draw(s)
};
IndexState g_index;

// CE/DE synchronization counters. On real hardware the DE waits on the CE's
// counter before a dependent draw. Our submit is synchronous and the CCB (CE
// work) always runs before the DCB (DE work), so WAIT_ON_CE_COUNTER is always
// already satisfied; the counters are tracked so the stream state matches and
// the packets are never mistaken for a desync.
uint64_t g_ce_counter = 0;
uint64_t g_de_counter = 0;

// A cycle in the IB chain would recurse until the stack overflowed. Real
// submissions are flat or a couple of levels deep.
constexpr uint32_t kMaxIbDepth = 8;

// --- completion labels -----------------------------------------------------

// The guest words this command processor writes: the EOP / EOS / RELEASE_MEM /
// WRITE_DATA fence labels a title's CPU threads and other rings poll to order
// themselves after this one. A WAIT_REG_MEM on anything else is waiting for a
// producer we do not run, and spinning on that buys nothing.
class FenceLabels {
 public:
  void Note(uint64_t address) {
    if (!address)
      return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (addresses_.size() < 4096)
      addresses_.insert(address & ~3ull);
  }
  bool Contains(uint64_t address) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return addresses_.count(address & ~3ull) != 0;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_set<uint64_t> addresses_;
};
FenceLabels g_fence_labels;

// Labels live in guest memory the game allocated (Garlic/Onion), in low guest
// heaps, or in the GnmDriver area. Accept any plausibly-mapped, non-low
// address; reject null/garbage.
bool IsLabelAddress(uint64_t address) {
  return address >= 0x10000ull && address < kGuestEnd;
}

// Our submit is synchronous: every draw in the buffer is finished by the time
// the walk passes these packets, so the fence the GPU would signal is complete
// the instant we process it. Writing it immediately is what lets the guest's
// CPU-side polls (the flip-done / submit-done labels Gnm spins on between
// frames) make progress. Without it the title stalls once the few in-flight
// display buffers drain.
void WriteLabel(uint64_t address, uint64_t value, bool is_64bit) {
  g_fence_labels.Note(address);
  if (!IsLabelAddress(address))
    return;
  if (is_64bit)
    *reinterpret_cast<volatile uint64_t*>(address) = value;
  else
    *reinterpret_cast<volatile uint32_t*>(address) =
        static_cast<uint32_t>(value);
}

// EOP/RELEASE_MEM DATA_SEL 3 (GPU clock) and 4 (system clock) tell the GPU to
// write its current 64-bit clock counter into the label, NOT the packet's
// immediate data (which is 0 for these). A title polling such a label for
// "non-zero == the GPU reached this point" needs a real, monotonically
// increasing, non-zero value; our submit is synchronous, so any advancing clock
// reads as "already complete". Without this Doom64's per-frame submit-done wait
// (a spin with a hard 2s timeout) burns the full 2s every frame -> ~0.5 fps.
uint64_t GpuClockTimestamp() {
  using namespace std::chrono;
  return static_cast<uint64_t>(
      duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
          .count());
}

// The label write shared by EOP and RELEASE_MEM, which encode DATA_SEL the same
// way: 1 = 32-bit immediate, 2 = 64-bit immediate, 3/4 = a clock counter.
void WriteEventLabel(const char* packet,
                     uint64_t address,
                     uint32_t data_sel,
                     uint64_t value) {
  if (data_sel == 1)
    WriteLabel(address, value, false);
  else if (data_sel == 2)
    WriteLabel(address, value, true);
  else if (data_sel >= 3)
    WriteLabel(address, GpuClockTimestamp(), true);
  TraceLabelWrite(packet, address, data_sel, value);
}

// --- constant engine RAM ---------------------------------------------------

// Liverpool's Constant Engine RAM: 48 KiB of on-chip scratch the CE fills and
// dumps to guest memory as the shaders' constant buffers. Every access is
// bounds checked here so a malformed packet can never write outside it.
class ConstRam {
 public:
  bool Fits(uint32_t offset, uint32_t dwords) const {
    return (uint64_t)offset + (uint64_t)dwords * 4 <= sizeof(data_);
  }
  void Write(uint32_t offset, const void* src, uint32_t dwords) {
    std::memcpy(data_ + offset, src, (size_t)dwords * 4);
  }
  void Read(uint32_t offset, void* dst, uint32_t dwords) const {
    std::memcpy(dst, data_ + offset, (size_t)dwords * 4);
  }
  uint32_t DwordAt(uint32_t offset) const {
    uint32_t value;
    std::memcpy(&value, data_ + offset, sizeof(value));
    return value;
  }

 private:
  uint8_t data_[48 * 1024] = {};
};
ConstRam g_const_ram;

// --- register writes -------------------------------------------------------

void SetRegs(uint32_t base, const uint32_t* body, uint32_t count) {
  if (!count)
    return;
  // Indexed SET packets use bits 28..31 for the index; only the low 16 bits are
  // the register offset. Treating the whole word as an offset drops Neo
  // register writes such as 0x40000258.
  const uint32_t first = Pm4SetRegAddress(base, body[0]);
  const uint32_t values = count - 1;
  for (uint32_t i = 0; i < values; i++)
    if (first + i < kRegFileSize)
      g_regs[first + i] = body[1 + i];
  NoteRegisterWrites(first, &body[1], values, base);
}

// --- packet handlers -------------------------------------------------------

// The guest's only way to make this ring wait for another engine: poll a word
// (or a register) until it satisfies a comparison. P.T. issues 86,611 of them a
// run and every one was ignored, so with the async compute rings walked on
// their own thread a draw could run before the dispatch that produced what it
// samples.
//
// body: [0] function/space, [1] addr lo or reg offset, [2] addr hi,
// [3] reference, [4] mask, [5] poll interval.
void HandleWaitRegMem(const uint32_t* body, uint32_t count) {
  if (count < 5)
    return;
  const uint32_t function = body[0] & 0x7;
  const bool memory_space = ((body[0] >> 4) & 1) != 0;
  const uint32_t reference = body[3], mask = body[4];
  const auto passes = [&](uint32_t polled) {
    const uint32_t a = polled & mask, b = reference & mask;
    switch (function) {
      case 1:
        return a < b;
      case 2:
        return a <= b;
      case 3:
        return a == b;
      case 4:
        return a != b;
      case 5:
        return a >= b;
      case 6:
        return a > b;
      default:
        return true;  // 0 = always, 7 = reserved
    }
  };

  const volatile uint32_t* polled = nullptr;
  if (memory_space) {
    const uint64_t address =
        ((static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1]) & ~3ull;
    // Only where we are the producer. A poll on a word nothing of ours writes
    // can never be satisfied, and waiting out its timeout is pure loss.
    if (IsGuestAddress(address) && g_fence_labels.Contains(address) &&
        utl::isMemoryRangeMapped(reinterpret_cast<const void*>(address), 4))
      polled = reinterpret_cast<const volatile uint32_t*>(address);
  } else if ((body[1] & 0xFFFF) < kRegFileSize) {
    polled = &g_regs[body[1] & 0xFFFF];
  }
  if (!polled)
    return;

  // Bounded, because a producer can still be one we dropped and an unbounded
  // poll would hang the title outright. Yield rather than spin: the thread that
  // will satisfy this needs the core.
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::microseconds(200);
  bool timed_out = false;
  while (!passes(*polled)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      timed_out = true;
      break;
    }
    std::this_thread::yield();
  }
  TraceWaitRegMem(timed_out);
}

// CP DMA. body: ctrl, srcLo/Hi, dstLo/Hi, command(byteCount). Doom64 uploads
// its level texture atlases this way, so without performing the copy the T#
// addresses stay zero and the 3D world samples blank textures. ctrl word:
// SRC_SEL[30:29], DST_SEL[21:20]; sel 0/3 = memory address, 2 = immediate data
// (a fill, not a copy); only true mem->mem is copied.
void HandleDmaData(rhi::Renderer& renderer,
                   const uint32_t* body,
                   uint32_t count) {
  if (count < 6)
    return;
  const uint32_t control = body[0];
  const uint32_t src_sel = (control >> 29) & 0x3;
  const uint32_t dst_sel = (control >> 20) & 0x3;
  const uint64_t src =
      (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1];
  const uint64_t dst =
      (static_cast<uint64_t>(body[4] & 0xFFFF) << 32) | body[3];
  const uint32_t bytes = body[5] & 0x1FFFFF;
  const bool src_is_memory = src_sel == 0 || src_sel == 3;
  const bool dst_is_memory = dst_sel == 0 || dst_sel == 3;
  // Only copy between REAL guest memory: the sel bits report "memory" even for
  // GDS/register targets (e.g. dst=0x3022c), which are not mapped in our
  // address space and segfault. Every real guest allocation sits far above
  // 16 MiB, so that floor excludes the on-chip GDS/low targets while keeping
  // texture/buffer uploads.
  const auto addressable = [](uint64_t address) {
    return address >= 0x1000000ull && address < kGuestEnd;
  };
  bool copied = false;
  if (!kNoCopy && src_is_memory && dst_is_memory && bytes &&
      bytes <= 0x1000000u && src != dst && addressable(src) &&
      addressable(src + bytes) && addressable(dst) &&
      addressable(dst + bytes)) {
    // src may be CS-written; land pending writes first. Copy even if the flush
    // fails: a possibly-stale source beats silently dropping the copy.
    rhi::FlushCsWrites(renderer);
    std::memcpy(reinterpret_cast<void*>(dst),
                reinterpret_cast<const void*>(src), bytes);
    copied = true;
  }
  TraceDmaData(control, body[5] & ~0x1fffffu, src_sel, dst_sel, src, dst, bytes,
               copied);
}

// Gnm writes its 32/64-bit submit/flip fence labels with WRITE_DATA. The
// control field's dst_sel encoding varies (the flip-label packet built by
// sceGnmInsertFlip uses control=5, not the [11:8]=memory form), so don't gate
// on dst_sel: a memory write resolves to a real guest label address, while a
// register write yields a tiny offset IsLabelAddress rejects.
// body: control, dstLo, dstHi, data...
void HandleWriteData(const uint32_t* body, uint32_t count) {
  if (count >= 3)
    TraceAddrWatch("WRITE_DATA",
                   (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1],
                   (count - 3) * 4u, count >= 4 ? body[3] : 0,
                   /*max_lines=*/12);
  if (count < 4)
    return;
  const uint64_t address =
      (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const uint32_t dwords = count - 3;
  if (IsLabelAddress(address) &&
      IsLabelAddress(address + (uint64_t)dwords * 4)) {
    std::memcpy(reinterpret_cast<void*>(address), &body[3], (size_t)dwords * 4);
    g_fence_labels.Note(address);
  }
  TraceDataWrite(address, dwords, dwords ? body[3] : 0);
}

// body: eventCtrl, addrLo, addrHi+sel, dataLo, dataHi
void HandleEventWriteEop(const uint32_t* body, uint32_t count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOP",
                   (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, /*max_lines=*/8);
  if (count < 4)
    return;
  const uint64_t address =
      (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  const uint64_t value =
      static_cast<uint64_t>(body[3]) |
      (static_cast<uint64_t>(count >= 5 ? body[4] : 0) << 32);
  WriteEventLabel("EOP", address, (body[2] >> 29) & 0x7, value);
}

// body: eventCtrl, selBits, addrLo, addrHi, dataLo, dataHi
void HandleReleaseMem(const uint32_t* body, uint32_t count) {
  if (count >= 4)
    TraceAddrWatch("RELEASE_MEM",
                   (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) | body[2], 8,
                   count >= 5 ? body[4] : 0, /*max_lines=*/8);
  if (count < 5)
    return;
  const uint64_t address =
      (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) | (body[2] & ~0x3u);
  const uint64_t value =
      static_cast<uint64_t>(body[4]) |
      (static_cast<uint64_t>(count >= 6 ? body[5] : 0) << 32);
  WriteEventLabel("RELEASE_MEM", address, (body[1] >> 29) & 0x7, value);
}

// body: eventCtrl, addrLo, addrHi+cmd, data
void HandleEventWriteEos(const uint32_t* body, uint32_t count) {
  if (count >= 3)
    TraceAddrWatch("EVENT_WRITE_EOS",
                   (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | body[1], 8,
                   count >= 4 ? body[3] : 0, /*max_lines=*/8);
  if (count < 4)
    return;
  const uint64_t address =
      (static_cast<uint64_t>(body[2] & 0xFFFF) << 32) | (body[1] & ~0x3u);
  WriteLabel(address, body[3], false);
  TraceEosLabel(address, body[3]);
}

void HandleDrawPacket(rhi::Renderer& renderer,
                      uint32_t op,
                      const uint32_t* body,
                      uint32_t count) {
  g_total_draws.fetch_add(1);
  // Ahead of the renderer gate: the watch these can arm is a kernel one, and a
  // run whose device failed to come up is when register state is worth having.
  const uint32_t frame = g_presented_frames + 1;
  const uint64_t ps_addr = g_regs.ShaderAddr(mmSPI_SHADER_PGM_LO_PS);
  MaybeArmRootWriteWatch(g_regs, ps_addr, frame);
  TraceRegisterSources(g_regs, ps_addr, frame);
  TraceFirstTexturedPs(g_regs, ps_addr);

  if (renderer.available()) {
    DrawPacket packet;
    packet.op = op;
    packet.body = body;
    packet.count = count;
    packet.index_type = g_index.type;
    packet.index_base = g_index.base;
    packet.indirect_base = g_index.indirect_base;
    packet.num_instances = g_index.num_instances;
    packet.frame = frame;

    rhi::DrawInfo d;
    const bool renderable = BuildDrawInfo(renderer, g_regs, packet, d);
    // On the first draw of a frame, and for every draw the renderer sees
    // including the ones dropped below, so a frame whose draws all decline
    // still ends (and presents) like any other.
    if (!g_frame_active) {
      rhi::BeginFrame(renderer);
      g_frame_active = true;
    }
    if (renderable)
      rhi::Draw(renderer, d);
  }
  TraceDrawRegisters(g_regs, op, body, count);
}

bool IsDraw(uint32_t op) {
  return op == IT_DRAW_INDEX_AUTO || op == IT_DRAW_INDEX_2 ||
         op == IT_DRAW_INDEX_OFFSET_2 || op == IT_DRAW_INDIRECT ||
         op == IT_DRAW_INDEX_INDIRECT || op == IT_DRAW_INDEX_MULTI_AUTO;
}

// --- the walks -------------------------------------------------------------

// Resolve an in-stream IT_INDIRECT_BUFFER body into a mapped host pointer and
// dword count. Mirrors the kernel's gc_insert_indirect_buffer checks: a
// non-zero ib_size whose GPU address sits in the guest range and is actually
// mapped. Any failure returns false so the caller skips the chain instead of
// dereferencing garbage.
bool ResolveIndirectBuffer(const uint32_t* body,
                           uint32_t count,
                           const uint32_t*& out,
                           uint32_t& out_dwords) {
  out_dwords = 0;
  if (count < 3)
    return false;
  const uint64_t address =
      (static_cast<uint64_t>(body[1] & 0xFF) << 32) | body[0];
  const uint32_t dwords = body[2] & 0xFFFFF;
  const uint64_t bytes = static_cast<uint64_t>(dwords) * 4;
  if (!IsGuestRange(address, bytes))
    return false;
  const void* p = reinterpret_cast<const void*>(address);
  if (!utl::isMemoryRangeMapped(p, bytes))
    return false;
  out = static_cast<const uint32_t*>(p);
  out_dwords = dwords;
  return true;
}

uint32_t WalkDcb(rhi::Renderer& renderer,
                 const uint32_t* p,
                 uint32_t words,
                 uint32_t depth,
                 bool dump);

// Walk one CCB (CE stream), recursing into any chained indirect buffers at
// `depth`. The CE runs ahead of the draw engine: it fills its on-chip RAM and
// dumps it to the guest memory the DE's draws then read as constant buffers.
void WalkCcb(rhi::Renderer& renderer,
             const uint32_t* p,
             uint32_t words,
             uint32_t depth) {
  uint32_t i = 0;
  while (i < words) {
    const uint32_t hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    if (type != Pm4Type::kType3) {
      if (type == Pm4Type::kType2 || hdr == 0)
        i += 1;
      else if (type == Pm4Type::kType0)
        i += 1 + Pm4Count(hdr);
      else
        break;  // type-1 desync
      continue;
    }
    const uint32_t op = Pm4Opcode(hdr), count = Pm4Count(hdr);
    const uint32_t* body = &p[i + 1];
    if (i + 1 + count > words)
      break;
    NoteCcbPacket(op);
    switch (op) {
      case IT_WRITE_CONST_RAM: {  // body[0] = byte offset, body[1..] = data
        if (!kCeOn)
          break;
        const uint32_t offset = body[0] & 0xFFFF;
        const uint32_t dwords = count > 1 ? count - 1 : 0;
        const bool fits = g_const_ram.Fits(offset, dwords);
        if (fits)
          g_const_ram.Write(offset, &body[1], dwords);
        TraceConstRam("write", offset, dwords, 0, fits ? "ok" : "off+n>ceram",
                      dwords ? body[1] : 0);
        break;
      }
      case IT_LOAD_CONST_RAM: {  // addrLo, addrHi, num_dwords, byte offset
        if (!kCeOn || count < 4)
          break;
        const uint64_t address =
            (static_cast<uint64_t>(body[1] & 0xFFFF) << 32) | body[0];
        const uint32_t dwords = body[2] & 0x7FFF, offset = body[3] & 0xFFFF;
        const bool in_guest = IsGuestRange(address, (uint64_t)dwords * 4);
        const bool fits = g_const_ram.Fits(offset, dwords);
        if (in_guest && fits)
          g_const_ram.Write(offset, reinterpret_cast<const void*>(address),
                            dwords);
        TraceConstRam(
            "load", offset, dwords, address,
            !in_guest ? "addr-not-guest"
            : !fits   ? "off+n>ceram"
                      : "ok",
            in_guest ? *reinterpret_cast<const uint32_t*>(address) : 0);
        break;
      }
      case IT_DUMP_CONST_RAM:
      case IT_DUMP_CONST_RAM_OFFSET: {  // offset, num_dwords, addrLo, addrHi
        if (!kCeOn || count < 4)
          break;
        const uint32_t offset = body[0] & 0xFFFF, dwords = body[1] & 0x7FFF;
        const uint64_t address =
            (static_cast<uint64_t>(body[3] & 0xFFFF) << 32) | body[2];
        const bool in_guest = IsGuestRange(address, (uint64_t)dwords * 4);
        const bool fits = g_const_ram.Fits(offset, dwords);
        if (in_guest && fits)
          g_const_ram.Read(offset, reinterpret_cast<void*>(address), dwords);
        TraceConstRam(op == IT_DUMP_CONST_RAM ? "dump" : "dump.off", offset,
                      dwords, address,
                      !in_guest ? "addr-not-guest"
                      : !fits   ? "off+n>ceram"
                                : "ok",
                      fits ? g_const_ram.DwordAt(offset) : 0);
        break;
      }
      case IT_INCREMENT_CE_COUNTER:  // the DE later waits on this value
        g_ce_counter++;
        break;
      case IT_INDIRECT_BUFFER_CNST:
      case IT_INDIRECT_BUFFER: {
        // The CE can chain further const buffers; follow them so chained
        // WRITE/LOAD/DISPATCH work is not silently dropped.
        const uint32_t* chain = nullptr;
        uint32_t chain_dwords = 0;
        if (depth < kMaxIbDepth &&
            ResolveIndirectBuffer(body, count, chain, chain_dwords)) {
          if (op == IT_INDIRECT_BUFFER_CNST)
            WalkCcb(renderer, chain, chain_dwords, depth + 1);
          else
            WalkDcb(renderer, chain, chain_dwords, depth + 1, false);
        }
        break;
      }
      default:
        break;
    }
    i += 1 + count;
  }
}

// Walk one DCB (DE stream), issuing draws and dispatches and recursing into any
// chained IT_INDIRECT_BUFFER at `depth`. Returns the walk position (dwords
// consumed) so the top-level caller can report how far it got.
uint32_t WalkDcb(rhi::Renderer& renderer,
                 const uint32_t* p,
                 uint32_t words,
                 uint32_t depth,
                 bool dump) {
  const bool time_packets = WantPacketCost();
  uint32_t i = 0;
  while (i < words) {
    const uint32_t hdr = p[i];
    const Pm4Type type = Pm4TypeOf(hdr);
    if (type == Pm4Type::kType2 || hdr == 0) {
      // Type-2 NOPs and the zero-dword alignment padding Gnm sprinkles between
      // packets; real packets resume after it.
      //
      // NOT the all-ones filler an async compute ring is initialised with:
      // 0xFFFFFFFF is a well-formed type-3 header (opcode 0xff, count 16384),
      // so it falls through to the packet path, where the truncation guard
      // stops the walk and every dispatch past the untouched ring is dropped.
      // Skipping it here would change which dispatches run, so that is a fix to
      // make and measure on its own.
      i += 1;
      continue;
    }
    if (type == Pm4Type::kType0) {
      // Type-0 writes a run of consecutive registers (base in hdr[15:0], count
      // in hdr[29:16]+1) directly into the register file. Treating this as a
      // desync and stopping dropped every later draw (the room floor) in any
      // command buffer that used type-0.
      const uint32_t count = Pm4Count(hdr);
      const uint32_t base = Pm4Type0Reg(hdr);  // absolute register offset
      const uint32_t available = std::min(count, words - i - 1);
      for (uint32_t k = 0; k < available; k++)
        if (base + k < kRegFileSize)
          g_regs[base + k] = p[i + 1 + k];
      NoteRegisterWrites(base, &p[i + 1], available, 0);
      i += 1 + count;
      continue;
    }
    if (type != Pm4Type::kType3) {
      TraceDesync(i, words, static_cast<uint32_t>(type), hdr, /*force=*/dump);
      break;  // a type-1 header is a genuine desync
    }

    const uint32_t op = Pm4Opcode(hdr);
    const uint32_t count = Pm4Count(hdr);  // body dword count
    const uint32_t* body = &p[i + 1];
    NotePacket(op);
    const auto op_start = time_packets
                              ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};
    if (dump)
      TraceDcbPacket(i, op, count);
    if (i + 1 + count > words)
      break;  // truncated / desync
    switch (op) {
      case IT_DISPATCH_DIRECT:
        DispatchCompute(renderer, g_regs, body, count);
        break;
      case IT_SET_CONTEXT_REG:
        SetRegs(kContextRegBase, body, count);
        break;
      case IT_SET_SH_REG:
      case IT_SET_SH_REG_INDEX:
        SetRegs(kShRegBase, body, count);
        break;
      case IT_SET_UCONFIG_REG:
        SetRegs(kUConfigRegBase, body, count);
        break;
      case IT_SET_CONFIG_REG:
        SetRegs(kConfigRegBase, body, count);
        break;
      case IT_INDEX_TYPE:
        if (count >= 1)
          g_index.type = body[0] & 0x3;
        break;
      case IT_INDEX_BASE:  // index buffer base (byte address) lo/hi
        if (count >= 2)
          g_index.base =
              (static_cast<uint64_t>(body[1] & 0xFF) << 32) | body[0];
        break;
      case IT_SET_BASE:
        // base_index 1 = DRAW_INDIRECT_BASE: where the indirect draws read
        // their argument structs from. body: baseIndex, addrLo, addrHi.
        if (count >= 3 && (body[0] & 0xF) == 1)
          g_index.indirect_base =
              (static_cast<uint64_t>(body[2] & 0xFF) << 32) | (body[1] & ~0x3u);
        break;
      case IT_NUM_INSTANCES:
        g_index.num_instances = (count >= 1 && body[0]) ? body[0] : 1;
        break;
      case IT_WAIT_REG_MEM:
        HandleWaitRegMem(body, count);
        break;
      case IT_DMA_DATA:
        HandleDmaData(renderer, body, count);
        break;
      case IT_WRITE_DATA:
        HandleWriteData(body, count);
        break;
      case IT_EVENT_WRITE_EOP:
        HandleEventWriteEop(body, count);
        break;
      case IT_RELEASE_MEM:
        HandleReleaseMem(body, count);
        break;
      case IT_EVENT_WRITE_EOS:
        HandleEventWriteEos(body, count);
        break;
      case IT_INDIRECT_BUFFER:
      case IT_INDIRECT_BUFFER_CNST: {  // chained buffer (nested CMDBUF)
        const uint32_t* chain = nullptr;
        uint32_t chain_dwords = 0;
        const bool followed =
            depth < kMaxIbDepth &&
            ResolveIndirectBuffer(body, count, chain, chain_dwords);
        TraceIndirectBuffer(i, depth, chain_dwords, followed);
        if (followed) {
          if (op == IT_INDIRECT_BUFFER_CNST)
            WalkCcb(renderer, chain, chain_dwords, depth + 1);
          else
            WalkDcb(renderer, chain, chain_dwords, depth + 1, dump);
        }
        break;
      }
      case IT_INCREMENT_CE_COUNTER:
        g_ce_counter = (g_ce_counter + 1) & 0xFFFFFF;
        TraceCounter("IT_INCREMENT_CE_COUNTER", g_ce_counter);
        break;
      case IT_INCREMENT_DE_COUNTER:
        g_de_counter = (g_de_counter + 1) & 0xFFFFFF;
        TraceCounter("IT_INCREMENT_DE_COUNTER", g_de_counter);
        break;
      case IT_WAIT_ON_CE_COUNTER:
        // The CE runs synchronously before the DCB, so the requested count is
        // already satisfied; only the state is kept.
        TraceWaitOnCeCounter(count >= 1 ? (body[0] & 0xFFFFFF) : 0,
                             g_ce_counter);
        break;
      default:
        if (IsDraw(op))
          HandleDrawPacket(renderer, op, body, count);
        else
          TraceUnhandledOpcode(op, count);
        break;
    }
    if (time_packets)
      NotePacketCost(op, std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now() - op_start)
                             .count());
    i += 1 + count;
  }
  return i;
}

// Wall time of one command-buffer walk, including the wait for the lock: a
// second submit thread blocked behind the first is time the guest is stalled on
// us either way.
struct ScopedWalkTimer {
  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();
  ~ScopedWalkTimer() {
    rhi::g_ns_dcb += std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
    rhi::g_dcb_n++;
  }
};

// The renderer comes up on the first submission rather than at startup: a title
// that never submits never needs a device.
void StartRendererOnce(rhi::Renderer& renderer) {
  if (g_renderer_started)
    return;
  g_renderer_started = true;
  rhi::Init(renderer);
  // The resource replay reads descriptor tables out of guest memory a compute
  // dispatch may still own; it is below the renderer, so it cannot ask itself.
  gcn::g_flush_guest_range = [](uint64_t address, uint64_t bytes) {
    rhi::FlushCsWritesRange(rhi::DefaultRenderer(), address, bytes);
  };
}

}  // namespace

void SetWriteWatchCallback(WriteWatchCallback callback) {
  SetWriteWatch(callback);
}

void SetPs4NeoMode(bool enabled) {
  std::lock_guard<std::mutex> lock(g_mutex);
  gcn::SetDefaultIsaMode(enabled ? gcn::IsaMode::kNeo : gcn::IsaMode::kBase);
}

void EndFrame(uint64_t scanout_base) {
  std::lock_guard<std::mutex> lock(g_mutex);
  // New frame -> shader code may have been rewritten; let CachedProgram
  // revalidate each address once next frame instead of once per draw.
  gcn::NextProgramCacheGeneration();
  rhi::Renderer& renderer = rhi::DefaultRenderer();
  if (!g_frame_active || !renderer.available())
    return;
  rhi::EndFrame(renderer, scanout_base);
  g_frame_active = false;
  g_presented_frames++;
}

void SubmitCcb(const void* ccb, uint32_t size_bytes) {
  if (!ccb || size_bytes < 4)
    return;
  std::lock_guard<std::mutex> lock(g_mutex);
  const uint32_t words = size_bytes / 4;
  TraceCcbSubmit(size_bytes, words);
  WalkCcb(rhi::DefaultRenderer(), static_cast<const uint32_t*>(ccb), words, 0);
  TraceCcbHistogram(words);
}

void SubmitDcb(const void* dcb, uint32_t size_bytes) {
  if (!dcb || size_bytes < 4)
    return;
  ScopedWalkTimer timer;
  // Time the wait for the lock apart from the walk: they mean opposite things,
  // one says "make the walk faster", the other "stop serialising the threads".
  const auto lock_start = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(g_mutex);
  rhi::g_ns_dcb_lock += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - lock_start)
                            .count();

  rhi::Renderer& renderer = rhi::DefaultRenderer();
  StartRendererOnce(renderer);

  const auto* p = static_cast<const uint32_t*>(dcb);
  const uint32_t words = size_bytes / 4;
  const uint64_t submission = g_total_submits.fetch_add(1) + 1;
  TraceSubmit(dcb, size_bytes, words, submission, g_total_draws.load());
  const bool dump = ShouldDumpDcb(size_bytes);
  TraceDcbStat(words);

  const uint32_t walked = WalkDcb(renderer, p, words, 0, dump);
  if (dump)
    TraceDcbWalkResult(p, words, walked);
  MaybeDumpOpcodeHistogram(walked, words);
}

}  // namespace gpu::ps4
