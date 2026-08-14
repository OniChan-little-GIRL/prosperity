#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * The window of the guest address space a PM4 packet may point at.
 *
 * Every address the command processor dereferences comes out of guest memory
 * the title filled: a V#/T# base, an index buffer, a fence label, a chained
 * command buffer. A descriptor that was never written reads as zero, one meant
 * for a resource we do not track reads as garbage, so each is checked here
 * before it becomes a pointer.
 *
 * A range test, not a mapping test: gpu/guest_memory.h answers whether the
 * bytes are readable right now and costs a syscall. Ask this one first.
 */

#include "base/arch.h"

namespace gpu::ps4 {

// Guest GPU allocations (Onion/Garlic) start at 0x10_0000_0000; the guest map
// ends below 0x200_0000_0000.
inline constexpr u64 kGuestBase = 0x1000000000ull;
inline constexpr u64 kGuestEnd = 0x20000000000ull;

inline bool IsGuestAddress(u64 address) {
  return address >= kGuestBase && address < kGuestEnd;
}

// A zero-byte range is not a range: it is a descriptor that says nothing. The
// `address < kGuestEnd` test is load-bearing, not implied by the one after it:
// without it `kGuestEnd - address` wraps and everything above the window
// passes.
inline bool IsGuestRange(u64 address, u64 bytes) {
  return bytes && address >= kGuestBase && address < kGuestEnd &&
         bytes <= kGuestEnd - address;
}

}  // namespace gpu::ps4
