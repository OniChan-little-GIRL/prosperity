#pragma once

/*
 * PS4Delta : PS4/PS5 emulation and research project
 *
 * /dev/random and /dev/urandom. Without them the open fails with ENOENT, and a
 * guest that seeds from one either runs with no entropy or takes a failure path
 * it was never meant to: Minecraft's V8 opens /dev/urandom while creating the
 * isolate for its gameplay view.
 */

#include "device.h"
#include "base/arch.h"

namespace krnl {
class proc;

class randomDevice : public device {
public:
  randomDevice(proc *);

  i64 read(void *buf, size_t len) override;
  i64 lseek(i64 off, int whence) override;
  int fstat(void *stat) override;
};
} // namespace krnl
