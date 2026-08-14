#pragma once

/*
 * PS4Delta : PS4 emulation and research project
 *
 * Copyright 2019-2020 Force67.
 * For information regarding licensing see LICENSE
 * in the root of the source tree.
 */

#include "base/arch.h"
#include <base/containers/vector.h>

namespace gfx {
class frameBase {
public:
  virtual ~frameBase() = default;
  virtual void toggleFullscreen() = 0;
  virtual void takeScreenshot(base::Vector<u8> &data, u32 sizeX,
                              u32 sizeY) = 0;
};
}
