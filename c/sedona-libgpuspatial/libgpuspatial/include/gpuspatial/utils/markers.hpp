// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
#pragma once

#include <cstdint>
#ifndef GPUSPATIAL_PROFILING
#define DISABLE_NVTX_MARKERS
#endif

#ifndef DISABLE_NVTX_MARKERS
#include <nvtx3/nvtx3.hpp>
#endif
// This file provide a simple wrapper around NVTX3 for marking GPU code regions and events
// for profiling purposes.
namespace gpuspatial {

struct Category {
  static constexpr uint32_t KernelWorkitems = 1;
  static constexpr uint32_t IntervalWorkitems = 2;
};

// Colors in ARGB format (Alpha, Red, Green, Blue)
struct Color {
  static constexpr uint32_t Red = 0xFF880000;
  static constexpr uint32_t Green = 0xFF008800;
  static constexpr uint32_t Blue = 0xFF000088;
  static constexpr uint32_t Yellow = 0xFFFFFF00;
  static constexpr uint32_t Default = 0;
};

#ifndef DISABLE_NVTX_MARKERS

struct Instrument {
  // ---------------------------------------------------------------------------
  // Helper: Create attributes correctly using constructors
  // ---------------------------------------------------------------------------
  static nvtx3::event_attributes create_attr(const char* msg, uint32_t color_val,
                                             uint32_t category_val) {
    // 1. Basic Message
    nvtx3::event_attributes attr{msg};

    // 2. Apply Color (if not default)
    if (color_val != Color::Default) {
      // Use nvtx3::rgb wrapping the uint32_t directly usually works,
      // but if it fails, we assign to the internal color_type directly via the generic
      // color wrapper
      attr = nvtx3::event_attributes{msg, nvtx3::color{color_val}};
    }

    // 3. Apply Category (if valid)
    // Note: We cannot "append" to an existing immutable object.
    // We must construct with all arguments at once.

    if (color_val != Color::Default && category_val != 0) {
      return nvtx3::event_attributes{msg, nvtx3::color{color_val},
                                     nvtx3::category{category_val}};
    } else if (color_val != Color::Default) {
      return nvtx3::event_attributes{msg, nvtx3::color{color_val}};
    } else if (category_val != 0) {
      return nvtx3::event_attributes{msg, nvtx3::category{category_val}};
    }

    return attr;
  }

  // ---------------------------------------------------------------------------
  // Instant Markers
  // ---------------------------------------------------------------------------
  static void Mark(const char* message, uint32_t color = Color::Default,
                   uint32_t category = 0) {
    nvtx3::mark(create_attr(message, color, category));
  }

  static void MarkInt(int64_t value, const char* message, uint32_t color = Color::Default,
                      uint32_t category = 0) {
    // Construct with payload immediately
    // Note: If you need color+category+payload, the constructor list gets long.
    // This covers the most common case: Message + Payload
    if (color == Color::Default && category == 0) {
      nvtx3::event_attributes attr{message, nvtx3::payload{value}};
      nvtx3::mark(attr);
    } else {
      // Fallback: manually construct complex attribute
      // Most NVTX3 versions support {msg, color, payload, category} in any order
      nvtx3::event_attributes attr{message, nvtx3::color{color},
                                   nvtx3::category{category}, nvtx3::payload{value}};
      nvtx3::mark(attr);
    }
  }

  static void MarkWorkitems(uint64_t items, const char* message = "Workitems") {
    nvtx3::event_attributes attr{message, nvtx3::payload{items},
                                 nvtx3::category{Category::KernelWorkitems}};
    nvtx3::mark(attr);
  }

  // ---------------------------------------------------------------------------
  // Scoped Ranges (RAII)
  // ---------------------------------------------------------------------------
  struct Range {
    nvtx3::scoped_range range;

    // Standard Range
    explicit Range(const char* message, uint32_t color = Color::Default,
                   uint32_t category = 0)
        : range(Instrument::create_attr(message, color, category)) {}

    // Payload Range (for workitems/intervals)
    explicit Range(const char* message, uint64_t payload,
                   uint32_t category = Category::IntervalWorkitems)
        : range(nvtx3::event_attributes{message, nvtx3::payload{payload},
                                        nvtx3::category{category}}) {}
  };
};

#else

// -----------------------------------------------------------------------------
// No-Op Implementation
// -----------------------------------------------------------------------------
struct Instrument {
  static inline void Mark(const char*, uint32_t = 0, uint32_t = 0) {}
  static inline void MarkInt(int64_t, const char*, uint32_t = 0, uint32_t = 0) {}
  static inline void MarkWorkitems(uint64_t, const char*) {}

  struct Range {
    explicit Range(const char*, uint32_t = 0, uint32_t = 0) {}
    explicit Range(const char*, uint64_t, uint32_t = 0) {}
  };
};

#endif  // DISABLE_NVTX_MARKERS

}  // namespace gpuspatial
