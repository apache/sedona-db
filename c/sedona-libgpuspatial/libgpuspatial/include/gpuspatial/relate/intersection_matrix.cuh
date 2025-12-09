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
// octal numbers,
namespace gpuspatial {

// Dimensions of the Intersection
enum class IntersectionDimension : int32_t {
  EmptySet = 0U,  // F (000 in binary)
  Dim0 = 1U,      // 0D (001 in binary) - Point
  Dim1 = 3U,      // 1D (011 in binary) - Line
  Dim2 = 7U       // 2D (111 in binary) - Area
};

// Components of the Intersection Matrix (A_component x B_component)
// Used to define the position/offset in the 9-cell matrix (3x3 grid)
enum class ComponentPair : int32_t {
  Interior_Interior = 0,
  Interior_Boundary = 1,
  Interior_Exterior = 2,
  Boundary_Interior = 3,
  Boundary_Boundary = 4,
  Boundary_Exterior = 5,
  Exterior_Interior = 6,
  Exterior_Boundary = 7,
  Exterior_Exterior = 8
};
namespace detail {
DEV_HOST_INLINE static constexpr int32_t GetValue(ComponentPair pair,
                                                  IntersectionDimension dim) {
  return static_cast<int32_t>(dim) << (static_cast<int32_t>(pair) * 3);
}
}  // namespace detail
/** Intersection Matrix (IM) defined by octal numbers
 * Dimension	Octal	Binary	Meaning
 *      F	0	0	Empty set (no intersection)
 *      0D	1	001	Point-dimensional intersection
 *      1D	3	011	Line-dimensional intersection
 *      2D	7	111	Area-dimensional intersection
 */
class IntersectionMatrix {
  static constexpr ComponentPair AllPairs[] = {
      ComponentPair::Interior_Interior, ComponentPair::Interior_Boundary,
      ComponentPair::Interior_Exterior, ComponentPair::Boundary_Interior,
      ComponentPair::Boundary_Boundary, ComponentPair::Boundary_Exterior,
      ComponentPair::Exterior_Interior, ComponentPair::Exterior_Boundary,
      ComponentPair::Exterior_Exterior};

 public:
  // Interior x Interior (Shift: 0)
  static constexpr int32_t INTER_INTER_0D = detail::GetValue(
      ComponentPair::Interior_Interior, IntersectionDimension::Dim0);  // 0000000001U
  static constexpr int32_t INTER_INTER_1D = detail::GetValue(
      ComponentPair::Interior_Interior, IntersectionDimension::Dim1);  // 0000000003U
  static constexpr int32_t INTER_INTER_2D = detail::GetValue(
      ComponentPair::Interior_Interior, IntersectionDimension::Dim2);  // 0000000007U

  // Interior x Boundary (Shift: 3)
  static constexpr int32_t INTER_BOUND_0D = detail::GetValue(
      ComponentPair::Interior_Boundary, IntersectionDimension::Dim0);  // 0000000010U
  static constexpr int32_t INTER_BOUND_1D = detail::GetValue(
      ComponentPair::Interior_Boundary, IntersectionDimension::Dim1);  // 0000000030U
  static constexpr int32_t INTER_BOUND_2D = detail::GetValue(
      ComponentPair::Interior_Boundary, IntersectionDimension::Dim2);  // 0000000070U

  // Interior x Exterior (Shift: 6)
  static constexpr int32_t INTER_EXTER_0D = detail::GetValue(
      ComponentPair::Interior_Exterior, IntersectionDimension::Dim0);  // 0000000100U
  static constexpr int32_t INTER_EXTER_1D = detail::GetValue(
      ComponentPair::Interior_Exterior, IntersectionDimension::Dim1);  // 0000000300U
  static constexpr int32_t INTER_EXTER_2D = detail::GetValue(
      ComponentPair::Interior_Exterior, IntersectionDimension::Dim2);  // 0000000700U

  // Boundary x Interior (Shift: 9)
  static constexpr int32_t BOUND_INTER_0D = detail::GetValue(
      ComponentPair::Boundary_Interior, IntersectionDimension::Dim0);  // 0000001000U
  static constexpr int32_t BOUND_INTER_1D = detail::GetValue(
      ComponentPair::Boundary_Interior, IntersectionDimension::Dim1);  // 0000003000U
  static constexpr int32_t BOUND_INTER_2D = detail::GetValue(
      ComponentPair::Boundary_Interior, IntersectionDimension::Dim2);  // 0000007000U

  // Boundary x Boundary (Shift: 12)
  static constexpr int32_t BOUND_BOUND_0D = detail::GetValue(
      ComponentPair::Boundary_Boundary, IntersectionDimension::Dim0);  // 0000010000U
  static constexpr int32_t BOUND_BOUND_1D = detail::GetValue(
      ComponentPair::Boundary_Boundary, IntersectionDimension::Dim1);  // 0000030000U
  static constexpr int32_t BOUND_BOUND_2D = detail::GetValue(
      ComponentPair::Boundary_Boundary, IntersectionDimension::Dim2);  // 0000070000U

  // Boundary x Exterior (Shift: 15)
  static constexpr int32_t BOUND_EXTER_0D = detail::GetValue(
      ComponentPair::Boundary_Exterior, IntersectionDimension::Dim0);  // 0000100000U
  static constexpr int32_t BOUND_EXTER_1D = detail::GetValue(
      ComponentPair::Boundary_Exterior, IntersectionDimension::Dim1);  // 0000300000U
  static constexpr int32_t BOUND_EXTER_2D = detail::GetValue(
      ComponentPair::Boundary_Exterior, IntersectionDimension::Dim2);  // 0000700000U

  // Exterior x Interior (Shift: 18)
  static constexpr int32_t EXTER_INTER_0D = detail::GetValue(
      ComponentPair::Exterior_Interior, IntersectionDimension::Dim0);  // 0001000000U
  static constexpr int32_t EXTER_INTER_1D = detail::GetValue(
      ComponentPair::Exterior_Interior, IntersectionDimension::Dim1);  // 0003000000U
  static constexpr int32_t EXTER_INTER_2D = detail::GetValue(
      ComponentPair::Exterior_Interior, IntersectionDimension::Dim2);  // 0007000000U

  // Exterior x Boundary (Shift: 21)
  static constexpr int32_t EXTER_BOUND_0D = detail::GetValue(
      ComponentPair::Exterior_Boundary, IntersectionDimension::Dim0);  // 0010000000U
  static constexpr int32_t EXTER_BOUND_1D = detail::GetValue(
      ComponentPair::Exterior_Boundary, IntersectionDimension::Dim1);  // 0030000000U
  static constexpr int32_t EXTER_BOUND_2D = detail::GetValue(
      ComponentPair::Exterior_Boundary, IntersectionDimension::Dim2);  // 0070000000U

  // Exterior x Exterior (Shift: 24)
  static constexpr int32_t EXTER_EXTER_0D = detail::GetValue(
      ComponentPair::Exterior_Exterior, IntersectionDimension::Dim0);  // 0100000000U
  static constexpr int32_t EXTER_EXTER_1D = detail::GetValue(
      ComponentPair::Exterior_Exterior, IntersectionDimension::Dim1);  // 0300000000U
  static constexpr int32_t EXTER_EXTER_2D = detail::GetValue(
      ComponentPair::Exterior_Exterior, IntersectionDimension::Dim2);  // 0700000000U

  // Full Mask (All 9 octal digits set to 7)
  static constexpr int32_t MASK_FULL = 0777777777U;

  static DEV_HOST_INLINE int32_t Transpose(int32_t status) {
    if (status < 0) return status;  // Handle error status

    return (
        // (0,0) <-> (0,0): No shift (Index 0)
        ((status & INTER_INTER_2D)) |

        // (0,1) <-> (1,0): Shift up by 6 (Index 1 to Index 3)
        ((status & INTER_BOUND_2D) << 6) |

        // (0,2) <-> (2,0): Shift up by 12 (Index 2 to Index 6)
        ((status & INTER_EXTER_2D) << 12) |

        // (1,0) <-> (0,1): Shift down by 6 (Index 3 to Index 1)
        ((status & BOUND_INTER_2D) >> 6) |

        // (1,1) <-> (1,1): No shift (Index 4)
        ((status & BOUND_BOUND_2D)) |

        // (1,2) <-> (2,1): Shift up by 6 (Index 5 to Index 7)
        ((status & BOUND_EXTER_2D) << 6) |

        // (2,0) <-> (0,2): Shift down by 12 (Index 6 to Index 2)
        ((status & EXTER_INTER_2D) >> 12) |

        // (2,1) <-> (1,2): Shift down by 6 (Index 7 to Index 5)
        ((status & EXTER_BOUND_2D) >> 6) |

        // (2,2) <-> (2,2): No shift (Index 8)
        ((status & EXTER_EXTER_2D)));
  }

  static void ToString(uint32_t status, char* res) {
    // Array of 2D masks for each component (used to extract the 3-bit value)
    // These are guaranteed to be compile-time constants.
    constexpr int32_t Dim2Masks[] = {INTER_INTER_2D, INTER_BOUND_2D, INTER_EXTER_2D,
                                     BOUND_INTER_2D, BOUND_BOUND_2D, BOUND_EXTER_2D,
                                     EXTER_INTER_2D, EXTER_BOUND_2D, EXTER_EXTER_2D};

    for (int i = 0; i < 9; ++i) {
      // 1. Extract the 3-bit value for the current component (cell)
      int32_t component_value = status & Dim2Masks[i];

      // 2. Get the 0D, 1D, 2D codes for the current cell 'i' to compare against
      ComponentPair pair = AllPairs[i];
      int32_t code0D = detail::GetValue(pair, IntersectionDimension::Dim0);
      int32_t code1D = detail::GetValue(pair, IntersectionDimension::Dim1);
      int32_t code2D = detail::GetValue(pair, IntersectionDimension::Dim2);

      // 3. Determine the dimension character ('0', '1', '2', or 'F')
      if (component_value == code0D) {
        res[i] = '0';
      } else if (component_value == code1D) {
        res[i] = '1';
      } else if (component_value == code2D) {
        res[i] = '2';
      } else {
        res[i] = 'F';
      }
    }

    // Ensure the string is null-terminated
    res[9] = '\0';
  }
};
}  // namespace gpuspatial
