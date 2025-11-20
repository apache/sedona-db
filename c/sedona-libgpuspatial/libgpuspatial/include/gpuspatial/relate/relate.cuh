/*
* PG-Strom Extension for GPU Acceleration on PostgreSQL Database
 *
 * Copyright (c) 2012-2024, KaiGai Kohei <kaigai@kaigai.gr.jp>
 * Copyright (c) 2017-2024, HeteroDB,Inc <contact@heterodb.com>
 *
 * Permission to use, copy, modify, and distribute this software and its
 * documentation for any purpose, without fee, and without a written agreement
 * is hereby granted, provided that the above copyright notice and this
 * paragraph and the following two paragraphs appear in all copies.
 *
 * IN NO EVENT SHALL HETERODB,INC BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 * SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION,
 * EVEN IF HETERODB,INC HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * HETERODB,INC SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 * THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND HETERODB,INC HAS
 * NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 */

#pragma once
#include "gpuspatial/geom/line_string.cuh"
#include "gpuspatial/geom/multi_line_string.cuh"
#include "gpuspatial/geom/multi_point.cuh"
#include "gpuspatial/geom/multi_polygon.cuh"
#include "gpuspatial/geom/point.cuh"
#include "gpuspatial/geom/polygon.cuh"
#include "gpuspatial/relate/im.cuh"
// Ref: https://github.com/heterodb/pg-strom/blob/master/src/xpu_postgis.cu
// A good visualize to cases
// https://dev.luciad.com/portal/productDocumentation/LuciadFusion/docs/articles/guide/geometry/images/interior_exterior_boundary.png

// For line-polygon test
#define IM__LINE_HEAD_CONTAINED 01000000000U
#define IM__LINE_TAIL_CONTAINED 02000000000U
#define RELATE_MAX_DEPTH (5)
#ifndef BITS_PER_BYTE
#define BITS_PER_BYTE 8
#endif
#ifndef SHRT_NBITS
#define SHRT_NBITS (sizeof(int16_t) * BITS_PER_BYTE)
#endif
#ifndef INT_NBITS
#define INT_NBITS (sizeof(int32_t) * BITS_PER_BYTE)
#endif
#ifndef LONG_NBITS
#define LONG_NBITS (sizeof(int64_t) * BITS_PER_BYTE)
#endif
namespace gpuspatial {

/**
 * Transpose an IM
 */
DEV_HOST_INLINE
int32_t IM__TWIST(int32_t status) {
  if (status < 0) return status; /* error */
  return (((status & IM__INTER_INTER_2D)) | ((status & IM__INTER_BOUND_2D) << 6) |
          ((status & IM__INTER_EXTER_2D) << 12) | ((status & IM__BOUND_INTER_2D) >> 6) |
          ((status & IM__BOUND_BOUND_2D)) | ((status & IM__BOUND_EXTER_2D) << 6) |
          ((status & IM__EXTER_INTER_2D) >> 12) | ((status & IM__EXTER_BOUND_2D) >> 6) |
          ((status & IM__EXTER_EXTER_2D)));
}

DEV_HOST_INLINE void IM__ToString(int32_t status, char* res) {
  if ((status & IM__INTER_INTER_2D) == IM__INTER_INTER_0D) {
    res[0] = '0';
  } else if ((status & IM__INTER_INTER_2D) == IM__INTER_INTER_1D) {
    res[0] = '1';
  } else if ((status & IM__INTER_INTER_2D) == IM__INTER_INTER_2D) {
    res[0] = '2';
  } else {
    res[0] = 'F';
  }

  if ((status & IM__INTER_BOUND_2D) == IM__INTER_BOUND_0D) {
    res[1] = '0';
  } else if ((status & IM__INTER_BOUND_2D) == IM__INTER_BOUND_1D) {
    res[1] = '1';
  } else if ((status & IM__INTER_BOUND_2D) == IM__INTER_BOUND_2D) {
    res[1] = '2';
  } else {
    res[1] = 'F';
  }

  if ((status & IM__INTER_EXTER_2D) == IM__INTER_EXTER_0D) {
    res[2] = '0';
  } else if ((status & IM__INTER_EXTER_2D) == IM__INTER_EXTER_1D) {
    res[2] = '1';
  } else if ((status & IM__INTER_EXTER_2D) == IM__INTER_EXTER_2D) {
    res[2] = '2';
  } else {
    res[2] = 'F';
  }

  if ((status & IM__BOUND_INTER_2D) == IM__BOUND_INTER_0D) {
    res[3] = '0';
  } else if ((status & IM__BOUND_INTER_2D) == IM__BOUND_INTER_1D) {
    res[3] = '1';
  } else if ((status & IM__BOUND_INTER_2D) == IM__BOUND_INTER_2D) {
    res[3] = '2';
  } else {
    res[3] = 'F';
  }

  if ((status & IM__BOUND_BOUND_2D) == IM__BOUND_BOUND_0D) {
    res[4] = '0';
  } else if ((status & IM__BOUND_BOUND_2D) == IM__BOUND_BOUND_1D) {
    res[4] = '1';
  } else if ((status & IM__BOUND_BOUND_2D) == IM__BOUND_BOUND_2D) {
    res[4] = '2';
  } else {
    res[4] = 'F';
  }

  if ((status & IM__BOUND_EXTER_2D) == IM__BOUND_EXTER_0D) {
    res[5] = '0';
  } else if ((status & IM__BOUND_EXTER_2D) == IM__BOUND_EXTER_1D) {
    res[5] = '1';
  } else if ((status & IM__BOUND_EXTER_2D) == IM__BOUND_EXTER_2D) {
    res[5] = '2';
  } else {
    res[5] = 'F';
  }

  if ((status & IM__EXTER_INTER_2D) == IM__EXTER_INTER_0D) {
    res[6] = '0';
  } else if ((status & IM__EXTER_INTER_2D) == IM__EXTER_INTER_1D) {
    res[6] = '1';
  } else if ((status & IM__EXTER_INTER_2D) == IM__EXTER_INTER_2D) {
    res[6] = '2';
  } else {
    res[6] = 'F';
  }

  if ((status & IM__EXTER_BOUND_2D) == IM__EXTER_BOUND_0D) {
    res[7] = '0';
  } else if ((status & IM__EXTER_BOUND_2D) == IM__EXTER_BOUND_1D) {
    res[7] = '1';
  } else if ((status & IM__EXTER_BOUND_2D) == IM__EXTER_BOUND_2D) {
    res[7] = '2';
  } else {
    res[7] = 'F';
  }

  if ((status & IM__EXTER_EXTER_2D) == IM__EXTER_EXTER_0D) {
    res[8] = '0';
  } else if ((status & IM__EXTER_EXTER_2D) == IM__EXTER_EXTER_1D) {
    res[8] = '1';
  } else if ((status & IM__EXTER_EXTER_2D) == IM__EXTER_EXTER_2D) {
    res[8] = '2';
  } else {
    res[8] = 'F';
  }
  res[9] = '\0';
}

/**
 * Relate LineSegment P1-P2 with MultiPolygon.
 * @tparam POINT_T
 * @tparam INDEX_T
 * @param P1
 * @param p1_is_head
 * @param P2
 * @param p2_is_tail
 * @param geom
 * @param nskips
 * @param last_polygons
 * @return
 */
template <typename POINT_T, typename INDEX_T>
DEV_HOST int32_t relate(const POINT_T& P1, bool p1_is_head, const POINT_T& P2,
                        bool p2_is_tail, const MultiPolygon<POINT_T, INDEX_T>& geom,
                        int32_t nskips, bool last_polygons, int stack_depth = 0) {
  int32_t nloops;
  int32_t retval = 0;
  int32_t status;
  int32_t nrings = 0;
  uint32_t __nrings_next;

  if (stack_depth >= RELATE_MAX_DEPTH) {
    return 0;
  }

  LineSegment<POINT_T> seg_p(P1, P2);
  /* centroid of P1-P2 */
  auto Pc = seg_p.centroid();

  nloops = geom.num_polygons();
  for (int k = 0; k < nloops; k++, nrings = __nrings_next) {
    char p1_location = '?';
    char p2_location = '?';
    char pc_location = '?';

    const auto& poly = geom.get_polygon(k);

    /* rewind to the point where recursive call is invoked */
    __nrings_next = nrings + poly.num_rings();
    if (__nrings_next < nskips) continue;
    if (poly.empty()) continue;

    /* check for each ring/hole */
    for (int i = 0; i < poly.num_rings(); i++, nrings++) {
      int32_t wn1 = 0;
      int32_t wn2 = 0;
      int32_t wnc = 0;
      int32_t pq1, pq2;

      p1_location = p2_location = pc_location = '?';

      const auto& ring = poly.get_ring(i);

      // TODO: Define error codes
      if (ring.empty()) {
        printf("Empty ring\n");
        return -1;
      } else if (!ring.is_valid()) {
        printf("Invalid ring\n");
        return -1;
      }

      if (nrings < nskips) continue;

      /* ring/hole must be closed. */
      auto Q1 = ring.get_point(0);
      POINT_T Q2;

      pq1 = seg_p.orientation(Q1);
      for (int j = 1; j < ring.num_points(); j++) {
        int32_t qp1, qp2, qpc;

        Q2 = ring.get_point(j);
        if (Q1 == Q2) continue; /* ignore zero length edge */
        LineSegment<POINT_T> seg_q(Q1, Q2);

        pq2 = seg_p.orientation(Q2);

        /*
         * Update the state of winding number algorithm to determine
         * the location of P1/P2 whether they are inside or outside
         * of the Q1-Q2 edge.
         */
        qp1 = seg_q.orientation(P1);

        if (qp1 < 0 && Q1.y() <= P1.y() && P1.y() < Q2.y())
          wn1++;
        else if (qp1 > 0 && Q2.y() <= P1.y() && P1.y() < Q1.y())
          wn1--;

        qp2 = seg_q.orientation(P2);
        if (qp2 < 0 && Q1.y() <= P2.y() && P2.y() < Q2.y())
          wn2++;
        else if (qp2 > 0 && Q2.y() <= P2.y() && P2.y() < Q1.y())
          wn2--;

        qpc = seg_q.orientation(Pc);
        if (qpc < 0 && Q1.y() <= Pc.y() && Pc.y() < Q2.y())
          wnc++;
        else if (qpc > 0 && Q2.y() <= Pc.y() && Pc.y() < Q1.y())
          wnc--;
        if (seg_q.locate_point(Pc) == PointLocation::kBoundary) pc_location = 'B';
#if 0
				printf("P1(%d,%d)-P2(%d,%d) Q1(%d,%d)-Q2(%d,%d) qp1=%d qp2=%d pq1=%d pq2=%d\n",
					   (int)P1.x, (int)P1.y, (int)P2.x, (int)P2.y,
					   (int)Q1.x, (int)Q1.y, (int)Q2.x, (int)Q2.y,
					   qp1, qp2, pq1, pq2);
#endif
        if (!qp1 && !qp2) {
          /* P1-P2 and Q1-Q2 are colinear */
          auto p1_in_qq = seg_q.locate_point(P1);
          auto p2_in_qq = seg_q.locate_point(P2);

          if (p1_in_qq != PointLocation::kOutside &&
              p2_in_qq != PointLocation::kOutside) {
            /* P1-P2 is fully contained by Q1-Q2 */
            if (p1_is_head) retval |= (IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED);
            if (p2_is_tail) retval |= (IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED);
            if (P1 == P2) {
              if (!p1_is_head && !p2_is_tail) retval |= IM__INTER_BOUND_0D;
            } else
              retval |= IM__INTER_BOUND_1D;
            return retval;
          }

          auto q1_in_pp = seg_p.locate_point(Q1);
          auto q2_in_pp = seg_p.locate_point(Q2);
          LineSegment<POINT_T> seg_p1q2(P1, Q2);
          LineSegment<POINT_T> seg_q1p2(Q1, P2);
          LineSegment<POINT_T> seg_p1q1(P1, Q1);
          LineSegment<POINT_T> seg_q2p1(Q2, P1);

          if (p1_in_qq != PointLocation::kOutside &&
              p2_in_qq == PointLocation::kOutside) {
            /* P1 is contained by Q1-Q2, but P2 is not */
            if (p1_is_head)
              retval |= (IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED);
            else
              retval |= IM__INTER_BOUND_0D;

            if (q1_in_pp == PointLocation::kInside) {
              /* case of Q2-P1-Q1-P2; Q1-P2 is out of bounds */
              assert(q2_in_pp != PointLocation::kInside);
              status = relate(Q1, false, P2, p2_is_tail, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              return (retval | status | IM__INTER_BOUND_1D);
            } else if (q2_in_pp == PointLocation::kInside) {
              /* case of Q1-P1-Q2-P2; Q2-P2 is out of bounds */
              assert(q1_in_pp != PointLocation::kInside);
              status = relate(Q2, false, P2, p2_is_tail, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              return (retval | status | IM__INTER_BOUND_1D);
            } else {
              assert(q1_in_pp == PointLocation::kBoundary ||
                     q2_in_pp == PointLocation::kBoundary);
            }
          } else if (p1_in_qq == PointLocation::kOutside &&
                     p2_in_qq != PointLocation::kOutside) {
            /* P2 is contained by Q1-Q2, but P2 is not */
            if (p2_is_tail)
              retval |= (IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED);
            else
              retval |= IM__INTER_BOUND_0D;

            if (q1_in_pp == PointLocation::kInside) {
              /* P1-Q1-P2-Q2; P1-Q1 is out of bounds */
              status = relate(P1, p1_is_head, Q1, false, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              return (retval | status | IM__INTER_BOUND_1D);
            } else if (q2_in_pp == PointLocation::kInside) {
              /* P1-Q2-P2-Q1; P1-Q2 is out of bounds */
              status = relate(P1, p1_is_head, Q2, false, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              return (retval | status | IM__INTER_BOUND_1D);
            }
          } else if (seg_p1q2.locate_point(Q1) != PointLocation::kOutside &&
                     seg_q1p2.locate_point(Q2) != PointLocation::kOutside) {
            /* case of P1-Q1-Q2-P2 */
            if (P1 != Q1) {
              status = relate(P1, p1_is_head, Q1, false, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              retval |= status;
            }
            if (Q2 != P2) {
              status = relate(Q2, false, P2, p2_is_tail, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              retval |= status;
            }
            return (retval | IM__INTER_BOUND_1D);
          } else if (seg_p1q1.locate_point(Q2) != PointLocation::kOutside &&
                     seg_q2p1.locate_point(Q1) != PointLocation::kOutside) {
            /* case of P1-Q2-Q1-P2 */
            if (P1 != Q2) {
              status = relate(P1, p1_is_head, Q2, false, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              retval |= status;
            }
            if (Q1 != P2) {
              status = relate(Q1, false, P2, p2_is_tail, geom, nrings, last_polygons,
                              stack_depth + 1);
              if (status < 0) return -1;
              retval |= status;
            }
            return (retval | IM__INTER_BOUND_1D);
          }
        } else if (qp1 == 0 && ((pq1 >= 0 && pq2 <= 0) || (pq1 <= 0 && pq2 >= 0))) {
          /* P1 touched Q1-Q2 */
          if (p1_is_head)
            retval |= (IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED);
          else
            retval |= IM__INTER_BOUND_0D;
          p1_location = 'B';
        } else if (qp2 == 0 && ((pq1 >= 0 && pq2 <= 0) || (pq1 <= 0 && pq2 >= 0))) {
          /* P2 touched Q1-Q2 */
          if (p2_is_tail)
            retval |= (IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED);
          else
            retval |= IM__INTER_BOUND_0D;
          p2_location = 'B';
        } else if (((qp1 >= 0 && qp2 <= 0) || (qp1 <= 0 && qp2 >= 0)) &&
                   ((pq1 >= 0 && pq2 <= 0) || (pq1 <= 0 && pq2 >= 0))) {
          /*
           * P1-P2 and Q1-Q2 crosses.
           *
           * The point where crosses is:
           *   P1 + r * (P2-P1) = Q1 + s * (Q2 - Q1)
           *   [0 < s,r < 1]
           *
           * frac = (P2.x-P1.x)(Q2.y-Q1.y)-(P2.y-P1.y)(Q2.x-Q1.x)
           * r = ((Q2.y - Q1.y) * (Q1.x-P1.x) -
           *      (Q1.x - Q1.x) * (Q1.y-P1.y)) / frac
           * s = ((P2.y - P1.y) * (Q1.x-P1.x) -
           *      (P2.x - P1.x) * (Q1.y-P1.y)) / frac
           *
           * C = P1 + r * (P2-P1)
           */
          using scala_t = typename POINT_T::scalar_t;
          scala_t r, frac;
          POINT_T C;

          frac = (P2.x() - P1.x()) * (Q2.y() - Q1.y()) -
                 (P2.y() - P1.y()) * (Q2.x() - Q1.x());
          assert(frac != 0.0);
          r = ((Q2.y() - Q1.y()) * (Q1.x() - P1.x()) -
               (Q2.x() - Q1.x()) * (Q1.y() - P1.y())) /
              frac;
          C.x() = P1.x() + r * (P2.x() - P1.x());
          C.y() = P1.y() + r * (P2.y() - P1.y());
#if 0
          printf(
              "P1(%.10lf,%.10lf)-P2(%.10lf,%.10lf) x Q1(%.10lf,%.10lf)-Q2(%.10lf,%lf) crosses at C(%.10lf,%.10lf) %d %d\n",
              P1.x(), P1.y(), P2.x(), P2.y(), Q1.x(), Q1.y(), Q2.x(), Q2.y(), C.x(),
              C.y(), (int)(!float_equal(P1.x(), C.x()) || !float_equal(P1.y(), C.y())),
              (int)(!float_equal(P2.x(), C.x()) || !float_equal(P2.y(), C.y())));
#endif
          if (P1 == C) {
            if (p1_is_head)
              retval |= (IM__BOUND_BOUND_0D | IM__LINE_HEAD_CONTAINED);
            else
              retval |= IM__INTER_BOUND_0D;
            p1_location = 'B';
          } else if (P2 == C) {
            if (p2_is_tail)
              retval |= (IM__BOUND_BOUND_0D | IM__LINE_TAIL_CONTAINED);
            else
              retval |= IM__INTER_BOUND_0D;
            p2_location = 'B';
          } else {
            /* try P1-C recursively */
            status = relate(P1, p1_is_head, C, false, geom, nrings, last_polygons,
                            stack_depth + 1);
            if (status < 0) return -1;
            retval |= status;
            /* try C-P2 recursively */
            status = relate(C, false, P2, p2_is_tail, geom, nrings, last_polygons,
                            stack_depth + 1);
            if (status < 0) return -1;
            retval |= status;
            return (retval | IM__INTER_BOUND_0D);
          }
        }
        /* move to the next edge */
        pq1 = pq2;
        Q1 = Q2;
      }
      /* location of P1,P2 and Pc */
      if (p1_location == '?') p1_location = (wn1 == 0 ? 'E' : 'I');
      if (p2_location == '?') p2_location = (wn2 == 0 ? 'E' : 'I');
      if (pc_location == '?') pc_location = (wnc == 0 ? 'E' : 'I');
#if 0
			printf("Poly(%d)/Ring(%d) P1(%d,%d)[%c]-P2(%d,%d)[%c] (Pc(%d,%d)[%c])\n",
				   k, i,
				   (int)P1.x, (int)P1.y, p1_location,
				   (int)P2.x, (int)P2.y, p2_location,
				   (int)Pc.x, (int)Pc.y, pc_location);
#endif
      if (i == 0) {
        /* case of ring-0 */
        if ((p1_location == 'I' && p2_location == 'I') ||
            (p1_location == 'I' && p2_location == 'B') ||
            (p1_location == 'B' && p2_location == 'I')) {
          /*
           * P1-P2 goes through inside of the polygon,
           * so don't need to check other polygons any more.
           */
          last_polygons = true;
        } else if (p1_location == 'B' && p2_location == 'B') {
          if (pc_location == 'B') return retval; /* P1-P2 exactly goes on boundary */
          if (pc_location == 'I') last_polygons = true;
          if (pc_location == 'E') break;
        } else if ((p1_location == 'B' && p2_location == 'E') ||
                   (p1_location == 'E' && p2_location == 'B') ||
                   (p1_location == 'E' && p2_location == 'E')) {
          /*
           * P1-P2 goes outside of the polygon, so don't need
           * to check holes of this polygon.
           */
          break;
        } else {
          /*
           * If P1-P2 would be I-E or E-I, it obviously goes
           * across the boundary line; should not happen.
           */
#if 1
          printf("P1 [%c] (%.2f,%.2f) P2 [%c] (%.2f,%.2f)\n", p1_location, P1.x(), P1.y(),
                 p2_location, P2.x(), P2.y());
#endif
          printf("unexpected segment-polygon relation\n");
          return -1;
        }
      } else {
        if ((p1_location == 'I' && p2_location == 'I') ||
            (p1_location == 'I' && p2_location == 'B') ||
            (p1_location == 'B' && p2_location == 'I') ||
            (p1_location == 'B' && p2_location == 'B' && pc_location == 'I')) {
          /*
           * P1-P2 goes throught inside of the hole.
           */
          return (retval | IM__INTER_EXTER_1D);
        }
      }
    }

    /*
     * 'last_polygons == true' means P1-P2 goes inside of the polygon
     * and didn't touch any holes.
     */
    if (last_polygons) {
      if (p1_is_head && p1_location != 'B')
        retval |= (IM__BOUND_INTER_0D | IM__LINE_HEAD_CONTAINED);
      if (p2_is_tail && p2_location != 'B')
        retval |= (IM__BOUND_INTER_0D | IM__LINE_TAIL_CONTAINED);
      return (retval | IM__INTER_INTER_1D);
    }
  }
  /*
   * Once the control reached here, it means P1-P2 never goes inside
   * of the polygons.
   */
  return (retval | IM__INTER_EXTER_1D);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const LinearRing<POINT_T>& ring,
                               const MultiPolygon<POINT_T, INDEX_T>& geom) {
  bool poly_has_inside = false;
  bool poly_has_outside = false;
  int32_t rflags = 0;
  int32_t boundary = 0;

  if (ring.empty()) return 0; /* empty */

  /* decrement nitems if tail items are duplicated */
  auto P1 = ring.get_point(ring.num_points() - 1);
  POINT_T P2;
  size_t nitems;

  for (nitems = ring.num_points(); nitems >= 2; nitems--) {
    P2 = ring.get_point(nitems - 2);
    if (P1 != P2) break;
  }
  /* checks for each edge */
  P1 = ring.get_point(0);
  const auto& mbr = geom.get_mbr();

  for (int i = 2; i <= nitems; i++) {
    P2 = ring.get_point(i - 1);
    if (P1 == P2) {
      continue;
    }
    int32_t status;

    if (std::max(P1.x(), P2.x()) < mbr.get_min().x() ||
        std::min(P1.x(), P2.x()) > mbr.get_max().x() ||
        std::max(P1.y(), P2.y()) < mbr.get_min().y() ||
        std::min(P1.y(), P2.y()) > mbr.get_max().y()) {
      status = (IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_INTER_2D |
                IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
    } else {
      status = relate(P1, false, P2, false, geom, 0, false);
      // char res[10];
      // IM__ToString(status, res);
      // printf("P1 (%lf, %lf), P2 (%lf, %lf), IM %s\n", P1.x(), P1.y(), P2.x(), P2.y(),
      // res);
      if (status < 0) return -1;
    }
    rflags |= status;
    P1 = P2;
  }
  /*
   * Simple check whether polygon is fully contained by the ring
   */
  for (int k = 0; k < geom.num_polygons(); k++) {
    const auto& poly = geom.get_polygon(k);
    if (poly.empty()) continue;
    auto exterior_ring = poly.get_ring(0);

    for (int i = 0; i < exterior_ring.num_points(); i++) {
      const auto& P = exterior_ring.get_point(i);
      auto location = ring.locate_point(P);

      if (location == PointLocation::kInside)
        poly_has_inside = true;
      else if (location == PointLocation::kOutside)
        poly_has_outside = true;
      else if (location != PointLocation::kBoundary)
        return -1;
    }
    if (poly_has_inside && poly_has_outside) break;
  }

  /*
   * transform rflags to ring-polygon relationship
   */
  if ((rflags & IM__INTER_BOUND_2D) == IM__INTER_BOUND_1D)
    boundary = IM__BOUND_BOUND_1D;
  else if ((rflags & IM__INTER_BOUND_2D) == IM__INTER_BOUND_0D)
    boundary = IM__BOUND_BOUND_0D;

  if ((rflags & IM__INTER_INTER_2D) == 0 && (rflags & IM__INTER_BOUND_2D) != 0 &&
      (rflags & IM__INTER_EXTER_2D) == 0) {
    /* ring equals to the polygon */
    return (IM__INTER_INTER_2D | IM__BOUND_BOUND_1D | IM__EXTER_EXTER_2D);
  } else if ((rflags & IM__INTER_INTER_2D) == 0 && (rflags & IM__INTER_BOUND_2D) == 0 &&
             (rflags & IM__INTER_EXTER_2D) != 0) {
    if (poly_has_outside) {
      /* disjoint */
      return (IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_INTER_2D |
              IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
    } else {
      /* ring fully contains the polygons */
      return (IM__INTER_INTER_2D | IM__INTER_BOUND_1D | IM__INTER_EXTER_2D |
              IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D);
    }
  } else if ((rflags & IM__INTER_INTER_2D) != 0 && (rflags & IM__INTER_BOUND_2D) != 0
             // TODO: Need this? && (rflags & IM__INTER_EXTER_2D) != 0
  ) {
    /* ring has intersection to the polygon */
    assert(boundary != 0);
    if ((rflags & IM__INTER_EXTER_2D) != 0) {
      boundary |= IM__BOUND_EXTER_1D;
    }
    return boundary | (IM__INTER_INTER_2D | IM__INTER_BOUND_1D | IM__INTER_EXTER_2D |
                       IM__BOUND_INTER_1D | IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D |
                       IM__EXTER_EXTER_2D);
  } else if ((rflags & IM__INTER_INTER_2D) == 0 && (rflags & IM__INTER_BOUND_2D) != 0 &&
             (rflags & IM__INTER_EXTER_2D) != 0) {
    if (poly_has_outside) {
      /* ring touched the polygon at a boundary, but no intersection */
      assert(boundary != 0);
      return boundary | (IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_INTER_2D |
                         IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
    } else {
      /* ring fully contains the polygon touched at boundaries */
      assert(boundary != 0);
      return boundary | (IM__INTER_INTER_2D | IM__INTER_BOUND_1D | IM__INTER_EXTER_2D |
                         IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D);
    }
  } else if ((rflags & IM__INTER_INTER_2D) != 0 && (rflags & IM__INTER_EXTER_2D) == 0) {
    /* ring is fully contained by the polygon; might be touched */
    return boundary | (IM__INTER_INTER_2D | IM__BOUND_INTER_1D | IM__EXTER_INTER_2D |
                       IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
  }
  // FIXME:
  printf("unknown intersection\n");
  return -1; /* unknown intersection */
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(MultiPoint<POINT_T> geom1, MultiPoint<POINT_T> geom2);

template <typename SCALA_T, int N_DIM>
DEV_HOST_INLINE int32_t relate(const Point<SCALA_T, N_DIM>& geom1,
                               const Point<SCALA_T, N_DIM>& geom2) {
  using point_t = Point<SCALA_T, N_DIM>;
  MultiPoint<point_t> p1, p2;
  if (!geom1.empty()) {
    p1 = {ArrayView<point_t>(const_cast<point_t*>(&geom1), 1), geom1.get_mbr()};
  }

  if (!geom2.empty()) {
    p2 = {ArrayView<point_t>(const_cast<point_t*>(&geom2), 1), geom2.get_mbr()};
  }
  return relate(p1, p2);
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1, const MultiPoint<POINT_T>& geom2) {
  MultiPoint<POINT_T> p1;
  if (!geom1.empty()) {
    p1 = {ArrayView<POINT_T>(const_cast<POINT_T*>(&geom1), 1), geom1.get_mbr()};
  }
  return relate(p1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2);
template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1, const LineString<POINT_T>& geom2) {
  size_t prefix_sum_parts[2] = {0, geom2.num_points()};
  MultiLineString<POINT_T, size_t> m2(ArrayView<size_t>(prefix_sum_parts, 2),
                                      geom2.get_vertices(), geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  MultiPoint<POINT_T> m1;
  if (!geom1.empty()) {
    m1 = {ArrayView<POINT_T>(const_cast<POINT_T*>(&geom1), 1), geom1.get_mbr()};
  }
  return relate(m1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  MultiPoint<POINT_T> m1;
  if (!geom1.empty()) {
    m1 = {ArrayView<POINT_T>(const_cast<POINT_T*>(&geom1), 1), geom1.get_mbr()};
  }

  auto prefix_sum_rings = geom2.get_prefix_sum_rings();
  auto vertices = geom2.get_vertices();

  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    prefix_sum_rings, vertices, geom2.get_mbr());

  return relate(m1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2,
                               PointLocation location) {
  int32_t retval = IM__EXTER_EXTER_2D;

  bool matched = false;

  retval |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;

  /* dive into the polygon */
  switch (location) {
    case PointLocation::kInside: {
      matched = true;
      retval |= IM__INTER_INTER_0D;
      break;
    }
    case PointLocation::kBoundary: {
      matched = true;
      retval |= IM__INTER_BOUND_0D;
      break;
    }
    case PointLocation::kOutside: {
      break;
    }
    default:
      return -1; /* error */
  }
  if (!matched) retval |= IM__INTER_EXTER_0D;
  return retval;
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  MultiPoint<POINT_T> p1;
  if (!geom1.empty()) {
    p1 = {ArrayView<POINT_T>(const_cast<POINT_T*>(&geom1), 1), geom1.get_mbr()};
  }
  return relate(p1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const POINT_T& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2,
                               ArrayView<PointLocation> locations) {
  assert(geom2.num_polygons() == locations.size());
  if (geom2.empty()) return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;
  int32_t retval = IM__EXTER_EXTER_2D;
  bool matched = false;

  for (int j = 0; j < geom2.num_polygons(); j++) {
    retval |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;

    /* dive into the polygon */
    switch (locations[j]) {
      case PointLocation::kInside: {
        matched = true;
        retval |= IM__INTER_INTER_0D;
        break;
      }
      case PointLocation::kBoundary: {
        matched = true;
        retval |= IM__INTER_BOUND_0D;
        break;
      }
      case PointLocation::kOutside: {
        break;
      }
      default:
        return -1; /* error */
    }
  }
  if (!matched) retval |= IM__INTER_EXTER_0D;
  return retval;
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1, const POINT_T& geom2) {
  MultiPoint<POINT_T> p2;
  if (!geom2.empty()) {
    p2 = {ArrayView<POINT_T>(const_cast<POINT_T*>(&geom2), 1), geom2.get_mbr()};
  }
  return relate(geom1, p2);
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(MultiPoint<POINT_T> geom1, MultiPoint<POINT_T> geom2) {
  int32_t nloops1;
  int32_t nloops2;
  int32_t retval;
  bool twist_retval = false;
  if (geom1.empty() && geom2.empty()) return IM__EXTER_EXTER_2D;
  if (geom1.empty()) return IM__EXTER_INTER_0D | IM__EXTER_EXTER_2D;
  if (geom2.empty()) return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;
  /*
   * micro optimization: geom2 should have smaller number of items
   */
  if (geom2.num_points() > 1) {
    if (geom1.num_points() == 1 || geom1.num_points() < geom2.num_points()) {
      thrust::swap(geom1, geom2);
      twist_retval = true;
    }
  }
  retval = IM__EXTER_EXTER_2D;
  nloops1 = geom1.num_points();
  nloops2 = geom2.num_points();

  for (int base = 0; base < nloops2; base += LONG_NBITS) {
    uint64_t matched2 = 0;
    uint64_t __mask;

    for (int i = 0; i < nloops1; i++) {
      auto const& pt1 = geom1.get_point(i);
      bool matched1 = false;

      for (int j = 0; j < nloops2; j++) {
        auto const& pt2 = geom2.get_point(j);

        if (pt1 == pt2) {
          retval |= IM__INTER_INTER_0D;
          matched1 = true;
          if (j >= base && j < base + LONG_NBITS) matched2 |= (1UL << (j - base));
        }
      }
      if (!matched1) retval |= IM__INTER_EXTER_0D;
    }
    if (base + LONG_NBITS >= nloops2)
      __mask = (1UL << (nloops2 - base)) - 1;
    else
      __mask = ~0UL;

    if (__mask != matched2) {
      retval |= IM__EXTER_INTER_0D;
      break;
    }
  }
  return (twist_retval ? IM__TWIST(retval) : retval);
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1,
                               const LineString<POINT_T>& geom2) {
  size_t prefix_sum_parts[2] = {0, geom2.num_points()};

  MultiLineString<POINT_T, size_t> m2(ArrayView<size_t>(prefix_sum_parts, 2),
                                      geom2.get_vertices(), geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  int32_t retval;

  /* shortcut if either-geometry is empty */
  if (geom1.empty() && geom2.empty()) return IM__EXTER_EXTER_2D;
  if (geom1.empty()) return IM__EXTER_INTER_1D | IM__EXTER_BOUND_0D | IM__EXTER_EXTER_2D;
  if (geom2.empty()) return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

  auto nloops1 = geom1.num_points();
  auto nloops2 = geom2.num_line_strings();

  retval = IM__EXTER_EXTER_2D;
  for (size_t base = 0; base < nloops2; base += LONG_NBITS) {
    uint64_t head_matched = 0UL;
    uint64_t tail_matched = 0UL;
    uint64_t boundary_mask = 0UL;

    /* walks on for each points */
    for (int i = 0; i < nloops1; i++) {
      const auto& P = geom1.get_point(i);
      bool matched = false;

      /* walks on for each linestrings */
      for (size_t j = 0; j < nloops2; j++) {
        auto ls = geom2.get_line_string(j);
        if (ls.empty()) continue;
        const auto& Q2 = ls.get_point(ls.num_points() - 1);
        const auto& Q1 = ls.get_point(0);

        if (!ls.is_zero_length()) {
          retval |= IM__EXTER_INTER_1D;
        }

        /* walks on vertex of the line edges */
        auto has_boundary = Q1 != Q2;
        if (has_boundary && (j >= base && j < base + LONG_NBITS))
          boundary_mask |= (1UL << (j - base));

        for (size_t k = 0; k < ls.num_segments(); k++) {
          const auto& seg = ls.get_line_segment(k);
          const auto& Q1 = seg.get_p1();
          const auto& Q2 = seg.get_p2();

          if (has_boundary) {
            if (k == 0 && P == Q1) {
              /* boundary case handling (head) */
              retval |= IM__INTER_BOUND_0D;
              matched = true;
              if (j >= base && j < base + LONG_NBITS) head_matched |= (1UL << (j - base));
              continue;
            } else if (k == ls.num_segments() - 1 && P == Q2) {
              /* boundary case handling (tail) */
              retval |= IM__INTER_BOUND_0D;
              matched = true;
              if (j >= base && j < base + LONG_NBITS) tail_matched |= (1UL << (j - base));
              continue;
            }
          }
          if (seg.covers(P)) {
            retval |= IM__INTER_INTER_0D;
            matched = true;
          }
        }
      }
      /*
       * This point is neither interior nor boundary of linestrings
       */
      if (!matched) retval |= IM__INTER_EXTER_0D;
    }
    /*
     * If herea are any linestring-edges not referenced by the points,
     * it needs to set EXTER-BOUND item.
     */
    if (head_matched != boundary_mask || tail_matched != boundary_mask) {
      retval |= IM__EXTER_BOUND_0D;
      break;
    }
  }
  return retval;
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2);

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    geom2.get_prefix_sum_rings(), geom2.get_vertices(),
                                    geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPoint<POINT_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  uint32_t nloops1;
  uint32_t nloops2;
  int32_t retval = IM__EXTER_EXTER_2D;

  if (geom1.empty()) {
    if (geom2.empty()) return IM__EXTER_EXTER_2D;
    return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
  } else if (geom2.empty())
    return IM__INTER_EXTER_0D | IM__EXTER_EXTER_2D;

  nloops1 = geom1.num_points();
  nloops2 = geom2.num_polygons();

  retval = IM__EXTER_EXTER_2D;
  for (int i = 0; i < nloops1; i++) {
    const auto& pt = geom1.get_point(i);
    bool matched = false;

    for (int j = 0; j < nloops2; j++) {
      const auto& poly = geom2.get_polygon(j);
      /* skip empty polygon */
      if (poly.empty()) continue;
      retval |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;
      auto& mbr = poly.get_mbr();
      if (!mbr.covers(pt.as_float())) {
        continue;
      }

      /* dive into the polygon */
      switch (poly.locate_point(pt)) {
        case PointLocation::kInside: {
          matched = true;
          retval |= IM__INTER_INTER_0D;
          break;
        }
        case PointLocation::kBoundary: {
          matched = true;
          retval |= IM__INTER_BOUND_0D;
          break;
        }
        case PointLocation::kOutside: {
          break;
        }
        default:
          return -1; /* error */
      }
    }
    if (!matched) retval |= IM__INTER_EXTER_0D;
  }
  return retval;
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1, const POINT_T& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1,
                               const MultiPoint<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1,
                               const LineString<POINT_T>& geom2) {
  MultiLineString<POINT_T, size_t> m1, m2;
  size_t prefix_sum_parts1[2] = {0, geom1.num_points()};
  size_t prefix_sum_parts2[2] = {0, geom2.num_points()};

  if (geom1.num_points() > 0) {
    m1 = {ArrayView<size_t>(prefix_sum_parts1, 2), geom1.get_vertices(), geom1.get_mbr()};
  }

  if (geom2.num_points() > 0) {
    m2 = {ArrayView<size_t>(prefix_sum_parts2, 2), geom2.get_vertices(), geom2.get_mbr()};
  }
  return relate(m1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  MultiLineString<POINT_T, INDEX_T> m1;
  INDEX_T prefix_sum_parts1[2] = {0, (INDEX_T)geom1.num_points()};
  if (geom1.num_points() > 0) {
    m1 = {ArrayView<INDEX_T>(prefix_sum_parts1, 2), geom1.get_vertices(),
          geom1.get_mbr()};
  }
  return relate(m1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  MultiLineString<POINT_T, INDEX_T> m1;
  INDEX_T prefix_sum_parts1[2] = {0, (INDEX_T)geom1.num_points()};
  if (geom1.num_points() > 0) {
    m1 = {ArrayView<INDEX_T>(prefix_sum_parts1, 2), geom1.get_vertices(),
          geom1.get_mbr()};
  }

  auto prefix_sum_rings = geom2.get_prefix_sum_rings();
  auto vertices = geom2.get_vertices();

  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    prefix_sum_rings, vertices, geom2.get_mbr());
  return relate(m1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const LineString<POINT_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  MultiLineString<POINT_T, INDEX_T> m1;
  INDEX_T prefix_sum_parts1[2] = {0, (INDEX_T)geom1.num_points()};
  if (geom1.num_points() > 0) {
    m1 = {ArrayView<INDEX_T>(prefix_sum_parts1, 2), geom1.get_vertices(),
          geom1.get_mbr()};
  }
  return relate(m1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const POINT_T& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const MultiPoint<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2);

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const LineString<POINT_T>& geom2) {
  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_points()};
  MultiLineString<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                       geom2.get_vertices(), geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST int32_t relate(bool p_has_boundary, POINT_T P1, bool p1_is_head, POINT_T P2,
                        bool p2_is_tail, const MultiLineString<POINT_T, INDEX_T>& geom,
                        uint32_t start) {
  int32_t retval = IM__EXTER_EXTER_2D;
  bool p1_contained = false;
  bool p2_contained = false;
  uint32_t index = start;
  LineSegment<POINT_T> seg_p(P1, P2);
  auto nloops = geom.num_line_strings();
  bool has_line = false;

  for (int k = 0; k < nloops; k++) {
    POINT_T Q1, Q2;
    int32_t __j = 2;
    const auto& line = geom.get_line_string(k);

    if (line.empty()) {
      continue; /* skip empty line */
    }
    has_line = true;
    // closed line string has no boundary
    bool q_has_boundary = !line.is_closed();

    if (start == 0) {
      Q1 = line.get_point(0);
      index++;
    } else if (index + line.num_points() <= start) {
      index += line.num_points();
      continue; /* skip this sub-line */
    } else {
      assert(index - start < line.num_points());
      Q1 = line.get_point(index - start);
      index++;
      __j = index - start + 2;
      start = 0;
    }

    for (int j = __j; j <= line.num_points(); j++, index++, Q1 = Q2) {
      bool q1_is_head = (j == 2);
      bool q2_is_tail = (j == line.num_points());
      int32_t status;

      Q2 = line.get_point(j - 1);

      LineSegment<POINT_T> seg_q(Q1, Q2);
      LineSegment<POINT_T> seg_p1q2(P1, Q2);
      LineSegment<POINT_T> seg_q1p2(Q1, P2);
      LineSegment<POINT_T> seg_p1q1(P1, Q1);
      LineSegment<POINT_T> seg_p2q1(P2, Q1);
      LineSegment<POINT_T> seg_q2p2(Q2, P2);
      LineSegment<POINT_T> seg_p2q2(P2, Q2);

      auto qp1 = seg_q.orientation(P1);
      auto qp2 = seg_q.orientation(P2);
      if ((qp1 > 0 && qp2 > 0) || (qp1 < 0 && qp2 < 0)) continue; /* no intersection */

      auto p1_in_qq = seg_q.locate_point(P1);
      auto p2_in_qq = seg_q.locate_point(P2);

      /* P1 is on Q1-Q2 */
      if (p1_in_qq != PointLocation::kOutside) {
        p1_contained = true;
        bool p1_is_bound = p_has_boundary && p1_is_head;
        bool p1_on_q_bound =
            q_has_boundary && ((q1_is_head && P1 == Q1) || (q2_is_tail && P1 == Q2));

        if (p1_is_bound && p1_on_q_bound) {
          retval |= IM__BOUND_BOUND_0D;
        } else if (p1_is_bound && !p1_on_q_bound) {
          retval |= IM__BOUND_INTER_0D;
        } else if (!p1_is_bound && p1_on_q_bound) {
          retval |= IM__INTER_BOUND_0D;
        } else {
          retval |= IM__INTER_INTER_0D;
        }
      }

      /* P2 is on Q1-Q2 */
      if (p2_in_qq != PointLocation::kOutside) {
        p2_contained = true;
        bool p2_is_bound = p_has_boundary && p2_is_tail;
        bool p2_on_q_bound =
            q_has_boundary && ((q1_is_head && P2 == Q1) || (q2_is_tail && P2 == Q2));

        if (p2_is_bound && p2_on_q_bound) {
          retval |= IM__BOUND_BOUND_0D;
        } else if (p2_is_bound && !p2_on_q_bound) {
          retval |= IM__BOUND_INTER_0D;
        } else if (!p2_is_bound && p2_on_q_bound) {
          retval |= IM__INTER_BOUND_0D;
        } else {
          retval |= IM__INTER_INTER_0D;
        }
      }

      /* P1-P2 and Q1-Q2 are colinear */
      if (qp1 == 0 && qp2 == 0) {
        if (p1_in_qq != PointLocation::kOutside && p2_in_qq != PointLocation::kOutside) {
          /* P1-P2 is fully contained by Q1-Q2 */
          p1_contained = p2_contained = true;
          if (P1 == P2)
            retval |= IM__INTER_INTER_0D;
          else
            retval |= IM__INTER_INTER_1D;
          goto out;
        } else if (p1_in_qq != PointLocation::kOutside &&
                   p2_in_qq == PointLocation::kOutside) {
          /* P1 is in Q1-Q2, but P2 is not, so Qx-P2 shall remain */
          p1_contained = true;
          if (seg_p.locate_point(Q1) == PointLocation::kInside) {
            P1 = Q1;
            p1_is_head = false;
            retval |= IM__INTER_INTER_1D;
          } else if (seg_p.locate_point(Q2) == PointLocation::kInside) {
            P1 = Q2;
            p1_is_head = false;
            retval |= IM__INTER_INTER_1D;
          }
        } else if (p1_in_qq == PointLocation::kOutside &&
                   p2_in_qq != PointLocation::kOutside) {
          /* P2 is in Q1-Q2, but P1 is not, so Qx-P1 shall remain */
          p2_contained = true;
          if (seg_p.locate_point(Q1) == PointLocation::kInside) {
            P2 = Q1;
            p2_is_tail = false;
            retval |= IM__INTER_INTER_1D;
          } else if (seg_p.locate_point(Q2) == PointLocation::kInside) {
            P2 = Q2;
            p2_is_tail = false;
            retval |= IM__INTER_INTER_1D;
          }
        } else if (seg_p1q2.locate_point(Q1) != PointLocation::kOutside &&
                   seg_q1p2.locate_point(Q2) != PointLocation::kOutside) {
          /* P1-Q1-Q2-P2 */
          if (P1 != Q1) {
            status = relate(p_has_boundary, P1, p1_is_head, Q1, false, geom, index + 1);
            if (status < 0) return -1;
            retval |= status;
          }
          if (Q2 != P2) {
            status = relate(p_has_boundary, Q2, false, P2, p2_is_tail, geom, index + 1);
            if (status < 0) return -1;
            retval |= status;
          }
          goto out;
        } else if (seg_p1q1.locate_point(Q2) != PointLocation::kOutside &&
                   seg_q2p2.locate_point(Q1) != PointLocation::kOutside) {
          /* P1-Q2-Q1-P2 */
          if (P1 != Q2) {
            status = relate(p_has_boundary, P1, p1_is_head, Q2, false, geom, index + 1);
            if (status < 0) return -1;
            retval |= status;
          }
          if (Q1 != P2) {
            status = relate(p_has_boundary, Q1, false, P2, p2_is_tail, geom, index + 1);
            if (status < 0) return -1;
            retval |= status;
          }
          goto out;
        } else {
          /* elsewhere P1-P2 and Q1-Q2 have no intersection */
        }
      } else {
        auto pq1 = seg_p2q1.orientation(P1);
        auto pq2 = seg_p2q2.orientation(P1);

        /* P1-P2 and Q1-Q2 crosses mutually */
        if (((pq1 > 0 && pq2 < 0) || (pq1 < 0 && pq2 > 0)) &&
            ((qp1 > 0 && qp2 < 0) || (qp1 < 0 && qp2 > 0))) {
          retval |= IM__INTER_INTER_0D;
        }
      }
    }
  }
  if (P1 != P2) retval |= IM__INTER_EXTER_1D;
out:
  if (has_line && p_has_boundary) {
    if (p1_is_head && !p1_contained) retval |= IM__BOUND_EXTER_0D;
    if (p2_is_tail && !p2_contained) retval |= IM__BOUND_EXTER_0D;
  }
  return retval;
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  POINT_T P1, P2;
  uint32_t nloops;
  int32_t retval1 = IM__EXTER_EXTER_2D;
  int32_t retval2 = IM__EXTER_EXTER_2D;
  int32_t status;

  /* special empty cases */
  if (geom1.empty()) {
    if (geom2.empty()) return IM__EXTER_EXTER_2D;
    return IM__EXTER_INTER_1D | IM__EXTER_BOUND_0D | IM__EXTER_EXTER_2D;
  } else if (geom2.empty())
    return IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_EXTER_2D;

  /* 1st loop */
  nloops = geom1.num_line_strings();
  for (int k = 0; k < nloops; k++) {
    const auto& line = geom1.get_line_string(k);
    if (line.empty()) continue; /* skip empty line */
    P1 = line.get_point(0);
    bool has_boundary = !line.is_closed();

    for (int i = 2; i <= line.num_points(); i++, P1 = P2) {
      P2 = line.get_point(i - 1);
      status = relate(has_boundary, P1, i == 2, P2, i == line.num_points(), geom2, 0);
      if (status < 0) return -1;
      retval1 |= status;
    }
  }
  /* 2nd loop (twisted) */
  nloops = geom2.num_line_strings();
  for (int k = 0; k < nloops; k++) {
    const auto& line = geom2.get_line_string(k);
    if (line.empty()) continue; /* skip empty line */
    P1 = line.get_point(0);
    bool has_boundary = !line.is_closed();

    for (int j = 2; j <= line.num_points(); j++, P1 = P2) {
      P2 = line.get_point(j - 1);
      status = relate(has_boundary, P1, j == 2, P2, j == line.num_points(), geom1, 0);
      if (status < 0) return -1;
      retval2 |= status;
    }
  }
  return retval1 | IM__TWIST(retval2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2);

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  auto prefix_sum_rings = geom2.get_prefix_sum_rings();
  auto vertices = geom2.get_vertices();

  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    prefix_sum_rings, vertices, geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiLineString<POINT_T, INDEX_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  int32_t retval = IM__EXTER_EXTER_2D;
  int32_t status;
  /* special empty cases */
  if (geom1.empty()) {
    if (geom2.empty()) return IM__EXTER_EXTER_2D;
    return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
  } else if (geom2.empty())
    return IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D | IM__EXTER_EXTER_2D;

  retval = IM__EXTER_EXTER_2D;

  if (!geom2.empty()) {
    retval |= IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D;
  }

  if (!geom1.get_mbr().intersects(geom2.get_mbr())) {
    return (retval | IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D);
  }

  for (size_t k = 0; k < geom1.num_line_strings(); k++) {
    bool has_boundary;
    bool p1_is_head = true;

    auto ls = geom1.get_line_string(k);
    if (ls.empty()) continue; /* empty */
    auto nitems = ls.num_points();
    POINT_T P1;
    auto P2 = ls.get_point(nitems - 1);

    /* decrement nitems if tail items are duplicated */
    for (nitems = ls.num_points(); nitems >= 2; nitems--) {
      P1 = ls.get_point(nitems - 2);
      if (P1 != P2) break;
    }
    /* checks for each edge */
    P1 = ls.get_point(0);
    has_boundary = P1 != P2;
    for (int i = 2; i <= nitems; i++) {
      P2 = ls.get_point(i - 1);
      if (P1 == P2) continue;

      const auto& mbr2 = geom2.get_mbr();

      if (std::max(P1.x(), P2.x()) < mbr2.get_min().x() ||
          std::min(P1.x(), P2.x()) > mbr2.get_max().x() ||
          std::max(P1.y(), P2.y()) < mbr2.get_min().y() ||
          std::min(P1.y(), P2.y()) > mbr2.get_max().y()) {
        retval |= (IM__INTER_EXTER_1D | IM__BOUND_EXTER_0D);
      } else {
        status = relate(P1, (has_boundary && p1_is_head), P2,
                        (has_boundary && i == nitems), geom2, 0, false);
        if (status < 0) return -1;
        retval |= status;
      }
      P1 = P2;
      p1_is_head = false;
    }

    if (has_boundary) {
      status = (IM__LINE_HEAD_CONTAINED | IM__LINE_TAIL_CONTAINED);
      if ((retval & status) != status) retval |= IM__BOUND_EXTER_0D;
    }
  }
  return (retval & IM__MASK_FULL);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const POINT_T& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const POINT_T& geom2, PointLocation location) {
  return IM__TWIST(relate(geom2, geom1, location));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const MultiPoint<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const LineString<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  auto prefix_sum_rings1 = geom1.get_prefix_sum_rings();
  auto vertices1 = geom1.get_vertices();

  INDEX_T prefix_sum_parts1[2] = {0, (INDEX_T)geom1.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m1(ArrayView<INDEX_T>(prefix_sum_parts1, 2),
                                    prefix_sum_rings1, vertices1, geom1.get_mbr());

  auto prefix_sum_rings2 = geom2.get_prefix_sum_rings();
  auto vertices2 = geom2.get_vertices();

  INDEX_T prefix_sum_parts2[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts2, 2),
                                    prefix_sum_rings2, vertices2, geom2.get_mbr());
  return relate(m1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const Polygon<POINT_T, INDEX_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  auto prefix_sum_rings = geom1.get_prefix_sum_rings();
  auto vertices = geom1.get_vertices();

  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom1.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m1(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    prefix_sum_rings, vertices, geom1.get_mbr());
  return relate(m1, geom2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const POINT_T& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const POINT_T& geom2, ArrayView<PointLocation> locations) {
  return IM__TWIST(relate(geom2, geom1, locations));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const MultiPoint<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const LineString<POINT_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const MultiLineString<POINT_T, INDEX_T>& geom2) {
  return IM__TWIST(relate(geom2, geom1));
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const Polygon<POINT_T, INDEX_T>& geom2) {
  auto prefix_sum_rings = geom2.get_prefix_sum_rings();
  auto vertices = geom2.get_vertices();

  INDEX_T prefix_sum_parts[2] = {0, (INDEX_T)geom2.num_rings()};

  MultiPolygon<POINT_T, INDEX_T> m2(ArrayView<INDEX_T>(prefix_sum_parts, 2),
                                    prefix_sum_rings, vertices, geom2.get_mbr());
  return relate(geom1, m2);
}

template <typename POINT_T, typename INDEX_T>
DEV_HOST_INLINE int32_t relate(const MultiPolygon<POINT_T, INDEX_T>& geom1,
                               const MultiPolygon<POINT_T, INDEX_T>& geom2) {
  int32_t nloops;
  int32_t retval = IM__EXTER_EXTER_2D;

  /* special empty cases */
  if (geom1.empty()) {
    if (geom2.empty()) return IM__EXTER_EXTER_2D;
    return IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D;
  } else if (geom2.empty())
    return IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_EXTER_2D;

  if (!geom1.get_mbr().intersects(geom2.get_mbr())) {
    return (IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D | IM__EXTER_INTER_2D |
            IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
  }

  nloops = geom1.num_polygons();
  for (int k = 0; k < nloops; k++) {
    int32_t __retval = 0; /* pending result for each polygon */
    const auto& poly = geom1.get_polygon(k);

    for (int i = 0; i < poly.num_rings(); i++) {
      const auto& ring = poly.get_ring(i);
      auto status = relate(ring, geom2);

      if (status < 0) return -1;
      if (i == 0) {
        __retval = status;
        if ((__retval & IM__INTER_INTER_2D) == 0)
          break; /* disjoint, so we can skip holes */
      } else {
        /* add boundaries, if touched/crossed */
        __retval |= (status & IM__BOUND_BOUND_2D);

        /* geom2 is disjoint from the hole? */
        if ((status & IM__INTER_INTER_2D) == 0) continue;
        /*
         * geom2 is fully contained by the hole, so reconstruct
         * the DE9-IM as disjointed polygon.
         */
        if ((status & IM__INTER_EXTER_2D) != 0 && (status & IM__EXTER_INTER_2D) == 0) {
          __retval =
              ((status & IM__BOUND_BOUND_2D) | IM__INTER_EXTER_2D | IM__BOUND_EXTER_1D |
               IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D | IM__EXTER_EXTER_2D);
          break;
        }

        /*
         * geom2 has a valid intersection with the hole, add it.
         */
        if ((status & IM__INTER_INTER_2D) != 0) {
          __retval |= (IM__BOUND_INTER_1D | IM__EXTER_INTER_2D | IM__EXTER_BOUND_1D);
          // FIXME: Only apply IM__EXTER_BOUND_1D if exterior of geom1 intersects boundary
          // of geom2
          // Refer: RelateTest - PolygonsNestedWithHole
          break;
        }
      }
    }
    retval |= __retval;
  }
  return retval;
}
}  // namespace gpuspatial
