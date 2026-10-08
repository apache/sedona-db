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

//! Grid placement shared by the raster constructors (`RS_MakeEmptyRaster`,
//! `RS_MakeRaster`): resolving an extent geometry or an affine transform plus
//! CRS argument into a geotransform and CRS, one row at a time.

use arrow_array::{Array, Float64Array, Int64Array, StringArray};
use arrow_schema::DataType;
use datafusion_common::{
    cast::{as_float64_array, as_int64_array, as_string_array},
    config::ConfigOptions,
    error::Result,
    exec_datafusion_err, exec_err,
};
use datafusion_expr::ColumnarValue;
use sedona_common::option::SedonaOptions;
use sedona_expr::item_crs::parse_item_crs_arg_type;
use sedona_geometry::{
    bounds::{WkbBounder2D, wkb_bounds_xy},
    interval::IntervalTrait,
    types::Edges,
};
use sedona_schema::{
    crs::{CachedSRIDToCrs, normalize_crs},
    datatypes::SedonaType,
};

use crate::executor::{GeomWkbCrsAccessor, RasterExecutor};

/// A resolved grid placement for one output raster.
pub(crate) struct GridGeometry {
    pub(crate) upper_left_x: f64,
    pub(crate) upper_left_y: f64,
    pub(crate) scale_x: f64,
    pub(crate) scale_y: f64,
    pub(crate) skew_x: f64,
    pub(crate) skew_y: f64,
    pub(crate) crs: Option<String>,
}

/// Per-row accessors for the grid-placement arguments of one kernel form.
pub(crate) enum Placement {
    Extent {
        accessor: GeomWkbCrsAccessor,
        /// Set when the extent is a geography; bounds follow spherical edges.
        bounder: Option<Box<dyn WkbBounder2D>>,
    },
    CellSize {
        upper_left_x: Float64Array,
        upper_left_y: Float64Array,
        cell_size: Float64Array,
    },
    Affine(Box<AffineColumns>),
}

/// The seven per-row columns of the affine form (boxed so the enum stays small).
pub(crate) struct AffineColumns {
    upper_left_x: Float64Array,
    upper_left_y: Float64Array,
    scale_x: Float64Array,
    scale_y: Float64Array,
    skew_x: Float64Array,
    skew_y: Float64Array,
    crs: CrsColumn,
}

/// The CRS argument of the affine form: an integer SRID or a CRS string.
pub(crate) enum CrsColumn {
    /// An EPSG code; `0` means no CRS.
    Srid {
        srid: Int64Array,
        srid_to_crs: CachedSRIDToCrs,
    },
    /// Any CRS string SedonaDB understands (`authority:code`, PROJJSON or
    /// WKT), normalized as `RS_SetCRS` does.
    Definition(StringArray),
}

impl CrsColumn {
    /// An integer SRID column from argument `arg`.
    pub(crate) fn srid(arg: &ColumnarValue, n: usize) -> Result<Self> {
        Ok(CrsColumn::Srid {
            srid: int_column(arg, n)?,
            srid_to_crs: CachedSRIDToCrs::new(),
        })
    }

    /// A CRS string column from argument `arg`.
    pub(crate) fn definition(arg: &ColumnarValue, n: usize) -> Result<Self> {
        Ok(CrsColumn::Definition(string_column(arg, n)?))
    }

    fn is_null(&self, i: usize) -> bool {
        match self {
            CrsColumn::Srid { srid, .. } => srid.is_null(i),
            CrsColumn::Definition(crs) => crs.is_null(i),
        }
    }

    fn crs(&mut self, name: &str, i: usize) -> Result<Option<String>> {
        match self {
            CrsColumn::Srid { srid, srid_to_crs } => srid_to_crs.get_crs(srid.value(i)),
            CrsColumn::Definition(crs) => {
                let crs = crs.value(i);
                normalize_crs(crs)
                    .map_err(|e| exec_datafusion_err!("{name}: invalid crs '{crs}': {e}"))
            }
        }
    }
}

impl Placement {
    /// The extent form: argument `g` is a geometry or geography whose envelope
    /// and CRS place the grid.
    pub(crate) fn extent(
        executor: &RasterExecutor,
        arg_types: &[SedonaType],
        g: usize,
        name: &str,
        config_options: Option<&ConfigOptions>,
    ) -> Result<Self> {
        Ok(Placement::Extent {
            accessor: executor.make_geom_wkb_crs_accessor(g)?,
            // A geography's envelope follows spherical edges, so it needs the
            // bounder registered for them rather than a planar coordinate scan.
            bounder: match edges_of(&arg_types[g])? {
                Edges::Spherical => Some(spherical_bounder(name, config_options)?),
                _ => None,
            },
        })
    }

    /// The square-pixel form: `upper_left_x, upper_left_y, cell_size` from
    /// argument `g` on, with no CRS.
    pub(crate) fn cell_size(args: &[ColumnarValue], g: usize, n: usize) -> Result<Self> {
        Ok(Placement::CellSize {
            upper_left_x: f64_column(&args[g], n)?,
            upper_left_y: f64_column(&args[g + 1], n)?,
            cell_size: f64_column(&args[g + 2], n)?,
        })
    }

    /// The full affine form: `upper_left_x, upper_left_y, scale_x, scale_y,
    /// skew_x, skew_y` from argument `g` on, followed by `crs`.
    pub(crate) fn affine(
        args: &[ColumnarValue],
        g: usize,
        n: usize,
        crs: CrsColumn,
    ) -> Result<Self> {
        Ok(Placement::Affine(Box::new(AffineColumns {
            upper_left_x: f64_column(&args[g], n)?,
            upper_left_y: f64_column(&args[g + 1], n)?,
            scale_x: f64_column(&args[g + 2], n)?,
            scale_y: f64_column(&args[g + 3], n)?,
            skew_x: f64_column(&args[g + 4], n)?,
            skew_y: f64_column(&args[g + 5], n)?,
            crs,
        })))
    }

    /// Resolve row `i`'s placement, or `None` when any of its inputs is null.
    /// `name` prefixes error messages.
    pub(crate) fn geometry(
        &mut self,
        name: &str,
        i: usize,
        width: i64,
        height: i64,
    ) -> Result<Option<GridGeometry>> {
        match self {
            Placement::Extent { accessor, bounder } => {
                let (maybe_wkb, crs) = accessor.get(i)?;
                let Some(wkb) = maybe_wkb else {
                    return Ok(None);
                };
                let (xmin, ymin, xmax, ymax) = extent_bounds(name, wkb, bounder.as_mut())?;
                Ok(Some(GridGeometry {
                    upper_left_x: xmin,
                    upper_left_y: ymax,
                    scale_x: (xmax - xmin) / width as f64,
                    scale_y: -(ymax - ymin) / height as f64,
                    skew_x: 0.0,
                    skew_y: 0.0,
                    crs: crs.map(|c| c.to_crs_string()),
                }))
            }
            Placement::CellSize {
                upper_left_x,
                upper_left_y,
                cell_size,
            } => {
                if upper_left_x.is_null(i) || upper_left_y.is_null(i) || cell_size.is_null(i) {
                    return Ok(None);
                }
                let cell_size = cell_size.value(i);
                Ok(Some(GridGeometry {
                    upper_left_x: upper_left_x.value(i),
                    upper_left_y: upper_left_y.value(i),
                    scale_x: cell_size,
                    scale_y: -cell_size,
                    skew_x: 0.0,
                    skew_y: 0.0,
                    crs: None,
                }))
            }
            Placement::Affine(cols) => {
                if cols.upper_left_x.is_null(i)
                    || cols.upper_left_y.is_null(i)
                    || cols.scale_x.is_null(i)
                    || cols.scale_y.is_null(i)
                    || cols.skew_x.is_null(i)
                    || cols.skew_y.is_null(i)
                    || cols.crs.is_null(i)
                {
                    return Ok(None);
                }
                let (upper_left_x, upper_left_y) =
                    (cols.upper_left_x.value(i), cols.upper_left_y.value(i));
                let (scale_x, scale_y) = (cols.scale_x.value(i), cols.scale_y.value(i));
                let (skew_x, skew_y) = (cols.skew_x.value(i), cols.skew_y.value(i));
                let transform = [upper_left_x, upper_left_y, scale_x, scale_y, skew_x, skew_y];
                if !transform.iter().all(|v| v.is_finite()) {
                    return exec_err!(
                        "{name}: geotransform must be finite, got upper_left=({upper_left_x}, \
                         {upper_left_y}) scale=({scale_x}, {scale_y}) skew=({skew_x}, {skew_y})"
                    );
                }
                // A zero determinant collapses the grid onto a line or point,
                // so pixels have no area and the transform has no inverse.
                if scale_x * scale_y - skew_x * skew_y == 0.0 {
                    return exec_err!(
                        "{name}: geotransform must be invertible, got scale=({scale_x}, \
                         {scale_y}) skew=({skew_x}, {skew_y}) with a zero determinant"
                    );
                }
                Ok(Some(GridGeometry {
                    upper_left_x,
                    upper_left_y,
                    scale_x,
                    scale_y,
                    skew_x,
                    skew_y,
                    crs: cols.crs.crs(name, i)?,
                }))
            }
        }
    }
}

/// The `(xmin, ymin, xmax, ymax)` envelope of an extent geometry, which must
/// span a positive width and height for the pixel size to be defined.
fn extent_bounds(
    name: &str,
    wkb: &[u8],
    bounder: Option<&mut Box<dyn WkbBounder2D>>,
) -> Result<(f64, f64, f64, f64)> {
    let ((xmin, xmax), (ymin, ymax)) = match bounder {
        // Geography: the registered spherical bounder decides the envelope,
        // which is not the planar extent of the coordinates (it accounts for
        // geodesic edges and antimeridian wraparound).
        Some(bounder) => {
            bounder.clear();
            bounder
                .update_wkb_bytes(wkb)
                .map_err(|e| exec_datafusion_err!("{name}: invalid extent geography: {e}"))?;
            let (x, y) = bounder.finish();
            if x.is_empty() || y.is_empty() {
                return exec_err!("{name}: extent geometry is empty");
            }
            // An extent crossing the antimeridian has a wraparound longitude
            // interval (lo > hi, covering lo..180 and -180..hi). Unroll it east
            // past 180 into one continuous span, e.g. [170, -170] -> [170, 190],
            // so the grid covers the 20 degrees between rather than the 340
            // degrees outside.
            let x = if x.is_wraparound() {
                (x.lo(), x.hi() + 360.0)
            } else {
                (x.lo(), x.hi())
            };
            (x, (y.lo(), y.hi()))
        }
        None => {
            let bbox = wkb_bounds_xy(wkb)
                .map_err(|e| exec_datafusion_err!("{name}: invalid extent geometry: {e}"))?;
            if bbox.is_empty() {
                return exec_err!("{name}: extent geometry is empty");
            }
            (
                (bbox.x().lo(), bbox.x().hi()),
                (bbox.y().lo(), bbox.y().hi()),
            )
        }
    };
    // A full interval (e.g. a geography around a pole spans every longitude)
    // has infinite bounds, which no pixel size can cover.
    if ![xmin, ymin, xmax, ymax].iter().all(|v| v.is_finite()) {
        return exec_err!(
            "{name}: extent must have a finite envelope, got \
             [{xmin}, {ymin}, {xmax}, {ymax}]"
        );
    }
    if !(xmax > xmin && ymax > ymin) {
        return exec_err!(
            "{name}: extent must span a positive width and height, \
             got envelope [{xmin}, {ymin}, {xmax}, {ymax}]"
        );
    }
    Ok((xmin, ymin, xmax, ymax))
}

/// Edge interpretation of a geometry/geography argument type, looking through
/// an item-level CRS struct (e.g. from `ST_SetCRS` with a CRS column) to the
/// geography inside it.
fn edges_of(arg_type: &SedonaType) -> Result<Edges> {
    let (item_type, _) = parse_item_crs_arg_type(arg_type)?;
    Ok(match item_type {
        SedonaType::Wkb(edges, _)
        | SedonaType::WkbView(edges, _)
        | SedonaType::WkbLarge(edges, _) => edges,
        _ => Edges::Planar,
    })
}

/// The spherical bounder registered in the session.
///
/// Spherical bounding needs an external implementation (s2geography), so unlike
/// the planar case there is no built-in fallback: a geography extent without a
/// registered bounder is an error rather than a silently planar envelope.
fn spherical_bounder(
    name: &str,
    config_options: Option<&ConfigOptions>,
) -> Result<Box<dyn WkbBounder2D>> {
    config_options
        .and_then(|options| options.extensions.get::<SedonaOptions>())
        .and_then(|options| {
            options
                .runtime
                .bounder_factory()
                .bounder_for_edge_type(Edges::Spherical)
        })
        .ok_or_else(|| {
            exec_datafusion_err!(
                "{name}: a geography extent needs a spherical bounder, \
                 but none is registered in this session"
            )
        })
}

pub(crate) fn int_column(arg: &ColumnarValue, n: usize) -> Result<Int64Array> {
    let array = arg.clone().cast_to(&DataType::Int64, None)?.into_array(n)?;
    Ok(as_int64_array(&array)?.clone())
}

pub(crate) fn f64_column(arg: &ColumnarValue, n: usize) -> Result<Float64Array> {
    let array = arg
        .clone()
        .cast_to(&DataType::Float64, None)?
        .into_array(n)?;
    Ok(as_float64_array(&array)?.clone())
}

pub(crate) fn string_column(arg: &ColumnarValue, n: usize) -> Result<StringArray> {
    let array = arg.clone().cast_to(&DataType::Utf8, None)?.into_array(n)?;
    Ok(as_string_array(&array)?.clone())
}
