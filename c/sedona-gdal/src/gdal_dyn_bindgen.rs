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
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]

use std::os::raw::{c_char, c_double, c_int, c_uchar, c_uint, c_void};

// --- Scalar type aliases ---

pub type GSpacing = i64;
pub type CPLErr = c_int;
pub type OGRErr = c_int;
pub type GDALRWFlag = c_int;
pub type OGRwkbByteOrder = c_int;
pub type GDALOpenFlags = c_uint;
pub type GDALRIOResampleAlg = c_int;

// --- Opaque handle types ---

pub type GDALDatasetH = *mut c_void;
pub type GDALDriverH = *mut c_void;
pub type GDALRasterBandH = *mut c_void;
pub type OGRSpatialReferenceH = *mut c_void;
pub type OGRGeometryH = *mut c_void;
pub type OGRLayerH = *mut c_void;
pub type OGRFeatureH = *mut c_void;
pub type OGRFieldDefnH = *mut c_void;
pub type VSILFILE = *mut c_void;

// --- Enum types ---

#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum GDALDataType {
    GDT_Unknown = 0,
    GDT_Byte = 1,
    GDT_Int8 = 14,
    GDT_UInt16 = 2,
    GDT_Int16 = 3,
    GDT_UInt32 = 4,
    GDT_Int32 = 5,
    GDT_UInt64 = 12,
    GDT_Int64 = 13,
    GDT_Float16 = 15,
    GDT_Float32 = 6,
    GDT_Float64 = 7,
    GDT_CInt16 = 8,
    GDT_CInt32 = 9,
    GDT_CFloat16 = 16,
    GDT_CFloat32 = 10,
    GDT_CFloat64 = 11,
}

impl GDALDataType {
    #[allow(clippy::result_unit_err)]
    pub fn try_from_ordinal(value: i32) -> Result<Self, ()> {
        match value {
            0 => Ok(GDALDataType::GDT_Unknown),
            1 => Ok(GDALDataType::GDT_Byte),
            2 => Ok(GDALDataType::GDT_UInt16),
            3 => Ok(GDALDataType::GDT_Int16),
            4 => Ok(GDALDataType::GDT_UInt32),
            5 => Ok(GDALDataType::GDT_Int32),
            6 => Ok(GDALDataType::GDT_Float32),
            7 => Ok(GDALDataType::GDT_Float64),
            8 => Ok(GDALDataType::GDT_CInt16),
            9 => Ok(GDALDataType::GDT_CInt32),
            10 => Ok(GDALDataType::GDT_CFloat32),
            11 => Ok(GDALDataType::GDT_CFloat64),
            12 => Ok(GDALDataType::GDT_UInt64),
            13 => Ok(GDALDataType::GDT_Int64),
            14 => Ok(GDALDataType::GDT_Int8),
            15 => Ok(GDALDataType::GDT_Float16),
            16 => Ok(GDALDataType::GDT_CFloat16),
            _ => Err(()),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OGRwkbGeometryType {
    wkbUnknown = 0,
    wkbPoint = 1,
    wkbLineString = 2,
    wkbPolygon = 3,
    wkbMultiPoint = 4,
    wkbMultiLineString = 5,
    wkbMultiPolygon = 6,
    wkbGeometryCollection = 7,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum OGRFieldType {
    OFTInteger = 0,
    OFTIntegerList = 1,
    OFTReal = 2,
    OFTRealList = 3,
    OFTString = 4,
    OFTStringList = 5,
    OFTWideString = 6,
    OFTWideStringList = 7,
    OFTBinary = 8,
    OFTDate = 9,
    OFTTime = 10,
    OFTDateTime = 11,
    OFTInteger64 = 12,
    OFTInteger64List = 13,
}

// --- Function pointer type aliases ---

/// Type alias for the GDAL transformer callback (`GDALTransformerFunc`).
///
/// Signature: `(pTransformerArg, bDstToSrc, nPointCount, x, y, z, panSuccess) -> c_int`
pub type GDALTransformerFunc = unsafe extern "C" fn(
    pTransformerArg: *mut c_void,
    bDstToSrc: c_int,
    nPointCount: c_int,
    x: *mut c_double,
    y: *mut c_double,
    z: *mut c_double,
    panSuccess: *mut c_int,
) -> c_int;

// --- Structs ---

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OGREnvelope {
    pub MinX: c_double,
    pub MaxX: c_double,
    pub MinY: c_double,
    pub MaxY: c_double,
}

/// GDAL progress callback type.
pub type GDALProgressFunc = Option<
    unsafe extern "C" fn(
        dfComplete: c_double,
        pszMessage: *const c_char,
        pProgressArg: *mut c_void,
    ) -> c_int,
>;

/// Extra arguments for `GDALRasterIOEx`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct GDALRasterIOExtraArg {
    pub nVersion: c_int,
    pub eResampleAlg: GDALRIOResampleAlg,
    pub pfnProgress: GDALProgressFunc,
    pub pProgressData: *mut c_void,
    pub bFloatingPointWindowValidity: c_int,
    pub dfXOff: c_double,
    pub dfYOff: c_double,
    pub dfXSize: c_double,
    pub dfYSize: c_double,
}

impl Default for GDALRasterIOExtraArg {
    fn default() -> Self {
        Self {
            nVersion: 1,
            eResampleAlg: GRIORA_NearestNeighbour,
            pfnProgress: None,
            pProgressData: std::ptr::null_mut(),
            bFloatingPointWindowValidity: 0,
            dfXOff: 0.0,
            dfYOff: 0.0,
            dfXSize: 0.0,
            dfYSize: 0.0,
        }
    }
}

// --- GDALRIOResampleAlg constants ---

pub const GRIORA_NearestNeighbour: GDALRIOResampleAlg = 0;
pub const GRIORA_Bilinear: GDALRIOResampleAlg = 1;
pub const GRIORA_Cubic: GDALRIOResampleAlg = 2;
pub const GRIORA_CubicSpline: GDALRIOResampleAlg = 3;
pub const GRIORA_Lanczos: GDALRIOResampleAlg = 4;
pub const GRIORA_Average: GDALRIOResampleAlg = 5;
pub const GRIORA_Mode: GDALRIOResampleAlg = 6;
pub const GRIORA_Gauss: GDALRIOResampleAlg = 7;

// --- GDAL open flags constants ---

pub const GDAL_OF_READONLY: GDALOpenFlags = 0x00;
pub const GDAL_OF_UPDATE: GDALOpenFlags = 0x01;
pub const GDAL_OF_RASTER: GDALOpenFlags = 0x02;
pub const GDAL_OF_VECTOR: GDALOpenFlags = 0x04;
pub const GDAL_OF_VERBOSE_ERROR: GDALOpenFlags = 0x40;

// --- GDALRWFlag constants ---

pub const GF_Read: GDALRWFlag = 0;
pub const GF_Write: GDALRWFlag = 1;

// --- CPLErr constants ---

pub const CE_None: CPLErr = 0;
pub const CE_Debug: CPLErr = 1;
pub const CE_Warning: CPLErr = 2;
pub const CE_Failure: CPLErr = 3;
pub const CE_Fatal: CPLErr = 4;

// --- OGRErr constants ---

pub const OGRERR_NONE: OGRErr = 0;

// --- OSRAxisMappingStrategy type and constants ---

pub type OSRAxisMappingStrategy = c_int;

pub const OAMS_TRADITIONAL_GIS_ORDER: OSRAxisMappingStrategy = 0;
pub const OAMS_AUTHORITY_COMPLIANT: OSRAxisMappingStrategy = 1;
pub const OAMS_CUSTOM: OSRAxisMappingStrategy = 2;

// --- OGRwkbByteOrder constants ---

pub const wkbXDR: OGRwkbByteOrder = 0; // Big endian
pub const wkbNDR: OGRwkbByteOrder = 1; // Little endian

// --- The main API struct mirroring C SedonaGdalApi ---

#[repr(C)]
#[derive(Debug, Copy, Clone, Default)]
pub(crate) struct SedonaGdalApi {
    // --- Dataset ---
    pub GDALOpenEx: Option<
        unsafe extern "C" fn(
            pszFilename: *const c_char,
            nOpenFlags: c_uint,
            papszAllowedDrivers: *const *const c_char,
            papszOpenOptions: *const *const c_char,
            papszSiblingFiles: *const *const c_char,
        ) -> GDALDatasetH,
    >,
    pub GDALClose: Option<unsafe extern "C" fn(hDS: GDALDatasetH)>,
    pub GDALGetRasterXSize: Option<unsafe extern "C" fn(hDataset: GDALDatasetH) -> c_int>,
    pub GDALGetRasterYSize: Option<unsafe extern "C" fn(hDataset: GDALDatasetH) -> c_int>,
    pub GDALGetRasterCount: Option<unsafe extern "C" fn(hDataset: GDALDatasetH) -> c_int>,
    pub GDALGetRasterBand:
        Option<unsafe extern "C" fn(hDS: GDALDatasetH, nBandId: c_int) -> GDALRasterBandH>,
    pub GDALGetGeoTransform:
        Option<unsafe extern "C" fn(hDS: GDALDatasetH, padfTransform: *mut c_double) -> CPLErr>,
    pub GDALSetGeoTransform:
        Option<unsafe extern "C" fn(hDS: GDALDatasetH, padfTransform: *mut c_double) -> CPLErr>,
    pub GDALGetProjectionRef: Option<unsafe extern "C" fn(hDS: GDALDatasetH) -> *const c_char>,
    pub GDALSetProjection:
        Option<unsafe extern "C" fn(hDS: GDALDatasetH, pszProjection: *const c_char) -> CPLErr>,
    pub GDALGetSpatialRef: Option<unsafe extern "C" fn(hDS: GDALDatasetH) -> OGRSpatialReferenceH>,
    pub GDALSetSpatialRef:
        Option<unsafe extern "C" fn(hDS: GDALDatasetH, hSRS: OGRSpatialReferenceH) -> CPLErr>,
    pub GDALCreateCopy: Option<
        unsafe extern "C" fn(
            hDriver: GDALDriverH,
            pszFilename: *const c_char,
            hSrcDS: GDALDatasetH,
            bStrict: c_int,
            papszOptions: *mut *mut c_char,
            pfnProgress: *mut c_void,
            pProgressData: *mut c_void,
        ) -> GDALDatasetH,
    >,
    pub GDALDatasetCreateLayer: Option<
        unsafe extern "C" fn(
            hDS: GDALDatasetH,
            pszName: *const c_char,
            hSpatialRef: OGRSpatialReferenceH,
            eGType: OGRwkbGeometryType,
            papszOptions: *mut *mut c_char,
        ) -> OGRLayerH,
    >,

    // --- Driver ---
    pub GDALAllRegister: Option<unsafe extern "C" fn()>,
    pub GDALGetDriverByName: Option<unsafe extern "C" fn(pszName: *const c_char) -> GDALDriverH>,
    pub GDALCreate: Option<
        unsafe extern "C" fn(
            hDriver: GDALDriverH,
            pszFilename: *const c_char,
            nXSize: c_int,
            nYSize: c_int,
            nBands: c_int,
            eType: GDALDataType,
            papszOptions: *mut *mut c_char,
        ) -> GDALDatasetH,
    >,

    // --- Band ---
    pub GDALAddBand: Option<
        unsafe extern "C" fn(
            hDS: GDALDatasetH,
            eType: GDALDataType,
            papszOptions: *mut *mut c_char,
        ) -> CPLErr,
    >,
    pub GDALRasterIO: Option<
        unsafe extern "C" fn(
            hRBand: GDALRasterBandH,
            eRWFlag: GDALRWFlag,
            nDSXOff: c_int,
            nDSYOff: c_int,
            nDSXSize: c_int,
            nDSYSize: c_int,
            pBuffer: *mut c_void,
            nBXSize: c_int,
            nBYSize: c_int,
            eBDataType: GDALDataType,
            nPixelSpace: GSpacing,
            nLineSpace: GSpacing,
        ) -> CPLErr,
    >,
    pub GDALRasterIOEx: Option<
        unsafe extern "C" fn(
            hRBand: GDALRasterBandH,
            eRWFlag: GDALRWFlag,
            nDSXOff: c_int,
            nDSYOff: c_int,
            nDSXSize: c_int,
            nDSYSize: c_int,
            pBuffer: *mut c_void,
            nBXSize: c_int,
            nBYSize: c_int,
            eBDataType: GDALDataType,
            nPixelSpace: GSpacing,
            nLineSpace: GSpacing,
            psExtraArg: *mut GDALRasterIOExtraArg,
        ) -> CPLErr,
    >,
    pub GDALGetRasterDataType: Option<unsafe extern "C" fn(hBand: GDALRasterBandH) -> GDALDataType>,
    pub GDALGetRasterBandXSize: Option<unsafe extern "C" fn(hBand: GDALRasterBandH) -> c_int>,
    pub GDALGetRasterBandYSize: Option<unsafe extern "C" fn(hBand: GDALRasterBandH) -> c_int>,
    pub GDALGetBlockSize: Option<
        unsafe extern "C" fn(hBand: GDALRasterBandH, pnXSize: *mut c_int, pnYSize: *mut c_int),
    >,
    pub GDALGetRasterNoDataValue:
        Option<unsafe extern "C" fn(hBand: GDALRasterBandH, pbSuccess: *mut c_int) -> c_double>,
    pub GDALSetRasterNoDataValue:
        Option<unsafe extern "C" fn(hBand: GDALRasterBandH, dfValue: c_double) -> CPLErr>,
    pub GDALDeleteRasterNoDataValue: Option<unsafe extern "C" fn(hBand: GDALRasterBandH) -> CPLErr>,
    pub GDALSetRasterNoDataValueAsUInt64:
        Option<unsafe extern "C" fn(hBand: GDALRasterBandH, nValue: u64) -> CPLErr>,
    pub GDALSetRasterNoDataValueAsInt64:
        Option<unsafe extern "C" fn(hBand: GDALRasterBandH, nValue: i64) -> CPLErr>,

    // --- SpatialRef ---
    pub OSRNewSpatialReference:
        Option<unsafe extern "C" fn(pszWKT: *const c_char) -> OGRSpatialReferenceH>,
    pub OSRSetFromUserInput: Option<
        unsafe extern "C" fn(hSRS: OGRSpatialReferenceH, pszDefinition: *const c_char) -> OGRErr,
    >,
    pub OSREPSGTreatsAsLatLong: Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH) -> c_int>,
    pub OSRGetDataAxisToSRSAxisMapping: Option<
        unsafe extern "C" fn(hSRS: OGRSpatialReferenceH, pnCount: *mut c_int) -> *const c_int,
    >,
    pub OSRGetAxisMappingStrategy:
        Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH) -> OSRAxisMappingStrategy>,
    pub OSRSetAxisMappingStrategy:
        Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH, strategy: OSRAxisMappingStrategy)>,
    pub OSRDestroySpatialReference: Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH)>,
    pub OSRExportToPROJJSON: Option<
        unsafe extern "C" fn(
            hSRS: OGRSpatialReferenceH,
            ppszResult: *mut *mut c_char,
            papszOptions: *const *const c_char,
        ) -> OGRErr,
    >,
    pub OSRClone: Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH) -> OGRSpatialReferenceH>,
    pub OSRRelease: Option<unsafe extern "C" fn(hSRS: OGRSpatialReferenceH)>,

    // --- Geometry ---
    pub OGR_G_CreateFromWkb: Option<
        unsafe extern "C" fn(
            pabyData: *const c_void,
            hSRS: OGRSpatialReferenceH,
            phGeometry: *mut OGRGeometryH,
            nBytes: c_int,
        ) -> OGRErr,
    >,
    pub OGR_G_CreateFromWkt: Option<
        unsafe extern "C" fn(
            ppszData: *mut *mut c_char,
            hSRS: OGRSpatialReferenceH,
            phGeometry: *mut OGRGeometryH,
        ) -> OGRErr,
    >,
    pub OGR_G_ExportToIsoWkb: Option<
        unsafe extern "C" fn(
            hGeom: OGRGeometryH,
            eOrder: OGRwkbByteOrder,
            pabyData: *mut c_uchar,
        ) -> OGRErr,
    >,
    pub OGR_G_WkbSize: Option<unsafe extern "C" fn(hGeom: OGRGeometryH) -> c_int>,
    pub OGR_G_GetEnvelope:
        Option<unsafe extern "C" fn(hGeom: OGRGeometryH, psEnvelope: *mut OGREnvelope)>,
    pub OGR_G_DestroyGeometry: Option<unsafe extern "C" fn(hGeom: OGRGeometryH)>,

    // --- Vector / Layer ---
    pub OGR_L_ResetReading: Option<unsafe extern "C" fn(hLayer: OGRLayerH)>,
    pub OGR_L_GetNextFeature: Option<unsafe extern "C" fn(hLayer: OGRLayerH) -> OGRFeatureH>,
    pub OGR_L_CreateField: Option<
        unsafe extern "C" fn(
            hLayer: OGRLayerH,
            hFieldDefn: OGRFieldDefnH,
            bApproxOK: c_int,
        ) -> OGRErr,
    >,
    pub OGR_L_GetFeatureCount:
        Option<unsafe extern "C" fn(hLayer: OGRLayerH, bForce: c_int) -> i64>,
    pub OGR_F_GetGeometryRef: Option<unsafe extern "C" fn(hFeat: OGRFeatureH) -> OGRGeometryH>,
    pub OGR_F_GetFieldIndex:
        Option<unsafe extern "C" fn(hFeat: OGRFeatureH, pszName: *const c_char) -> c_int>,
    pub OGR_F_GetFieldAsDouble:
        Option<unsafe extern "C" fn(hFeat: OGRFeatureH, iField: c_int) -> c_double>,
    pub OGR_F_GetFieldAsInteger:
        Option<unsafe extern "C" fn(hFeat: OGRFeatureH, iField: c_int) -> c_int>,
    pub OGR_F_IsFieldSetAndNotNull:
        Option<unsafe extern "C" fn(hFeat: OGRFeatureH, iField: c_int) -> c_int>,
    pub OGR_F_Destroy: Option<unsafe extern "C" fn(hFeat: OGRFeatureH)>,
    pub OGR_Fld_Create:
        Option<unsafe extern "C" fn(pszName: *const c_char, eType: OGRFieldType) -> OGRFieldDefnH>,
    pub OGR_Fld_Destroy: Option<unsafe extern "C" fn(hDefn: OGRFieldDefnH)>,

    // --- VSI ---
    pub VSIFileFromMemBuffer: Option<
        unsafe extern "C" fn(
            pszFilename: *const c_char,
            pabyData: *mut c_uchar,
            nDataLength: i64,
            bTakeOwnership: c_int,
        ) -> VSILFILE,
    >,
    pub VSIFCloseL: Option<unsafe extern "C" fn(fp: VSILFILE) -> c_int>,
    pub VSIUnlink: Option<unsafe extern "C" fn(pszFilename: *const c_char) -> c_int>,
    pub VSIGetMemFileBuffer: Option<
        unsafe extern "C" fn(
            pszFilename: *const c_char,
            pnDataLength: *mut i64,
            bUnlinkAndSeize: c_int,
        ) -> *mut c_uchar,
    >,
    pub VSIFree: Option<unsafe extern "C" fn(pData: *mut c_void)>,
    pub VSIMalloc: Option<unsafe extern "C" fn(nSize: usize) -> *mut c_void>,

    // --- VRT ---
    pub VRTCreate: Option<unsafe extern "C" fn(nXSize: c_int, nYSize: c_int) -> GDALDatasetH>,
    pub VRTAddSimpleSource: Option<
        unsafe extern "C" fn(
            hVRTBand: GDALRasterBandH,
            hSrcBand: GDALRasterBandH,
            nSrcXOff: c_int,
            nSrcYOff: c_int,
            nSrcXSize: c_int,
            nSrcYSize: c_int,
            nDstXOff: c_int,
            nDstYOff: c_int,
            nDstXSize: c_int,
            nDstYSize: c_int,
            pszResampling: *const c_char,
            dfNoDataValue: c_double,
        ) -> CPLErr,
    >,

    // --- Rasterize / Polygonize ---
    pub GDALRasterizeGeometries: Option<
        unsafe extern "C" fn(
            hDS: GDALDatasetH,
            nBandCount: c_int,
            panBandList: *const c_int,
            nGeomCount: c_int,
            pahGeometries: *const OGRGeometryH,
            pfnTransformer: *mut c_void,
            pTransformArg: *mut c_void,
            padfGeomBurnValues: *const c_double,
            papszOptions: *mut *mut c_char,
            pfnProgress: *mut c_void,
            pProgressData: *mut c_void,
        ) -> CPLErr,
    >,
    pub GDALFPolygonize: Option<
        unsafe extern "C" fn(
            hSrcBand: GDALRasterBandH,
            hMaskBand: GDALRasterBandH,
            hOutLayer: OGRLayerH,
            iPixValField: c_int,
            papszOptions: *mut *mut c_char,
            pfnProgress: *mut c_void,
            pProgressData: *mut c_void,
        ) -> CPLErr,
    >,
    pub GDALPolygonize: Option<
        unsafe extern "C" fn(
            hSrcBand: GDALRasterBandH,
            hMaskBand: GDALRasterBandH,
            hOutLayer: OGRLayerH,
            iPixValField: c_int,
            papszOptions: *mut *mut c_char,
            pfnProgress: *mut c_void,
            pProgressData: *mut c_void,
        ) -> CPLErr,
    >,

    // --- Version ---
    pub GDALVersionInfo: Option<unsafe extern "C" fn(pszRequest: *const c_char) -> *const c_char>,

    // --- Config ---
    pub CPLSetThreadLocalConfigOption:
        Option<unsafe extern "C" fn(pszKey: *const c_char, pszValue: *const c_char)>,

    // --- Error ---
    pub CPLGetLastErrorNo: Option<unsafe extern "C" fn() -> c_int>,
    pub CPLGetLastErrorMsg: Option<unsafe extern "C" fn() -> *const c_char>,
    pub CPLErrorReset: Option<unsafe extern "C" fn()>,

    // --- Data Type ---
    pub GDALGetDataTypeSizeBytes: Option<unsafe extern "C" fn(eDataType: GDALDataType) -> c_int>,

    // --- C++ API ---
    pub MEMDatasetCreate: Option<
        unsafe extern "C" fn(
            pszFilename: *const c_char,
            nXSize: c_int,
            nYSize: c_int,
            nBandsIn: c_int,
            eType: GDALDataType,
            papszOptions: *mut *mut c_char,
        ) -> GDALDatasetH,
    >,
}
