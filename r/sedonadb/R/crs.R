#' Parse CRS from GeoArrow metadata
#'
#' @param crs_json A JSON string representing the CRS (PROJJSON or authority code)
#' @returns A list with components: authority_code (e.g., "EPSG:5070"), srid (integer),
#' @keywords internal
sd_parse_crs <- function(crs_json) {
  parse_crs_metadata(crs_json)
}
