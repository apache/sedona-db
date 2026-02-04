# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import geopandas as gpd
import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Convert GeoJSON spatial data to GeoParquet format."
    )

    parser.add_argument(
        "-i",
        "--input",
        default="polygons.geojson",
        help="Path to the input GeoJSON file (default: polygons.geojson)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="polygons.parquet",
        help="Path for the output Parquet file (default: polygons.parquet)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    try:
        if args.verbose:
            print(f"Loading {args.input}...")
        gdf = gpd.read_file(args.input)

        if args.verbose:
            print(f"Writing to {args.output}...")
        gdf.to_parquet(args.output, index=False)

        print(f"Successfully converted {len(gdf)} geometries to {args.output}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
