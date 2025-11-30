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
import sys
import os
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point


def calculate_bbox_and_generate_points(geoparquet_path, n_points, output_path):
    # 1. Read the GeoParquet file to get geometry and CRS information
    print(f"Reading source GeoParquet file: {geoparquet_path}")
    try:
        # We read the file to get the bounding box and, critically, the CRS.
        gdf_source = gpd.read_parquet(geoparquet_path)
    except Exception as e:
        print(f"Error reading GeoParquet file: {e}")
        return

    # 2. Calculate the Bounding Box
    minx, miny, maxx, maxy = gdf_source.total_bounds

    print("\nCalculated Bounding Box:")
    print(f"  Min X: {minx}")
    print(f"  Min Y: {miny}")
    print(f"  Max X: {maxx}")
    print(f"  Max Y: {maxy}")
    print(f"  Source CRS: {gdf_source.crs}")

    # 3. Generate n random points within the Bounding Box
    print(f"\nGenerating {n_points} random points...")

    # Generate random coordinates
    random_x = np.random.uniform(minx, maxx, n_points)
    random_y = np.random.uniform(miny, miny, n_points)

    # 4. Create a GeoDataFrame from the points

    # Create Shapely Point objects from the coordinates
    # We use a list comprehension for efficiency
    geometries = [Point(x, y) for x, y in zip(random_x, random_y)]

    # Create a Pandas DataFrame for other attributes (if any)
    data = pd.DataFrame(
        {"id": np.arange(n_points), "original_x": random_x, "original_y": random_y}
    )

    # Create the GeoDataFrame, assigning the geometries and the CRS
    # from the source file to ensure spatial correctness.
    gdf_points = gpd.GeoDataFrame(
        data,
        geometry=geometries,
        crs=gdf_source.crs,  # IMPORTANT: Use the source CRS
    )

    print(f"Successfully created GeoDataFrame with {n_points} points.")

    # 5. Save the new GeoDataFrame to a GeoParquet file
    print(f"\nSaving points to GeoParquet file: {output_path}")

    try:
        # GeoPandas handles the GeoParquet writing, using pyarrow internally
        gdf_points.to_parquet(output_path, engine="pyarrow")
        print("Save complete.")
        print(f"First 5 rows of the saved GeoParquet:\n{gdf_points.head()}")
    except Exception as e:
        print(f"Error saving GeoParquet file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        # Prints usage instruction for command line
        print(
            "Usage: python geotools_save.py <input_geoparquet_path> <number_of_points> <output_geoparquet_path>"
        )
        print(
            "Example: python geotools_save.py input.geoparquet 1000 generated_points.parquet"
        )
        sys.exit(1)

    # Get arguments from command line
    input_path = sys.argv[1]

    try:
        num_points = int(sys.argv[2])
    except ValueError:
        print("Error: The number of points must be an integer.")
        sys.exit(1)

    output_path = sys.argv[3]

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at path: {input_path}")
        sys.exit(1)

    calculate_bbox_and_generate_points(input_path, num_points, output_path)
