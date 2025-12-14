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
import numpy as np  # Used for efficient random sampling indices
import pyarrow.parquet as pq  # Used for high-speed Arrow-native I/O
import argparse  # New import for handling command-line arguments


def get_geom_types(input_path: str):
    """Reads a GeoParquet file with GeoPandas and prints the geometry types."""
    try:
        # Use GeoPandas to correctly interpret GeoParquet metadata and geometry
        print("\n--- Geometry Type Analysis ---")
        print("Reading file with GeoPandas to determine geometry types...")
        gdf = gpd.read_parquet(input_path)

        if gdf.empty:
            print("   -> File is empty, no geometry types found.")
            return

        # Get unique geometry types from the geometry column, dropping NaN values
        unique_types = gdf.geometry.geom_type.dropna().unique()

        if len(unique_types) > 0:
            print(f"Distinct Geometry Types Found: {', '.join(unique_types)}")
        else:
            print(
                "No valid geometry types found in the GeoParquet file (column might be empty or missing)."
            )
        print("----------------------------")

    except FileNotFoundError:
        print(
            f"Error: Input file not found at {input_path}. Cannot analyze geometry types."
        )
    except Exception as e:
        print(f"An error occurred while analyzing geometry types: {e}")


def sample_geoparquet_arrow(input_path: str, output_path: str, fraction: float):
    """
    Reads a GeoParquet file using PyArrow, samples a specified fraction of the data
    by sampling indices with NumPy, and writes the result back using PyArrow.

    This is the Arrow-native and highly efficient way to sample Parquet files.

    Args:
        input_path (str): Path to the input GeoParquet file.
        output_path (str): Path where the sampled GeoParquet file will be saved.
        fraction (float): The fraction (0.0 to 1.0) of data to sample.
    """
    try:
        # 1. Read the GeoParquet file as a PyArrow Table
        print(f"1. Reading GeoParquet file from: {input_path} using PyArrow.")
        table = pq.read_table(input_path)

        total_rows = len(table)
        print(f"   -> Total rows read: {total_rows}")

        if total_rows == 0:
            print("   -> Input file is empty. Skipping sampling.")
            return

        # 2. Sample indices using NumPy
        print(f"2. Sampling {fraction * 100:.1f}% of the data using NumPy indices...")

        num_to_sample = int(total_rows * fraction)

        # Create a list of all row indices
        all_indices = np.arange(total_rows)

        # --- FIX: Use default_rng to correctly set the seed for reproducibility ---
        # Initialize a random number generator for reproducibility
        rng = np.random.default_rng(42)

        # Randomly choose indices to keep (without replacement)
        sampled_indices = rng.choice(all_indices, size=num_to_sample, replace=False)
        # --- END FIX ---

        # Use PyArrow's .take() to select the rows (fast, zero-copy operation)
        sampled_table = table.take(sampled_indices)

        sampled_rows = len(sampled_table)
        print(f"   -> Sampled rows generated: {sampled_rows}")

        # 3. Write the sampled PyArrow Table to a new Parquet file
        print(f"3. Writing sampled data to: {output_path} using PyArrow.")

        pq.write_table(sampled_table, output_path, row_group_size=10000, version="2.6")
        print(f"   -> Sampling complete. New file saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}. Please check the path.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


def main():
    """Main function to parse arguments and run the sampling process."""
    parser = argparse.ArgumentParser(
        description="Sample a GeoParquet file using high-performance PyArrow/NumPy."
    )

    # Required arguments for input and output files
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to the input GeoParquet file (e.g., data.parquet).",
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save the output sampled GeoParquet file (e.g., sampled_data.parquet).",
    )

    # Optional argument for the sampling fraction
    parser.add_argument(
        "-f",
        "--fraction",
        type=float,
        default=0.10,
        help="Fraction of data to sample (default: 0.10 for 10%%).",
    )

    args = parser.parse_args()

    # Validate fraction range
    if not (0.0 < args.fraction <= 1.0):
        print(f"Error: Fraction must be between 0.0 and 1.0. Got {args.fraction}.")
        return
    get_geom_types(args.input_path)

    # Run the core sampling logic
    sample_geoparquet_arrow(args.input_path, args.output_path, args.fraction)
    get_geom_types(args.output_path)


if __name__ == "__main__":
    main()
