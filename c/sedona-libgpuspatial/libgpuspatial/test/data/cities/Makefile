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
# Variables
URL := https://raw.githubusercontent.com/geoarrow/geoarrow-data/v0.2.0/natural-earth/files/natural-earth_cities_geo.parquet
INPUT_FILE := natural-earth_cities_geo.parquet
PYTHON_SCRIPT := ../gen_points.py
OUTPUT_POINTS := generated_points.parquet
NUM_POINTS := 1000

.PHONY: all clean generate

# The default target runs both download and point generation
all: $(OUTPUT_POINTS)

# --- Download Target ---

# Target to download the GeoParquet file
$(INPUT_FILE):
	@echo "--- Downloading $(INPUT_FILE) ---"
	# Use curl to download the file. The -L flag handles redirects.
	curl -L $(URL) -o $(INPUT_FILE)
	@echo "--- Download complete ---"

# --- Generation Target ---

# Target to generate points, which depends on the input file being present
$(OUTPUT_POINTS): $(INPUT_FILE)
	@echo "--- Generating $(NUM_POINTS) random points from $(INPUT_FILE) ---"
	python $(PYTHON_SCRIPT) $(INPUT_FILE) $(NUM_POINTS) $(OUTPUT_POINTS)
	@echo "--- Point generation complete. Output: $(OUTPUT_POINTS) ---"

# An explicit target to run generation if you don't want to rely on 'all'
generate: $(OUTPUT_POINTS)

# --- Cleanup Target ---

# Target to remove all generated and downloaded files
clean:
	@echo "--- Cleaning up files ---"
	rm -f $(INPUT_FILE) $(OUTPUT_POINTS)
	@echo "--- Cleanup complete ---"
