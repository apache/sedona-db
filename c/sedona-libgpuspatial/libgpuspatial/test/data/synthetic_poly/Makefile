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
GEN_URL = https://raw.githubusercontent.com/aseldawy/spider/refs/heads/master/html/cgi/generator.py
GEN_SCRIPT = /tmp/spider.py
CONV_SCRIPT = ../convert.py

# Common Generation Params
DIST = uniform
COUNT = 10000
SEED1 = 1
SEED2 = 2

.PHONY: all clean setup

# Build all parquet files
all: polygons1.parquet polygons2.parquet

setup:
	@python3 -c "import geopandas, pyarrow" || (echo "Missing: pip install geopandas pyarrow"; exit 1)
	@python3 -c "import cgi" || (pip install legacy-cgi)

$(GEN_SCRIPT):
	@echo "Downloading generator..."
	@curl -sLo $(GEN_SCRIPT) $(GEN_URL)

# --- Polygon 1 Generation & Conversion ---
polygons1.parquet: $(GEN_SCRIPT) setup
	@echo "Generating and converting polygons 1..."
	python3 $(GEN_SCRIPT) \
		distribution=$(DIST) \
		cardinality=$(COUNT) \
		dimensions=2 \
		seed=$(SEED1) \
		geometry=polygon \
		polysize=0.01 \
		maxseg=8 \
		format=geojson > polygons1.geojson
	python3 $(CONV_SCRIPT) --input polygons1.geojson --output polygons1.parquet --verbose
	rm polygons1.geojson

# --- Polygon 2 Generation & Conversion ---
polygons2.parquet: $(GEN_SCRIPT) setup
	@echo "Generating and converting polygons 2..."
	python3 $(GEN_SCRIPT) \
		distribution=$(DIST) \
		cardinality=$(COUNT) \
		dimensions=2 \
		seed=$(SEED2) \
		geometry=polygon \
		polysize=0.01 \
		maxseg=8 \
		format=geojson > polygons2.geojson
	python3 $(CONV_SCRIPT) --input polygons2.geojson --output polygons2.parquet --verbose
	rm polygons2.geojson

clean:
	rm -f *.geojson *.parquet
