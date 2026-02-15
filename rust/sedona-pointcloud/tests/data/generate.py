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

from pathlib import Path
import laspy
import numpy as np


# Some links for reference
#
# * ASPRS: <https://www.asprs.org/>
# * ASPRS (GitHub): <https://github.com/ASPRSorg/LAS>
# * OGC LAS Specification Standard: <https://www.ogc.org/standards/las/>
# * LAZ Specification 1.4: <https://downloads.rapidlasso.de/doc/LAZ_Specification_1.4_R1.pdf>


DATA_DIR = Path(__file__).resolve().parent


LAS_VERSIONS = [f"1.{p}" for p in range(5)]  # 1.0 - 1.4
POINT_FORMAT = list(range(11))  # 0 - 10 (>= 6 for LAS 1.4+)

# Pragmatic choice
version = LAS_VERSIONS[4]
point_format = POINT_FORMAT[6]

# Header
header = laspy.LasHeader(point_format=point_format, version=version)
header.offsets = np.array([1.0, 1.0, 1.0])
header.scales = np.array([0.1, 0.1, 0.1])


# -----------------------------------------------------------------------------
# Extra attribute test file with a single point (extra.laz)
# -----------------------------------------------------------------------------
DATA_TYPES = [
    "uint8",
    "int8",
    "uint16",
    "int16",
    "uint32",
    "int32",
    "uint64",
    "int64",
    "float32",
    "float64",
]

# Extra attributes
for dt in DATA_TYPES:
    name = f"{dt}_plain"
    header.add_extra_dim(laspy.point.format.ExtraBytesParams(name, dt, "", None, None))

    name = f"{dt}_scaled"
    header.add_extra_dim(
        laspy.point.format.ExtraBytesParams(name, dt, "", [10.0], [0.1])
    )

    name = f"{dt}_nodata"
    header.add_extra_dim(
        laspy.point.format.ExtraBytesParams(name, dt, "", None, None, [42])
    )

# Write laz with one point
with laspy.open(
    DATA_DIR.joinpath("extra.laz"), mode="w", header=header, do_compress=True
) as writer:
    point_record = laspy.ScaleAwarePointRecord.zeros(point_count=1, header=header)
    point_record.x = [0.5]
    point_record.y = [0.5]
    point_record.z = [0.5]

    for dt in DATA_TYPES:
        name = f"{dt}_plain"
        point_record[name] = [21]

        name = f"{dt}_scaled"
        point_record[name] = [21]

        name = f"{dt}_nodata"
        point_record[name] = [42]

    writer.write_points(point_record)


# -----------------------------------------------------------------------------
# Large test file to evaluate pruning (large.laz)
# -----------------------------------------------------------------------------
with laspy.open(
    DATA_DIR.joinpath("large.laz"), mode="w", header=header, do_compress=True
) as writer:
    N = 100000

    point_record = laspy.ScaleAwarePointRecord.zeros(point_count=N, header=header)

    # create two distinct chunks
    point_record.x = [0.5] * int(N / 2) + [1] * int(N / 2)
    point_record.y = [0.5] * int(N / 2) + [1] * int(N / 2)
    point_record.z = [0.5] * int(N / 2) + [1] * int(N / 2)

    writer.write_points(point_record)
