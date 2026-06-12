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

# Build Windows wheels for sedonadb-zarr: pure Rust + pyo3, no vcpkg. NASM is
# required for aws-lc-sys assembly; cmake (preinstalled) builds c-blosc +
# aws-lc-sys.
#
# Local usage:
# $env:CIBW_BUILD="cp313-win_amd64"; .\wheels-build-zarr-windows.ps1

$originalDirectory = Get-Location
$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path

# Fetch NASM (required for aws-lc-sys assembly) if not already present.
$NASM_URL = "https://www.nasm.us/pub/nasm/releasebuilds/2.16.03/win64/nasm-2.16.03-win64.zip"
$NASM_DIR = "$scriptDirectory\nasm-2.16.03"
$NASM_ZIP = "$scriptDirectory\nasm.zip"

if (-not (Test-Path $NASM_DIR)) {
	New-Item -Path $NASM_DIR -ItemType Directory -Force | Out-Null
	Invoke-WebRequest -Uri $NASM_URL -OutFile $NASM_ZIP
	Expand-Archive -Path $NASM_ZIP -DestinationPath $scriptDirectory -Force
	Remove-Item -Path $NASM_ZIP -Force
}

# Add NASM to PATH
$env:PATH += ";$NASM_DIR"

# Vendor any runtime DLLs into the wheel (typically a no-op; Rust links static).
$env:CIBW_REPAIR_WHEEL_COMMAND_WINDOWS="delvewheel repair -v --exclude=combase.dll --wheel-dir={dest_dir} {wheel}"

# Quality of life: don't change the working directory of the calling script even when it fails
$parentDirectory = Split-Path -Parent (Split-Path -Parent $scriptDirectory)
try {
	Push-Location "$parentDirectory"
	python -m cibuildwheel --output-dir python\sedonadb-zarr\dist python\sedonadb-zarr
	Pop-Location
}
finally {
	# Restore the original working directory
	Set-Location -Path $originalDirectory
}
