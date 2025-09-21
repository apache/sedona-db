<!---
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# sedona-tg

## How to update the prebuilt bindings

If `SEDONA_TG_BINDINGS_OUTPUT_PATH` envvar is set, the new bindings will be
written to the specified path. The path is relative to the manifest dir of
this crate.

```sh
SEDONA_TG_BINDINGS_OUTPUT_PATH=src/bindings/x86_64-unknown-linux-gnu.rs cargo build -p sedona-tg
```

Note that, the generated file doesn't contain the license header, so you need
to add it manually (hopefully, this will be automated).

### Windows

On Windows, you might need to set `PATH` and `PKG_CONFIG_SYSROOT_DIR` for
`pkg-config`. For example, when using Rtools45, you progbably need these
envvars:

```ps1
$env:PATH = "C:\rtools45\x86_64-w64-mingw32.static.posix\bin;$env:PATH"
$env:PKG_CONFIG_SYSROOT_DIR = "C:\rtools45\x86_64-w64-mingw32.static.posix\"

$env:SEDONA_TG_BINDINGS_OUTPUT_PATH = "src/bindings/x86_64-pc-windows-gnu.rs"

cargo build -p sedona-tg
```
