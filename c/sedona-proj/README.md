# sedona-proj

## How to update the prebuilt bindings

If `SEDONA_PROJ_BINDINGS_OUTPUT_PATH` envvar is set, the new bindings will be
written to the specified path. The path is relative to the manifest dir of
this crate.

```sh
SEDONA_PROJ_BINDINGS_OUTPUT_PATH=src/bindings/x86_64-unknown-linux-gnu.rs cargo build -p sedona-proj
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

$env:SEDONA_PROJ_BINDINGS_OUTPUT_PATH = "src/bindings/x86_64-pc-windows-gnu.rs"

cargo build -p sedona-proj
```
