// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Shared runtime discovery for native GDAL and PROJ installations.
//!
//! Language bindings may add package-specific discovery (for example, Python
//! wheels bundled by rasterio or pyproj), but prefix-based installations such
//! as Conda and Homebrew should resolve identically in every binding.

use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeLibrary {
    Gdal,
    Proj,
}

impl Display for NativeLibrary {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gdal => f.write_str("GDAL"),
            Self::Proj => f.write_str("PROJ"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeLibraryInstallation {
    pub shared_library: PathBuf,
    pub data_dir: PathBuf,
}

impl NativeLibrary {
    pub fn system_library_name(self) -> &'static str {
        self.system_library_names()[0]
    }

    pub fn system_library_names(self) -> &'static [&'static str] {
        match (self, std::env::consts::OS) {
            (Self::Gdal, "windows") => &["gdal.dll"],
            (Self::Gdal, "macos") => &["libgdal.dylib"],
            (Self::Gdal, _) => &["libgdal.so"],
            (Self::Proj, "windows") => &["proj.dll", "proj_9.dll"],
            (Self::Proj, "macos") => &["libproj.dylib"],
            (Self::Proj, _) => &["libproj.so"],
        }
    }

    /// Return loadable names and discovered files in platform library paths.
    ///
    /// The directory scan matters on Linux distributions that install only a
    /// versioned runtime library (for example `libgdal.so.34`) unless the
    /// separate development package is installed.
    pub fn system_candidates(self) -> Vec<PathBuf> {
        let mut candidates = self
            .system_library_names()
            .iter()
            .map(PathBuf::from)
            .collect::<Vec<_>>();

        let search_var = match std::env::consts::OS {
            "windows" => "PATH",
            "macos" => "DYLD_LIBRARY_PATH",
            _ => "LD_LIBRARY_PATH",
        };
        let mut search_dirs = std::env::var_os(search_var)
            .map(|paths| std::env::split_paths(&paths).collect::<Vec<_>>())
            .unwrap_or_default();

        if cfg!(target_os = "linux") {
            search_dirs.extend([
                PathBuf::from("/usr/local/lib"),
                PathBuf::from("/usr/lib"),
                PathBuf::from("/lib"),
                PathBuf::from(format!("/usr/lib/{}-linux-gnu", std::env::consts::ARCH)),
                PathBuf::from(format!("/lib/{}-linux-gnu", std::env::consts::ARCH)),
            ]);
        } else if cfg!(target_os = "macos") {
            search_dirs.extend([
                PathBuf::from("/opt/homebrew/lib"),
                PathBuf::from("/usr/local/lib"),
            ]);
        }

        for dir in search_dirs {
            if let Ok(path) = self.find_shared_library(&dir) {
                if !candidates.contains(&path) {
                    candidates.push(path);
                }
            }
        }

        candidates
    }

    fn glob_parts(self) -> (&'static str, &'static str) {
        match (self, std::env::consts::OS) {
            (Self::Gdal, "windows") => ("gdal", ".dll"),
            (Self::Gdal, "macos") => ("libgdal", ".dylib"),
            (Self::Gdal, _) => ("libgdal.so", ""),
            (Self::Proj, "windows") => ("proj", ".dll"),
            (Self::Proj, "macos") => ("libproj", ".dylib"),
            (Self::Proj, _) => ("libproj.so", ""),
        }
    }

    pub fn from_prefix(self, prefix: &Path) -> Result<NativeLibraryInstallation, String> {
        if prefix.as_os_str().is_empty() || !prefix.is_dir() {
            return Err(format!(
                "Can't configure {self} from prefix '{}': does not exist",
                prefix.display()
            ));
        }

        let (lib_dir, share_dir) = if cfg!(windows) {
            (prefix.join("Library/bin"), prefix.join("Library/share"))
        } else {
            (prefix.join("lib"), prefix.join("share"))
        };

        let data_dir = share_dir.join(match self {
            Self::Gdal => "gdal",
            Self::Proj => "proj",
        });
        if !data_dir.is_dir() {
            return Err(format!(
                "Can't find {self} data directory '{}'",
                data_dir.display()
            ));
        }

        Ok(NativeLibraryInstallation {
            shared_library: self.find_shared_library(&lib_dir)?,
            data_dir,
        })
    }

    pub fn find_shared_library(self, lib_dir: &Path) -> Result<PathBuf, String> {
        let (prefix, suffix) = self.glob_parts();
        let entries = std::fs::read_dir(lib_dir).map_err(|e| {
            format!(
                "Can't read {self} library directory '{}': {e}",
                lib_dir.display()
            )
        })?;

        let mut matches = entries
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                    return false;
                };
                name.starts_with(prefix)
                    && (suffix.is_empty() || name.ends_with(suffix))
                    && !name.contains("_d.")
            })
            .collect::<Vec<_>>();

        matches.sort_by_key(|path| {
            path.file_name()
                .map(|name| name.len())
                .unwrap_or(usize::MAX)
        });
        matches.into_iter().next().ok_or_else(|| {
            format!(
                "Can't find {self} shared library in '{}'",
                lib_dir.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_names_match_platform() {
        let gdal = NativeLibrary::Gdal.system_library_name();
        let proj = NativeLibrary::Proj.system_library_name();
        assert!(gdal.to_ascii_lowercase().contains("gdal"));
        assert!(proj.to_ascii_lowercase().contains("proj"));
        assert!(!NativeLibrary::Gdal.system_candidates().is_empty());
    }

    #[test]
    fn missing_prefix_has_useful_error() {
        let err = NativeLibrary::Proj
            .from_prefix(Path::new("/path/that/does/not/exist"))
            .unwrap_err();
        assert!(err.contains("PROJ"));
        assert!(err.contains("does not exist"));
    }
}
