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

/// Make a call to Scarf for usage analytics.
///
/// # Arguments
///
/// * `language` - The language identifier (e.g., "rust", "cli")
pub fn make_scarf_call(language: &str) {
    let language = language.to_string();
    std::thread::spawn(move || {
        let _ = scarf_request(&language);
    });
}

fn scarf_request(language: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Check for user opt-out
    if std::env::var("SCARF_NO_ANALYTICS").is_ok() || std::env::var("DO_NOT_TRACK").is_ok() {
        return Ok(());
    }

    // Detect architecture and OS
    let arch = std::env::consts::ARCH.to_lowercase().replace(' ', "_");
    let os = std::env::consts::OS.to_lowercase().replace(' ', "_");

    // Construct Scarf URL
    let scarf_url = format!("https://sedona.gateway.scarf.sh/sedona-db/{arch}/{os}/{language}");

    // Make the request using std::net::TcpStream for a simple HTTP GET
    if let Ok(url) = url::Url::parse(&scarf_url) {
        if let Some(host) = url.host_str() {
            let port = url.port().unwrap_or(443);
            let path = url.path();

            // Try to make a simple HTTP request
            if let Ok(addr) = format!("{host}:{port}").parse::<std::net::SocketAddr>() {
                if let Ok(mut stream) =
                    std::net::TcpStream::connect_timeout(&addr, std::time::Duration::from_secs(1))
                {
                    let request =
                        format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
                    let _ = std::io::Write::write_all(&mut stream, request.as_bytes());
                }
            }
        }
    }

    Ok(())
}
