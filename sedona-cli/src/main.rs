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

use std::collections::HashMap;
use std::env;
use std::num::NonZeroUsize;
use std::path::Path;
use std::process::ExitCode;
use std::sync::{Arc, LazyLock};

use datafusion::error::{DataFusionError, Result};
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool, TrackConsumersPool};
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use sedona::context::SedonaContext;
use sedona::memory_pool::{SedonaFairSpillPool, DEFAULT_UNSPILLABLE_RESERVE_RATIO};
use sedona_cli::{
    exec,
    pool_type::PoolType,
    print_format::PrintFormat,
    print_options::{MaxRows, PrintOptions},
    DATAFUSION_CLI_VERSION,
};

use clap::Parser;

#[cfg(feature = "mimalloc")]
use mimalloc::MiMalloc;

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, Parser, PartialEq)]
#[clap(author, version, about, long_about= None)]
struct Args {
    #[clap(
        short = 'p',
        long,
        help = "Path to your data, default to current directory",
        value_parser(parse_valid_data_dir)
    )]
    data_path: Option<String>,

    #[clap(
        short = 'c',
        long,
        num_args = 0..,
        help = "Execute the given command string(s), then exit. Commands are expected to be non empty.",
        value_parser(parse_command)
    )]
    command: Vec<String>,

    #[clap(
        short = 'm',
        long,
        help = "The memory pool limitation (e.g. '10g'), default to None (no limit)",
        value_parser(extract_memory_pool_size)
    )]
    memory_limit: Option<usize>,

    #[clap(
        long,
        help = "Specify the memory pool type 'greedy' or 'fair'",
        default_value_t = PoolType::Greedy
    )]
    mem_pool_type: PoolType,

    #[clap(
        long,
        help = "The fraction of memory reserved for unspillable consumers (0.0 - 1.0)",
        default_value_t = DEFAULT_UNSPILLABLE_RESERVE_RATIO
    )]
    unspillable_reserve_ratio: f64,

    #[clap(
        short,
        long,
        num_args = 0..,
        help = "Execute commands from file(s), then exit",
        value_parser(parse_valid_file)
    )]
    file: Vec<String>,

    #[clap(
        short = 'r',
        long,
        num_args = 0..,
        help = "Run the provided files on startup instead of ~/.datafusionrc",
        value_parser(parse_valid_file),
        conflicts_with = "file"
    )]
    rc: Option<Vec<String>>,

    #[clap(long, value_enum, default_value_t = PrintFormat::Automatic)]
    format: PrintFormat,

    #[clap(
        short,
        long,
        help = "Reduce printing other than the results and work quietly"
    )]
    quiet: bool,

    #[clap(
        long,
        help = "The max number of rows to display for 'Table' format\n[possible values: numbers(0/10/...), inf(no limit)]",
        default_value = "40"
    )]
    maxrows: MaxRows,

    #[clap(long, help = "Enables console syntax highlighting")]
    color: bool,
}

#[tokio::main]
/// Calls [`main_inner`], then handles printing errors and returning the correct exit code
pub async fn main() -> ExitCode {
    if let Err(e) = main_inner().await {
        println!("Error: {e}");
        return ExitCode::FAILURE;
    }

    ExitCode::SUCCESS
}

/// Main CLI entrypoint
async fn main_inner() -> Result<()> {
    env_logger::init();

    #[cfg(feature = "mimalloc")]
    {
        use libmimalloc_sys::{mi_free, mi_malloc, mi_realloc};
        use sedona_tg::tg::set_allocator;

        // Configure tg to use mimalloc
        unsafe { set_allocator(mi_malloc, mi_realloc, mi_free) }
            .expect("Failed to set tg allocator");
    }

    let args = Args::parse();

    if !args.quiet {
        println!("Sedona CLI v{DATAFUSION_CLI_VERSION}");
    }

    if let Some(ref path) = args.data_path {
        let p = Path::new(path);
        env::set_current_dir(p).unwrap();
    };

    let mut rt_builder = RuntimeEnvBuilder::new();
    // set memory pool size
    if let Some(memory_limit) = args.memory_limit {
        // set memory pool type
        let pool: Arc<dyn MemoryPool> = match args.mem_pool_type {
            PoolType::Fair => Arc::new(TrackConsumersPool::new(
                SedonaFairSpillPool::new(memory_limit, args.unspillable_reserve_ratio),
                NonZeroUsize::new(10).unwrap(),
            )),
            PoolType::Greedy => Arc::new(TrackConsumersPool::new(
                GreedyMemoryPool::new(memory_limit),
                NonZeroUsize::new(10).unwrap(),
            )),
        };

        rt_builder = rt_builder.with_memory_pool(pool)
    }
    let runtime_env = rt_builder.build_arc()?;

    let ctx = SedonaContext::new_local_interactive_with_runtime_env(runtime_env).await?;

    let mut print_options = PrintOptions {
        format: args.format,
        quiet: args.quiet,
        maxrows: args.maxrows,
        color: args.color,
        multi_line_rows: false,
        ascii: false,
    };

    let commands = args.command;
    let files = args.file;

    if commands.is_empty() && files.is_empty() {
        return exec::exec_from_repl(&ctx, &mut print_options)
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)));
    }

    if !files.is_empty() {
        exec::exec_from_files(&ctx, files, &print_options).await?;
    }

    if !commands.is_empty() {
        exec::exec_from_commands(&ctx, commands, &print_options).await?;
    }

    Ok(())
}

fn parse_valid_file(dir: &str) -> Result<String, String> {
    if Path::new(dir).is_file() {
        Ok(dir.to_string())
    } else {
        Err(format!("Invalid file '{dir}'"))
    }
}

fn parse_valid_data_dir(dir: &str) -> Result<String, String> {
    if Path::new(dir).is_dir() {
        Ok(dir.to_string())
    } else {
        Err(format!("Invalid data directory '{dir}'"))
    }
}

fn parse_command(command: &str) -> Result<String, String> {
    if !command.is_empty() {
        Ok(command.to_string())
    } else {
        Err("-c flag expects only non empty commands".to_string())
    }
}

#[derive(Debug, Clone, Copy)]
enum ByteUnit {
    Byte,
    KiB,
    MiB,
    GiB,
    TiB,
}

impl ByteUnit {
    fn multiplier(&self) -> u64 {
        match self {
            ByteUnit::Byte => 1,
            ByteUnit::KiB => 1 << 10,
            ByteUnit::MiB => 1 << 20,
            ByteUnit::GiB => 1 << 30,
            ByteUnit::TiB => 1 << 40,
        }
    }
}

fn parse_size_string(size: &str, label: &str) -> Result<usize, String> {
    static BYTE_SUFFIXES: LazyLock<HashMap<&'static str, ByteUnit>> = LazyLock::new(|| {
        let mut m = HashMap::new();
        m.insert("b", ByteUnit::Byte);
        m.insert("k", ByteUnit::KiB);
        m.insert("kb", ByteUnit::KiB);
        m.insert("m", ByteUnit::MiB);
        m.insert("mb", ByteUnit::MiB);
        m.insert("g", ByteUnit::GiB);
        m.insert("gb", ByteUnit::GiB);
        m.insert("t", ByteUnit::TiB);
        m.insert("tb", ByteUnit::TiB);
        m
    });

    static SUFFIX_REGEX: LazyLock<regex::Regex> =
        LazyLock::new(|| regex::Regex::new(r"^(-?[0-9]+)([a-z]+)?$").unwrap());

    let lower = size.to_lowercase();
    if let Some(caps) = SUFFIX_REGEX.captures(&lower) {
        let num_str = caps.get(1).unwrap().as_str();
        let num = num_str
            .parse::<usize>()
            .map_err(|_| format!("Invalid numeric value in {label} '{size}'"))?;

        let suffix = caps.get(2).map(|m| m.as_str()).unwrap_or("b");
        let unit = BYTE_SUFFIXES
            .get(suffix)
            .ok_or_else(|| format!("Invalid {label} '{size}'"))?;
        let total_bytes = usize::try_from(unit.multiplier())
            .ok()
            .and_then(|multiplier| num.checked_mul(multiplier))
            .ok_or_else(|| format!("{label} '{size}' is too large"))?;

        Ok(total_bytes)
    } else {
        Err(format!("Invalid {label} '{size}'"))
    }
}

pub fn extract_memory_pool_size(size: &str) -> Result<usize, String> {
    parse_size_string(size, "memory pool size")
}
