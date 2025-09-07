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

use std::env;
use std::path::Path;
use std::process::ExitCode;

use datafusion::error::{DataFusionError, Result};
use sedona::context::SedonaContext;
use sedona_cli::{
    exec,
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

    // Make Scarf call for usage analytics
    sedona::scarf::make_scarf_call("cli");

    if let Some(ref path) = args.data_path {
        let p = Path::new(path);
        env::set_current_dir(p).unwrap();
    };

    let ctx = SedonaContext::new_local_interactive().await?;

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
