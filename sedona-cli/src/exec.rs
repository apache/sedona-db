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

//! Execution functions

use std::fs::File;
use std::io::{BufRead, BufReader};

use crate::helper::split_from_semicolon;
use crate::print_format::PrintFormat;
use crate::{
    command::{Command, OutputFormat},
    helper::CliHelper,
    print_options::{MaxRows, PrintOptions},
};
use futures::StreamExt;
use sedona::context::SedonaContext;

use datafusion::common::instant::Instant;
use datafusion::common::plan_err;
use datafusion::error::Result;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::execution_plan::EmissionType;
use datafusion::physical_plan::{execute_stream, ExecutionPlanProperties};

use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::physical_plan::spill::get_record_batch_memory_size;
use rustyline::error::ReadlineError;
use rustyline::Editor;
use tokio::signal;

/// run and execute SQL statements and commands, against a context with the given print options
pub async fn exec_from_commands(
    ctx: &SedonaContext,
    commands: Vec<String>,
    print_options: &PrintOptions,
) -> Result<()> {
    for sql in commands {
        exec_and_print(ctx, print_options, sql).await?;
    }

    Ok(())
}

/// run and execute SQL statements and commands in a file
pub async fn exec_from_files(
    ctx: &SedonaContext,
    files: Vec<String>,
    print_options: &PrintOptions,
) -> Result<()> {
    let files = files
        .into_iter()
        .map(|file_path| File::open(file_path).unwrap())
        .collect::<Vec<_>>();

    for file in files {
        let mut reader = BufReader::new(file);
        exec_from_lines(ctx, &mut reader, print_options).await?;
    }

    Ok(())
}

pub async fn exec_from_lines(
    ctx: &SedonaContext,
    reader: &mut BufReader<File>,
    print_options: &PrintOptions,
) -> Result<()> {
    let mut query = "".to_owned();

    for line in reader.lines() {
        match line {
            Ok(line) if line.starts_with("#!") => {
                continue;
            }
            Ok(line) if line.starts_with("--") => {
                continue;
            }
            Ok(line) => {
                let line = line.trim_end();
                query.push_str(line);
                if line.ends_with(';') {
                    match exec_and_print(ctx, print_options, query).await {
                        Ok(_) => {}
                        Err(err) => eprintln!("{err}"),
                    }
                    query = "".to_string();
                } else {
                    query.push('\n');
                }
            }
            _ => {
                break;
            }
        }
    }

    // run the left over query if the last statement doesn't contain ‘;’
    // ignore if it only consists of '\n'
    if query.contains(|c| c != '\n') {
        exec_and_print(ctx, print_options, query).await?;
    }

    Ok(())
}

/// run and execute SQL statements and commands against a context with the given print options
pub async fn exec_from_repl(
    ctx: &SedonaContext,
    print_options: &mut PrintOptions,
) -> rustyline::Result<()> {
    let mut rl = Editor::new()?;
    rl.set_helper(Some(CliHelper::new(
        ctx.ctx
            .task_ctx()
            .session_config()
            .options()
            .sql_parser
            .dialect,
        print_options.color,
    )));
    rl.load_history(".history").ok();

    loop {
        match rl.readline("> ") {
            Ok(line) if line.starts_with('\\') => {
                rl.add_history_entry(line.trim_end())?;
                let command = line.split_whitespace().collect::<Vec<_>>().join(" ");
                if let Ok(cmd) = &command[1..].parse::<Command>() {
                    match cmd {
                        Command::Quit => break,
                        Command::OutputFormat(subcommand) => {
                            if let Some(subcommand) = subcommand {
                                if let Ok(command) = subcommand.parse::<OutputFormat>() {
                                    if let Err(e) = command.execute(print_options).await {
                                        eprintln!("{e}")
                                    }
                                } else {
                                    eprintln!("'\\{}' is not a valid command", &line[1..]);
                                }
                            } else {
                                println!("Output format is {:?}.", print_options.format);
                            }
                        }
                        _ => {
                            if let Err(e) = cmd.execute(ctx, print_options).await {
                                eprintln!("{e}")
                            }
                        }
                    }
                } else {
                    eprintln!("'\\{}' is not a valid command", &line[1..]);
                }
            }
            Ok(line) => {
                let lines = split_from_semicolon(&line);
                for line in lines {
                    rl.add_history_entry(line.trim_end())?;
                    tokio::select! {
                        res = exec_and_print(ctx, print_options, line) => match res {
                            Ok(_) => {}
                            Err(err) => eprintln!("{err}"),
                        },
                        _ = signal::ctrl_c() => {
                            println!("^C");
                            continue
                        },
                    }
                    // dialect might have changed
                    rl.helper_mut().unwrap().set_dialect(
                        &ctx.ctx
                            .task_ctx()
                            .session_config()
                            .options()
                            .sql_parser
                            .dialect,
                    );
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\\q");
                break;
            }
            Err(err) => {
                eprintln!("Unknown error happened {err:?}");
                break;
            }
        }
    }

    rl.save_history(".history")
}

pub(super) async fn exec_and_print(
    ctx: &SedonaContext,
    print_options: &PrintOptions,
    sql: String,
) -> Result<()> {
    let now = Instant::now();

    let dfs = ctx.multi_sql(&sql).await?;

    for df in dfs {
        let adjusted = AdjustedPrintOptions::new(print_options.clone());
        let adjusted = adjusted.with_plan(df.logical_plan());
        let physical_plan = df.create_physical_plan().await?;

        // Track memory usage for the query result if it's bounded
        let task_ctx = ctx.ctx.task_ctx();
        let mut reservation =
            MemoryConsumer::new("DataFusion-Cli").register(task_ctx.memory_pool());

        if physical_plan.boundedness().is_unbounded() {
            if physical_plan.pipeline_behavior() == EmissionType::Final {
                return plan_err!(
                    "The given query can generate a valid result only once \
                    the source finishes, but the source is unbounded"
                );
            }
            // As the input stream comes, we can generate results.
            // However, memory safety is not guaranteed.
            let stream = execute_stream(physical_plan, task_ctx.clone())?;
            print_options.print_stream(stream, now, ctx).await?;
        } else {
            // Bounded stream; collected results size is limited by the maxrows option
            let schema = physical_plan.schema();
            let mut stream = execute_stream(physical_plan, task_ctx.clone())?;
            let mut results = vec![];
            let mut row_count = 0_usize;
            let max_rows = match print_options.maxrows {
                MaxRows::Unlimited => usize::MAX,
                MaxRows::Limited(n) => n,
            };
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let curr_num_rows = batch.num_rows();
                // Stop collecting results if the number of rows exceeds the limit
                // results batch should include the last batch that exceeds the limit
                if row_count < max_rows + curr_num_rows {
                    // Try to grow the reservation to accommodate the batch in memory
                    reservation.try_grow(get_record_batch_memory_size(&batch))?;
                    results.push(batch);
                }
                row_count += curr_num_rows;
            }
            adjusted
                .into_inner()
                .print_batches(schema, &results, now, row_count, ctx)?;
            reservation.free();
        }
    }

    Ok(())
}

/// Track adjustments to the print options based on the plan / statement being executed
#[derive(Debug)]
struct AdjustedPrintOptions {
    inner: PrintOptions,
}

impl AdjustedPrintOptions {
    fn new(inner: PrintOptions) -> Self {
        Self { inner }
    }

    /// Adjust print options based on any plan specific requirements
    fn with_plan(mut self, plan: &LogicalPlan) -> Self {
        // For plans like `Explain` ignore `MaxRows` option and always display
        // all rows
        if matches!(
            plan,
            LogicalPlan::Explain(_) | LogicalPlan::DescribeTable(_) | LogicalPlan::Analyze(_)
        ) {
            self.inner.maxrows = MaxRows::Unlimited;
            self.inner.multi_line_rows = true;
        }
        self
    }

    /// Finalize and return the inner `PrintOptions`
    fn into_inner(mut self) -> PrintOptions {
        if self.inner.format == PrintFormat::Automatic {
            self.inner.format = PrintFormat::Table;
        }

        self.inner
    }
}
