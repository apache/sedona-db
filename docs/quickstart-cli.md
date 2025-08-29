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

# CLI Quickstart

SedonaDB's command-line interface provides an interactive SQL shell that can be used to
leverage the SedonaDB engine for SQL-only/shell-centric workflows. SedonaDB's CLI is
based on the [DataFusion CLI](https://datafusion.apache.org/user-guide/cli/index.html),
whose documentation may be useful for advanced features not covered in detail here.

## Installation

You can install `sedona-cli` using Cargo:

```shell
cargo install sedona-cli
```

## Usage

Running `sedona-cli` from a terminal will start an interactive SQL shell. Queries must end
in a semicolon (`;`) and can be cleared with `Control-C`.

```
Sedona CLI v0.0.1
> SELECT ST_Point(0, 1) as geom;
┌────────────┐
│    geom    │
│     wkb    │
╞════════════╡
│ POINT(0 1) │
└────────────┘

1 row(s)/1 column(s) fetched.
Elapsed 0.024 seconds.
```

See the [SQL Reference]() for details on the SQL functions and features available to the CLI.

## Help

From the interactive shell, use `\?` for special command help:

```
> \?
Command,Description
\d,list tables
\d name,describe table
\q,quit datafusion-cli
\?,help
\h,function list
\h function,search function
\quiet (true|false)?,print or set quiet mode
\pset [NAME [VALUE]],"set table output option
(format)"
```

From the command line, use `--help` to list launch options and/or options for interacting
with the CLI in a non-interactive context.

```
Command Line Client for Sedona's DataFusion-based query engine.

Usage: sedona-cli [OPTIONS]

Options:
  -p, --data-path <DATA_PATH>   Path to your data, default to current directory
  -c, --command [<COMMAND>...]  Execute the given command string(s), then exit. Commands are expected to be non empty.
  -f, --file [<FILE>...]        Execute commands from file(s), then exit
  -r, --rc [<RC>...]            Run the provided files on startup instead of ~/.datafusionrc
      --format <FORMAT>         [default: automatic] [possible values: csv, tsv, table, json, nd-json, automatic]
  -q, --quiet                   Reduce printing other than the results and work quietly
      --maxrows <MAXROWS>       The max number of rows to display for 'Table' format
                                [possible values: numbers(0/10/...), inf(no limit)] [default: 40]
      --color                   Enables console syntax highlighting
  -h, --help                    Print help
  -V, --version                 Print version
```
