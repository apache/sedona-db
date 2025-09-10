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
use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use comfy_table::presets::UTF8_FULL;
use comfy_table::{Cell, CellAlignment, ColumnConstraint, ContentArrangement, Row, Table, Width};
use datafusion::arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::error::Result;
use datafusion_common::format::DEFAULT_FORMAT_OPTIONS;
use datafusion_common::{DataFusionError, ScalarValue};
use datafusion_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF};
use sedona_expr::scalar_udf::SedonaScalarUDF;
use sedona_schema::{
    datatypes::{Edges, SedonaType},
    matchers::ArgMatcher,
};
use std::iter::zip;
use std::sync::Arc;

use crate::context::SedonaContext;

/// Print results to writable output
///
/// See [DisplayTableOptions] for ways to control the output. The context
/// is required to resolve the function that can handle extensions to avoid
/// hard-coding build-time support for every type that might be displayed.
pub fn show_batches<'a, W: std::io::Write>(
    ctx: &SedonaContext,
    writer: &mut W,
    schema: &Schema,
    batches: Vec<RecordBatch>,
    options: DisplayTableOptions<'a>,
) -> Result<()> {
    let format_fn = ctx
        .functions
        .scalar_udf("sd_format")
        .ok_or(DataFusionError::Internal(
            "sd_format UDF does not exist".to_string(),
        ))?
        .clone();

    let mut table = DisplayTable::try_new(schema, batches, options)?.with_format_fn(format_fn);
    table.negotiate_hidden_columns()?;
    table.write(writer)
}

/// Options for displaying tables
#[derive(Clone)]
pub struct DisplayTableOptions<'a> {
    /// Options to use for types that use Arrow's [ArrayFormatter]
    pub arrow_options: FormatOptions<'a>,

    /// Maximum width of the table in characters
    ///
    /// Defaults to 100 when used in a non-interactive context, or the width of the
    /// terminal otherwise.
    pub table_width: u16,

    /// The minimum number of characters to use for any particular column
    ///
    /// The actual column may use fewer characters if there is no value in a column
    /// that requires this width; however, this parameter prevents string or
    /// binary-like columns from shrinking to a uselessly small width. Defaults to 8.
    pub min_column_width: u16,

    /// The maximum number of rows in any given cell
    ///
    /// This defaults to one; however, may be set to more if printing values like
    /// DataFusion Analyze/Explain output that is designed to be displayed in this way.
    pub max_row_height: usize,

    /// Character display mode
    ///
    /// Defaults to [DisplayMode::ASCII] when used in a non-interactive context or
    /// [DisplayMode::Utf8] when used from a terminal.
    pub display_mode: DisplayMode,
}

impl DisplayTableOptions<'static> {
    /// Create new options for a non-interactive context
    pub fn new() -> Self {
        Self {
            arrow_options: DEFAULT_FORMAT_OPTIONS,
            table_width: 100,
            min_column_width: 8,
            max_row_height: 1,
            display_mode: DisplayMode::ASCII,
        }
    }

    /// Create new options for an interactive terminal context
    pub fn tty() -> Self {
        let defaults = Self::new();
        let terminal_width = Table::new().width().unwrap_or(100);
        Self {
            table_width: terminal_width,
            display_mode: DisplayMode::Utf8,
            ..defaults
        }
    }
}

/// Character display mode options
#[derive(Debug, Clone)]
pub enum DisplayMode {
    /// Only use ASCII characters for table display
    ///
    /// This option is similar to the existing DataFusion CLI and Spark output.
    ASCII,
    /// Use unicode characters for table display
    ///
    /// This will typically result in a much prettier and more compact table.
    Utf8,
}

impl Default for DisplayTableOptions<'static> {
    fn default() -> Self {
        Self::new()
    }
}

struct DisplayTable<'a> {
    columns: Vec<DisplayColumn>,
    num_rows: usize,
    options: DisplayTableOptions<'a>,
}

impl<'a> DisplayTable<'a> {
    pub fn try_new<'b: 'a>(
        schema: &Schema,
        batches: Vec<RecordBatch>,
        options: DisplayTableOptions<'a>,
    ) -> Result<Self> {
        let num_rows = batches.iter().map(|batch| batch.num_rows()).sum();

        let columns = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| {
                DisplayColumn::try_new(
                    f,
                    batches
                        .iter()
                        .map(|batch| batch.column(i).clone())
                        .collect(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            columns,
            num_rows,
            options,
        })
    }

    pub fn with_format_fn(self, format_fn: SedonaScalarUDF) -> Self {
        Self {
            columns: self
                .columns
                .into_iter()
                .map(|col| DisplayColumn {
                    format_fn: Some(format_fn.clone()),
                    ..col
                })
                .collect(),
            ..self
        }
    }

    /// Mark the smallest number of columns as possible as "hidden" such that
    /// the output width can be satisfied
    ///
    /// This heuristic checks the minimum width required to output the table
    /// and hides columns starting in the middle and continuing from the
    /// right and left of the hidden block (alternately) until there are
    /// no more columns left to hide or the satisfying the output width will
    /// be possible.
    ///
    /// The output width still might not be respected if the width of the last
    /// remaining column is wider than the allotted output.
    pub fn negotiate_hidden_columns(&mut self) -> Result<()> {
        let mut iterations = 0;
        while self.count_visible_columns() > 1 && self.minimum_width()? > self.options.table_width {
            // If there are no hidden columns, hide the middle one. Otherwise, alternate hiding
            // the right and left columns.
            if self.count_visible_columns() == self.columns.len() {
                let middle_index = self.columns.len() / 2;
                self.columns[middle_index].hidden = true;
            } else if iterations % 2 == 0 {
                let (i, _) = self
                    .columns
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, f)| f.hidden)
                    .unwrap();
                if let Some(next_column) = self.columns.get_mut(i + 1) {
                    next_column.hidden = true;
                } else {
                    // No more columns to hide!
                    return Ok(());
                }
            } else {
                let (i, _) = self
                    .columns
                    .iter()
                    .enumerate()
                    .find(|(_, f)| f.hidden)
                    .unwrap();
                if let Some(next_column) = self.columns.get_mut(i - 1) {
                    next_column.hidden = true;
                } else {
                    // No more columns to hide!
                    return Ok(());
                }
            }

            iterations += 1;
        }

        Ok(())
    }

    /// Build and write this table to the writer
    ///
    /// This does not hide any columns (use [Self::negotiate_hidden_columns] prior to calling
    /// this method to negotiate which columns need to be hidden to satisfy all constraints).
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> Result<()> {
        let mut table = Table::new();
        table.set_content_arrangement(ContentArrangement::Dynamic);
        table.set_width(self.options.table_width);

        match self.options.display_mode {
            DisplayMode::ASCII => {
                table.load_preset("||--+-++|    ++++++");
            }
            DisplayMode::Utf8 => {
                table.load_preset(UTF8_FULL);
            }
        }

        table.set_truncation_indicator(self.truncation_indicator());

        let mut columns = self
            .columns
            .iter()
            .filter(|col| !col.hidden)
            .cloned()
            .collect::<Vec<_>>();

        // If we've hidden any columns, insert a column of truncation indicators
        if let Some((i, _)) = self.columns.iter().enumerate().find(|(_, col)| col.hidden) {
            columns.insert(
                i,
                DisplayColumn::new_omitted_indicator(self.num_rows, self.truncation_indicator()),
            );
        }

        let mut header = Row::from(
            columns
                .iter()
                .map(|col| {
                    col.header(&self.options)
                        .set_alignment(CellAlignment::Center)
                })
                .collect::<Vec<_>>(),
        );
        header.max_height(1 + self.options.arrow_options.types_info() as usize);
        table.set_header(header);

        for (col, table_col) in zip(&columns, table.column_iter_mut()) {
            table_col.set_constraint(col.constraint(&self.options)?);
        }

        let formatted_columns = columns
            .iter()
            .map(|col| col.cells(&self.options))
            .collect::<Result<Vec<_>>>()?;
        for i in 0..self.num_rows {
            let mut row = Row::from(formatted_columns.iter().map(|col_vec| col_vec[i].clone()));
            row.max_height(self.options.max_row_height);
            table.add_row(row);
        }

        writeln!(writer, "{table}")?;
        Ok(())
    }

    /// Count the number of non-hidden columns
    fn count_visible_columns(&self) -> usize {
        self.columns.iter().map(|col| !col.hidden as usize).sum()
    }

    /// Calculate the smallest number of characters that would be required to display
    /// all (non-hidden) columns in this table. This applies the constraints that
    /// numeric columns must not be truncated and that we constrain the minimum
    /// number of characters in other columns (based on user options).
    fn minimum_width(&self) -> Result<u16> {
        // Leftmost border
        let mut total_size = 1;
        let mut hidden_count = 0;

        for col in &self.columns {
            // Padding on both sides plus separator
            total_size += 3;
            match col.constraint(&self.options)? {
                ColumnConstraint::Hidden => hidden_count += 1,
                ColumnConstraint::Absolute(Width::Fixed(width)) => total_size += width,
                ColumnConstraint::LowerBoundary(Width::Fixed(width)) => total_size += width,
                _ => {
                    total_size += 0;
                }
            }
        }

        if hidden_count > 0 {
            // Exactly one padding + separator + truncation indicator
            total_size += 3 + self
                .truncation_indicator()
                .len()
                .try_into()
                .unwrap_or(u16::MAX);
        }

        Ok(total_size)
    }

    /// Calculate the indicator used to indicate that characters from a value were hidden
    fn truncation_indicator(&self) -> &'static str {
        match self.options.display_mode {
            DisplayMode::ASCII => "...",
            DisplayMode::Utf8 => "…",
        }
    }
}

#[derive(Debug, Clone)]
struct DisplayColumn {
    name: String,
    sedona_type: SedonaType,
    raw_values: Vec<ArrayRef>,
    format_fn: Option<SedonaScalarUDF>,
    hidden: bool,
}

impl DisplayColumn {
    /// Create a new display column
    pub fn try_new(field: &Field, raw_values: Vec<ArrayRef>) -> Result<Self> {
        Ok(Self {
            name: field.name().to_string(),
            sedona_type: SedonaType::from_storage_field(field)?,
            raw_values,
            format_fn: None,
            hidden: false,
        })
    }

    /// Create a new column that only contains truncation indicators
    ///
    /// This is used to indicate that columns were hidden in the process of
    /// ensuring the output fit within the available width.
    pub fn new_omitted_indicator(size: usize, truncation_string: &str) -> Self {
        let raw_values = (0..size)
            .map(|_| Some(truncation_string.to_string()))
            .collect::<StringArray>();
        Self {
            name: truncation_string.to_string(),
            sedona_type: SedonaType::Arrow(DataType::Utf8),
            raw_values: vec![Arc::new(raw_values)],
            format_fn: None,
            hidden: false,
        }
    }

    /// Resolve the header cell
    ///
    /// Depending on the options, this will either be the name or the
    /// name and the type.
    pub fn header(&self, options: &DisplayTableOptions) -> Cell {
        // Don't print the type ever if it's a continuation column
        let is_continuation = self.name == "…" || self.name == "...";
        let display_type = match &self.sedona_type {
            SedonaType::Wkb(Edges::Planar, _) | SedonaType::WkbView(Edges::Planar, _) => {
                "geometry".to_string()
            }
            SedonaType::Wkb(Edges::Spherical, _) | SedonaType::WkbView(Edges::Spherical, _) => {
                "geography".to_string()
            }
            _ => self.sedona_type.to_string().to_lowercase(),
        };
        if options.arrow_options.types_info() && !is_continuation {
            Cell::new(format!("{}\n{}", self.name, display_type)).set_delimiter('\0')
        } else {
            Cell::new(self.name.clone()).set_delimiter('\0')
        }
    }

    /// Resolve the [ColumnConstraint] based on the data type and options
    ///
    /// This is where we ensure that numeric values are not truncated and that
    /// other columns do not shrink to fewer than the minimum column width set in
    /// the display options.
    pub fn constraint(&self, options: &DisplayTableOptions) -> Result<ColumnConstraint> {
        if self.hidden {
            return Ok(ColumnConstraint::Hidden);
        }

        let header_width = self.header_width(options);
        let content_width = self.content_width(options)?;
        let total_width = header_width.max(content_width);
        let padding_width = 2;

        let is_numeric = ArgMatcher::is_numeric();
        if is_numeric.match_type(&self.sedona_type) {
            Ok(ColumnConstraint::LowerBoundary(Width::Fixed(
                content_width + padding_width,
            )))
        } else {
            Ok(ColumnConstraint::LowerBoundary(Width::Fixed(
                options.min_column_width.min(total_width + padding_width),
            )))
        }
    }

    fn header_width(&self, options: &DisplayTableOptions) -> u16 {
        self.header(options)
            .content()
            .lines()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0)
            .try_into()
            .unwrap_or(u16::MAX)
    }

    fn content_width(&self, options: &DisplayTableOptions) -> Result<u16> {
        Ok(self
            .cells(options)?
            .iter()
            .map(|cell| cell.content().chars().count())
            .max()
            .unwrap_or(0)
            .try_into()
            .unwrap_or(u16::MAX))
    }

    /// Compute the actual cells for this column
    ///
    /// These cells may contain more output than is actually shown (i.e., the rendering
    /// engine might decide to omit some of the content if there is not enough space).
    pub fn cells(&self, options: &DisplayTableOptions) -> Result<Vec<Cell>> {
        let mut cells = Vec::new();

        for array in &self.raw_values {
            let format_proxy_array = self.format_proxy(array, options)?;
            let formatter = ArrayFormatter::try_new(&format_proxy_array, &options.arrow_options)?;
            for i in 0..format_proxy_array.len() {
                cells.push(self.cell(formatter.value(i)));
            }
        }

        Ok(cells)
    }

    /// Resolve the array that should be formatted. This is where we apply formatting to
    /// extension type columns like Geometry and Geography that would otherwise appear as
    /// their raw storage bytes.
    fn format_proxy(&self, array: &ArrayRef, options: &DisplayTableOptions) -> Result<ArrayRef> {
        if let Some(format) = &self.format_fn {
            let format_udf: ScalarUDF = format.clone().into();

            let options_scalar = ScalarValue::Utf8(Some(format!(
                r#"{{"width_hint": {}}}"#,
                (options.table_width as usize).saturating_mul(options.max_row_height)
            )));

            let arg_fields = vec![
                Arc::new(self.sedona_type.to_storage_field("", true)?),
                Arc::new(Field::new("", DataType::Utf8, true)),
            ];

            let args = ReturnFieldArgs {
                arg_fields: &arg_fields,
                scalar_arguments: &[None, None],
            };
            let return_field = format_udf.return_field_from_args(args)?;

            let args = ScalarFunctionArgs {
                args: vec![ColumnarValue::Array(array.clone()), options_scalar.into()],
                arg_fields,
                number_rows: array.len(),
                return_field,
            };

            let format_proxy_value = format_udf.invoke_with_args(args)?;
            format_proxy_value.to_array(array.len())
        } else {
            Ok(array.clone())
        }
    }

    /// Create a cell for this column. This is where alignment for numeric
    /// output is applied.
    fn cell<T: ToString>(&self, content: T) -> Cell {
        let cell = Cell::new(content).set_delimiter('\0');
        let is_numeric = ArgMatcher::is_numeric();
        if is_numeric.match_type(&self.sedona_type) {
            cell.set_alignment(CellAlignment::Right)
        } else {
            cell.set_alignment(CellAlignment::Left)
        }
    }
}

#[cfg(test)]
mod test {

    use rstest::rstest;
    use sedona_schema::datatypes::{
        WKB_GEOGRAPHY, WKB_GEOMETRY, WKB_VIEW_GEOGRAPHY, WKB_VIEW_GEOMETRY,
    };
    use sedona_testing::create::create_array;

    use super::*;

    fn test_cols() -> Vec<(&'static str, (SedonaType, ArrayRef))> {
        let short_chars: ArrayRef =
            arrow_array::create_array!(Utf8, [Some("abcd"), Some("efgh"), None]);
        let long_chars: ArrayRef = arrow_array::create_array!(
            Utf8,
            [
                Some("you see, what happened was"),
                Some("the elephant stomped so may times that"),
                None
            ]
        );

        let numeric: ArrayRef =
            arrow_array::create_array!(Int32, [Some(123456789), Some(987654321), None]);

        let geometry = create_array(
            &[
                Some("POINT (0 1)"),
                Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                None,
            ],
            &WKB_GEOMETRY,
        );

        vec![
            ("shrt", (SedonaType::Arrow(DataType::Utf8), short_chars)),
            ("long", (SedonaType::Arrow(DataType::Utf8), long_chars)),
            ("numeric", (SedonaType::Arrow(DataType::Int32), numeric)),
            ("geometry", (WKB_GEOMETRY, geometry)),
        ]
    }

    fn render_cols<'a>(
        cols: Vec<(&'static str, (SedonaType, ArrayRef))>,
        options: DisplayTableOptions<'a>,
    ) -> Vec<String> {
        let fields = cols
            .iter()
            .map(|(name, (sedona_type, _))| sedona_type.to_storage_field(name, true).unwrap())
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new(fields));
        let batch =
            RecordBatch::try_new(schema, cols.into_iter().map(|(_, (_, col))| col).collect())
                .unwrap();
        let ctx = SedonaContext::new();

        let mut out = Vec::new();
        show_batches(&ctx, &mut out, &batch.schema(), vec![batch], options).unwrap();

        String::from_utf8(out)
            .unwrap()
            .lines()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn render_utf8() {
        let mut options = DisplayTableOptions::new();
        options.display_mode = DisplayMode::Utf8;
        assert_eq!(
            render_cols(test_cols()[0..2].to_vec(), options.clone()),
            vec![
                "┌──────┬────────────────────────────────────────┐",
                "│ shrt ┆                  long                  │",
                "╞══════╪════════════════════════════════════════╡",
                "│ abcd ┆ you see, what happened was             │",
                "├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤",
                "│ efgh ┆ the elephant stomped so may times that │",
                "├╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤",
                "│      ┆                                        │",
                "└──────┴────────────────────────────────────────┘"
            ]
        );
    }

    #[test]
    fn render_multiline() {
        let cols = vec![(
            "multiline",
            (
                SedonaType::Arrow(DataType::Utf8),
                arrow_array::create_array!(Utf8, [Some("one\ntwo\nthree")]) as ArrayRef,
            ),
        )];

        // By default, multiple lines are truncated
        let mut options = DisplayTableOptions::new();
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+-----------+",
                "| multiline |",
                "+-----------+",
                "| one...    |",
                "+-----------+"
            ]
        );

        // ...but they can be displayed
        options.max_row_height = usize::MAX;
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+-----------+",
                "| multiline |",
                "+-----------+",
                "| one       |",
                "| two       |",
                "| three     |",
                "+-----------+"
            ]
        );
    }

    #[test]
    fn numeric_truncate_header_but_not_content() {
        let cols = vec![(
            "a very long column name",
            (
                SedonaType::Arrow(DataType::Int32),
                arrow_array::create_array!(Int32, [123456789]) as ArrayRef,
            ),
        )];

        // The content should never be truncated but the header can be
        let mut options = DisplayTableOptions::new();
        options.table_width = 5;
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+-----------+",
                "| a very... |",
                "+-----------+",
                "| 123456789 |",
                "+-----------+"
            ]
        );

        // ...but the header should display if there is room
        let mut options = DisplayTableOptions::new();
        options.table_width = 50;
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+-------------------------+",
                "| a very long column name |",
                "+-------------------------+",
                "|               123456789 |",
                "+-------------------------+"
            ]
        );
    }

    #[rstest]
    #[case(WKB_GEOMETRY)]
    #[case(WKB_VIEW_GEOMETRY)]
    fn geometry_header(#[case] sedona_type: SedonaType) {
        let cols = vec![(
            "geometry",
            (
                sedona_type.clone(),
                create_array(
                    &[
                        Some("POINT (0 1)"),
                        Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                        None,
                    ],
                    &sedona_type,
                ) as ArrayRef,
            ),
        )];

        let mut options = DisplayTableOptions::new();
        options.table_width = 50;
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+----------------------------+",
                "|          geometry          |",
                "+----------------------------+",
                "| POINT(0 1)                 |",
                "| POLYGON((0 0,1 0,0 1,0 0)) |",
                "|                            |",
                "+----------------------------+"
            ]
        );
    }

    #[rstest]
    #[case(WKB_GEOGRAPHY)]
    #[case(WKB_VIEW_GEOGRAPHY)]
    fn geography_header(#[case] sedona_type: SedonaType) {
        let cols = vec![(
            "geography",
            (
                sedona_type.clone(),
                create_array(
                    &[
                        Some("POINT (0 1)"),
                        Some("POLYGON ((0 0, 1 0, 0 1, 0 0))"),
                        None,
                    ],
                    &sedona_type,
                ) as ArrayRef,
            ),
        )];

        let mut options = DisplayTableOptions::new();
        options.table_width = 50;
        assert_eq!(
            render_cols(cols.clone(), options.clone()),
            vec![
                "+----------------------------+",
                "|          geography         |",
                "+----------------------------+",
                "| POINT(0 1)                 |",
                "| POLYGON((0 0,1 0,0 1,0 0)) |",
                "|                            |",
                "+----------------------------+"
            ]
        );
    }

    #[test]
    fn render_col_type() {
        let mut options = DisplayTableOptions::new();
        options.table_width = 50;
        options.arrow_options = options.arrow_options.with_types_info(true);
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+--------------+-----------+--------------+",
                "| shrt |     long     |  numeric  |   geometry   |",
                "| utf8 |     utf8     |   int32   |   geometry   |",
                "+------+--------------+-----------+--------------+",
                "| abcd | you see, ... | 123456789 | POINT(0 1)   |",
                "| efgh | the eleph... | 987654321 | POLYGON((... |",
                "|      |              |           |              |",
                "+------+--------------+-----------+--------------+"
            ]
        );
    }

    #[test]
    fn render_full() {
        // Default width is wide enough to print everything
        let options = DisplayTableOptions::new();
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+----------------------------------------+-----------+----------------------------+",
                "| shrt |                  long                  |  numeric  |          geometry          |",
                "+------+----------------------------------------+-----------+----------------------------+",
                "| abcd | you see, what happened was             | 123456789 | POINT(0 1)                 |",
                "| efgh | the elephant stomped so may times that | 987654321 | POLYGON((0 0,1 0,0 1,0 0)) |",
                "|      |                                        |           |                            |",
                "+------+----------------------------------------+-----------+----------------------------+"
            ]
        );
    }

    #[test]
    fn render_truncate_one() {
        // The first thing that happens is that the renderer abbreviates the long characters
        let mut options = DisplayTableOptions::new();
        options.table_width = 80;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+------------------------------+-----------+----------------------------+",
                "| shrt |             long             |  numeric  |          geometry          |",
                "+------+------------------------------+-----------+----------------------------+",
                "| abcd | you see, what happened was   | 123456789 | POINT(0 1)                 |",
                "| efgh | the elephant stomped so m... | 987654321 | POLYGON((0 0,1 0,0 1,0 0)) |",
                "|      |                              |           |                            |",
                "+------+------------------------------+-----------+----------------------------+"
            ]
        );
    }

    #[test]
    fn render_truncate_two() {
        // Then the geometry column also becomes abbreviated
        let mut options = DisplayTableOptions::new();
        options.table_width = 60;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+-------------------+-----------+-------------------+",
                "| shrt |        long       |  numeric  |      geometry     |",
                "+------+-------------------+-----------+-------------------+",
                "| abcd | you see, what ... | 123456789 | POINT(0 1)        |",
                "| efgh | the elephant s... | 987654321 | POLYGON((0 0,1... |",
                "|      |                   |           |                   |",
                "+------+-------------------+-----------+-------------------+"
            ]
        );
    }

    #[test]
    fn render_truncate_min() {
        // This is the narrowest width where all columns are shown
        let mut options = DisplayTableOptions::new();
        options.table_width = 46;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+------------+-----------+------------+",
                "| shrt |    long    |  numeric  |  geometry  |",
                "+------+------------+-----------+------------+",
                "| abcd | you see... | 123456789 | POINT(0 1) |",
                "| efgh | the ele... | 987654321 | POLYGON... |",
                "|      |            |           |            |",
                "+------+------------+-----------+------------+"
            ]
        );
    }

    #[test]
    fn render_omit_one() {
        // Now we omit a column to satisfy constraints
        let mut options = DisplayTableOptions::new();
        options.table_width = 45;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+---------------+-----+--------------+",
                "| shrt |      long     | ... |   geometry   |",
                "+------+---------------+-----+--------------+",
                "| abcd | you see, w... | ... | POINT(0 1)   |",
                "| efgh | the elepha... | ... | POLYGON((... |",
                "|      |               | ... |              |",
                "+------+---------------+-----+--------------+"
            ]
        );
    }

    #[test]
    fn render_omit_two() {
        // Now we omit two columns to satisfy constraints
        let mut options = DisplayTableOptions::new();
        options.table_width = 35;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+-----+--------------------+",
                "| shrt | ... |      geometry      |",
                "+------+-----+--------------------+",
                "| abcd | ... | POINT(0 1)         |",
                "| efgh | ... | POLYGON((0 0,1 ... |",
                "|      | ... |                    |",
                "+------+-----+--------------------+"
            ]
        );
    }

    #[test]
    fn render_omit_three() {
        // Now we omit three columns to satisfy constraints
        let mut options = DisplayTableOptions::new();
        options.table_width = 25;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+-----+",
                "| shrt | ... |",
                "+------+-----+",
                "| abcd | ... |",
                "| efgh | ... |",
                "|      | ... |",
                "+------+-----+"
            ]
        );
    }

    #[test]
    fn render_always_show_at_least_one() {
        // Even with a completely unreasonable width constraint we show one column
        let mut options = DisplayTableOptions::new();
        options.table_width = 1;
        assert_eq!(
            render_cols(test_cols(), options.clone()),
            vec![
                "+------+-----+",
                "| shrt | ... |",
                "+------+-----+",
                "| abcd | ... |",
                "| efgh | ... |",
                "|      | ... |",
                "+------+-----+"
            ]
        );
    }
}
