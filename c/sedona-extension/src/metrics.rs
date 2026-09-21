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

//! Serde support for DataFusion execution plan metrics.
//!
//! DataFusion's metric types do not implement serde. This module provides an
//! owned, serializable snapshot that can be converted back into a real
//! [`ExecutionPlanMetricsSet`].

use std::{
    any::Any,
    fmt::{Display, Formatter},
    sync::{Arc, Mutex},
    time::Duration,
};

use datafusion_physical_plan::metrics::{
    Count, CustomMetricValue, ExecutionPlanMetricsSet, Gauge, Label, Metric, MetricCategory,
    MetricType, MetricValue, MetricsSet, PruningMetrics, RatioMergeStrategy, RatioMetrics, Time,
    Timestamp,
};
use serde::{Deserialize, Serialize};

/// An owned, serde-compatible snapshot of an [`ExecutionPlanMetricsSet`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableExecutionPlanMetricsSet {
    pub metrics: Vec<SerializableMetric>,
}

impl SerializableExecutionPlanMetricsSet {
    pub fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self::from_metrics_set(&metrics.clone_inner())
    }

    pub fn from_metrics_set(metrics: &MetricsSet) -> Self {
        Self {
            metrics: metrics
                .iter()
                .map(|metric| SerializableMetric::from(metric.as_ref()))
                .collect(),
        }
    }

    pub fn into_metrics_set(self) -> MetricsSet {
        self.metrics
            .into_iter()
            .map(|metric| Arc::new(metric.into_metric()))
            .collect()
    }

    pub fn into_execution_plan_metrics_set(self) -> ExecutionPlanMetricsSet {
        self.into_metrics_set().into()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableMetric {
    pub value: SerializableMetricValue,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub labels: Vec<SerializableLabel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partition: Option<usize>,
    pub metric_type: SerializableMetricType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<SerializableMetricCategory>,
}

impl From<&Metric> for SerializableMetric {
    fn from(metric: &Metric) -> Self {
        Self {
            value: metric.value().into(),
            labels: metric.labels().iter().map(Into::into).collect(),
            partition: metric.partition(),
            metric_type: metric.metric_type().into(),
            category: metric.metric_category().map(Into::into),
        }
    }
}

impl SerializableMetric {
    fn into_metric(self) -> Metric {
        let labels = self.labels.into_iter().map(Into::into).collect();
        let mut metric = Metric::new_with_labels(self.value.into(), self.partition, labels)
            .with_type(self.metric_type.into());
        if let Some(category) = self.category {
            metric = metric.with_category(category.into());
        }
        metric
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableLabel {
    pub name: String,
    pub value: String,
}

impl From<&Label> for SerializableLabel {
    fn from(label: &Label) -> Self {
        Self {
            name: label.name().to_owned(),
            value: label.value().to_owned(),
        }
    }
}

impl From<SerializableLabel> for Label {
    fn from(label: SerializableLabel) -> Self {
        Self::new(label.name, label.value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializableMetricType {
    Summary,
    Dev,
}

impl From<MetricType> for SerializableMetricType {
    fn from(value: MetricType) -> Self {
        match value {
            MetricType::Summary => Self::Summary,
            MetricType::Dev => Self::Dev,
        }
    }
}

impl From<SerializableMetricType> for MetricType {
    fn from(value: SerializableMetricType) -> Self {
        match value {
            SerializableMetricType::Summary => Self::Summary,
            SerializableMetricType::Dev => Self::Dev,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializableMetricCategory {
    Rows,
    Bytes,
    Timing,
    Uncategorized,
}

impl From<MetricCategory> for SerializableMetricCategory {
    fn from(value: MetricCategory) -> Self {
        match value {
            MetricCategory::Rows => Self::Rows,
            MetricCategory::Bytes => Self::Bytes,
            MetricCategory::Timing => Self::Timing,
            MetricCategory::Uncategorized => Self::Uncategorized,
        }
    }
}

impl From<SerializableMetricCategory> for MetricCategory {
    fn from(value: SerializableMetricCategory) -> Self {
        match value {
            SerializableMetricCategory::Rows => Self::Rows,
            SerializableMetricCategory::Bytes => Self::Bytes,
            SerializableMetricCategory::Timing => Self::Timing,
            SerializableMetricCategory::Uncategorized => Self::Uncategorized,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SerializableMetricValue {
    OutputRows {
        value: usize,
    },
    ElapsedCompute {
        nanos: usize,
    },
    SpillCount {
        value: usize,
    },
    SpilledBytes {
        value: usize,
    },
    OutputBytes {
        value: usize,
    },
    OutputBatches {
        value: usize,
    },
    SpilledRows {
        value: usize,
    },
    CurrentMemoryUsage {
        value: usize,
    },
    Count {
        name: String,
        value: usize,
    },
    Gauge {
        name: String,
        value: usize,
    },
    Time {
        name: String,
        nanos: usize,
    },
    StartTimestamp {
        value: Option<String>,
    },
    EndTimestamp {
        value: Option<String>,
    },
    PruningMetrics {
        name: String,
        pruned: usize,
        matched: usize,
        fully_matched: usize,
    },
    Ratio {
        name: String,
        part: usize,
        total: usize,
        merge_strategy: SerializableRatioMergeStrategy,
        display_raw_values: bool,
    },
    /// Opaque custom metrics preserve their observable state, but not their
    /// concrete Rust implementation.
    Custom {
        name: String,
        display: String,
        value: usize,
    },
}

impl From<&MetricValue> for SerializableMetricValue {
    fn from(value: &MetricValue) -> Self {
        match value {
            MetricValue::OutputRows(value) => Self::OutputRows {
                value: value.value(),
            },
            MetricValue::ElapsedCompute(value) => Self::ElapsedCompute {
                nanos: value.value(),
            },
            MetricValue::SpillCount(value) => Self::SpillCount {
                value: value.value(),
            },
            MetricValue::SpilledBytes(value) => Self::SpilledBytes {
                value: value.value(),
            },
            MetricValue::OutputBytes(value) => Self::OutputBytes {
                value: value.value(),
            },
            MetricValue::OutputBatches(value) => Self::OutputBatches {
                value: value.value(),
            },
            MetricValue::SpilledRows(value) => Self::SpilledRows {
                value: value.value(),
            },
            MetricValue::CurrentMemoryUsage(value) => Self::CurrentMemoryUsage {
                value: value.value(),
            },
            MetricValue::Count { name, count } => Self::Count {
                name: name.to_string(),
                value: count.value(),
            },
            MetricValue::Gauge { name, gauge } => Self::Gauge {
                name: name.to_string(),
                value: gauge.value(),
            },
            MetricValue::Time { name, time } => Self::Time {
                name: name.to_string(),
                nanos: time.value(),
            },
            MetricValue::StartTimestamp(value) => Self::StartTimestamp {
                value: value.value().map(|value| value.to_rfc3339()),
            },
            MetricValue::EndTimestamp(value) => Self::EndTimestamp {
                value: value.value().map(|value| value.to_rfc3339()),
            },
            MetricValue::PruningMetrics {
                name,
                pruning_metrics,
            } => Self::PruningMetrics {
                name: name.to_string(),
                pruned: pruning_metrics.pruned(),
                matched: pruning_metrics.matched(),
                fully_matched: pruning_metrics.fully_matched(),
            },
            MetricValue::Ratio {
                name,
                ratio_metrics,
            } => Self::Ratio {
                name: name.to_string(),
                part: ratio_metrics.part(),
                total: ratio_metrics.total(),
                merge_strategy: ratio_metrics.merge_strategy().into(),
                display_raw_values: ratio_metrics.display_raw_values(),
            },
            MetricValue::Custom { name, value } => Self::Custom {
                name: name.to_string(),
                display: value.to_string(),
                value: value.as_usize(),
            },
        }
    }
}

impl From<SerializableMetricValue> for MetricValue {
    fn from(value: SerializableMetricValue) -> Self {
        match value {
            SerializableMetricValue::OutputRows { value } => Self::OutputRows(count(value)),
            SerializableMetricValue::ElapsedCompute { nanos } => Self::ElapsedCompute(time(nanos)),
            SerializableMetricValue::SpillCount { value } => Self::SpillCount(count(value)),
            SerializableMetricValue::SpilledBytes { value } => Self::SpilledBytes(count(value)),
            SerializableMetricValue::OutputBytes { value } => Self::OutputBytes(count(value)),
            SerializableMetricValue::OutputBatches { value } => Self::OutputBatches(count(value)),
            SerializableMetricValue::SpilledRows { value } => Self::SpilledRows(count(value)),
            SerializableMetricValue::CurrentMemoryUsage { value } => {
                Self::CurrentMemoryUsage(gauge(value))
            }
            SerializableMetricValue::Count { name, value } => Self::Count {
                name: name.into(),
                count: count(value),
            },
            SerializableMetricValue::Gauge { name, value } => Self::Gauge {
                name: name.into(),
                gauge: gauge(value),
            },
            SerializableMetricValue::Time { name, nanos } => Self::Time {
                name: name.into(),
                time: time(nanos),
            },
            SerializableMetricValue::StartTimestamp { value } => {
                Self::StartTimestamp(timestamp(value))
            }
            SerializableMetricValue::EndTimestamp { value } => Self::EndTimestamp(timestamp(value)),
            SerializableMetricValue::PruningMetrics {
                name,
                pruned,
                matched,
                fully_matched,
            } => {
                let metrics = PruningMetrics::new();
                metrics.add_pruned(pruned);
                metrics.add_matched(matched);
                metrics.add_fully_matched(fully_matched);
                Self::PruningMetrics {
                    name: name.into(),
                    pruning_metrics: metrics,
                }
            }
            SerializableMetricValue::Ratio {
                name,
                part,
                total,
                merge_strategy,
                display_raw_values,
            } => {
                let metrics = RatioMetrics::new()
                    .with_merge_strategy(merge_strategy.into())
                    .with_display_raw_values(display_raw_values);
                metrics.set_part(part);
                metrics.set_total(total);
                Self::Ratio {
                    name: name.into(),
                    ratio_metrics: metrics,
                }
            }
            SerializableMetricValue::Custom {
                name,
                display,
                value,
            } => Self::Custom {
                name: name.into(),
                value: Arc::new(SerializedCustomMetric::new(display, value)),
            },
        }
    }
}

fn count(value: usize) -> Count {
    let count = Count::new();
    count.add(value);
    count
}

fn gauge(value: usize) -> Gauge {
    let gauge = Gauge::new();
    gauge.set(value);
    gauge
}

fn time(nanos: usize) -> Time {
    let time = Time::new();
    if nanos > 0 {
        time.add_duration(Duration::from_nanos(nanos as u64));
    }
    time
}

fn timestamp(value: Option<String>) -> Timestamp {
    let timestamp = Timestamp::new();
    if let Some(value) = value {
        // Values emitted by `to_rfc3339` always parse successfully. Keeping
        // deserialization infallible also makes older/hand-authored JSON with
        // an invalid timestamp behave like an unrecorded timestamp.
        if let Ok(value) = value.parse() {
            timestamp.set(value);
        }
    }
    timestamp
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SerializableRatioMergeStrategy {
    AddPartAddTotal,
    AddPartSetTotal,
    SetPartAddTotal,
}

impl From<&RatioMergeStrategy> for SerializableRatioMergeStrategy {
    fn from(value: &RatioMergeStrategy) -> Self {
        match value {
            RatioMergeStrategy::AddPartAddTotal => Self::AddPartAddTotal,
            RatioMergeStrategy::AddPartSetTotal => Self::AddPartSetTotal,
            RatioMergeStrategy::SetPartAddTotal => Self::SetPartAddTotal,
        }
    }
}

impl From<SerializableRatioMergeStrategy> for RatioMergeStrategy {
    fn from(value: SerializableRatioMergeStrategy) -> Self {
        match value {
            SerializableRatioMergeStrategy::AddPartAddTotal => Self::AddPartAddTotal,
            SerializableRatioMergeStrategy::AddPartSetTotal => Self::AddPartSetTotal,
            SerializableRatioMergeStrategy::SetPartAddTotal => Self::SetPartAddTotal,
        }
    }
}

#[derive(Debug)]
struct SerializedCustomMetric {
    state: Mutex<SerializedCustomMetricState>,
}

#[derive(Debug, Clone)]
struct SerializedCustomMetricState {
    display: String,
    value: usize,
}

impl SerializedCustomMetric {
    fn new(display: String, value: usize) -> Self {
        Self {
            state: Mutex::new(SerializedCustomMetricState { display, value }),
        }
    }

    fn snapshot(&self) -> SerializedCustomMetricState {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl Display for SerializedCustomMetric {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.snapshot().display)
    }
}

impl CustomMetricValue for SerializedCustomMetric {
    fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
        Arc::new(Self::new(String::new(), 0))
    }

    fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
        let Some(other) = other.as_any().downcast_ref::<Self>() else {
            return;
        };
        let other = other.snapshot();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.display.is_empty() {
            state.display = other.display;
        }
        state.value = state.value.saturating_add(other.value);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_usize(&self) -> usize {
        self.snapshot().value
    }

    fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
        other.as_any().downcast_ref::<Self>().is_some_and(|other| {
            let this = self.snapshot();
            let other = other.snapshot();
            this.display == other.display && this.value == other.value
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_json_roundtrip_preserves_all_metric_variants() {
        let values = vec![
            SerializableMetricValue::OutputRows { value: 1 },
            SerializableMetricValue::ElapsedCompute { nanos: 2 },
            SerializableMetricValue::SpillCount { value: 3 },
            SerializableMetricValue::SpilledBytes { value: 4 },
            SerializableMetricValue::OutputBytes { value: 5 },
            SerializableMetricValue::OutputBatches { value: 6 },
            SerializableMetricValue::SpilledRows { value: 7 },
            SerializableMetricValue::CurrentMemoryUsage { value: 8 },
            SerializableMetricValue::Count {
                name: "count".to_owned(),
                value: 9,
            },
            SerializableMetricValue::Gauge {
                name: "gauge".to_owned(),
                value: 10,
            },
            SerializableMetricValue::Time {
                name: "time".to_owned(),
                nanos: 11,
            },
            SerializableMetricValue::StartTimestamp {
                value: Some("2026-09-21T12:34:56.123456789+00:00".to_owned()),
            },
            SerializableMetricValue::EndTimestamp { value: None },
            SerializableMetricValue::PruningMetrics {
                name: "pruning".to_owned(),
                pruned: 12,
                matched: 13,
                fully_matched: 14,
            },
            SerializableMetricValue::Ratio {
                name: "ratio".to_owned(),
                part: 15,
                total: 16,
                merge_strategy: SerializableRatioMergeStrategy::AddPartSetTotal,
                display_raw_values: false,
            },
            SerializableMetricValue::Custom {
                name: "custom".to_owned(),
                display: "custom display".to_owned(),
                value: 17,
            },
        ];
        let snapshot = SerializableExecutionPlanMetricsSet {
            metrics: values
                .into_iter()
                .enumerate()
                .map(|(partition, value)| SerializableMetric {
                    value,
                    labels: vec![SerializableLabel {
                        name: "source".to_owned(),
                        value: "test".to_owned(),
                    }],
                    partition: Some(partition),
                    metric_type: SerializableMetricType::Summary,
                    category: Some(SerializableMetricCategory::Rows),
                })
                .collect(),
        };

        let json = serde_json::to_string(&snapshot).unwrap();
        let decoded: SerializableExecutionPlanMetricsSet = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, snapshot);

        let metrics = decoded.into_execution_plan_metrics_set();
        let reconstructed = SerializableExecutionPlanMetricsSet::new(&metrics);
        assert_eq!(reconstructed, snapshot);
    }

    #[test]
    fn json_is_structured_for_external_consumers() {
        let metrics = ExecutionPlanMetricsSet::new();
        let count = Count::new();
        count.add(42);
        metrics.register(Arc::new(
            Metric::new_with_labels(
                MetricValue::Count {
                    name: "files_scanned".into(),
                    count,
                },
                Some(3),
                vec![Label::new("format", "parquet")],
            )
            .with_type(MetricType::Summary)
            .with_category(MetricCategory::Rows),
        ));

        let json =
            serde_json::to_value(SerializableExecutionPlanMetricsSet::new(&metrics)).unwrap();
        assert_eq!(json["metrics"][0]["value"]["type"], "count");
        assert_eq!(json["metrics"][0]["value"]["name"], "files_scanned");
        assert_eq!(json["metrics"][0]["value"]["value"], 42);
        assert_eq!(json["metrics"][0]["partition"], 3);
        assert_eq!(json["metrics"][0]["labels"][0]["name"], "format");
        assert_eq!(json["metrics"][0]["metric_type"], "summary");
        assert_eq!(json["metrics"][0]["category"], "rows");
    }
}
