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
    collections::HashMap,
    fmt::{Display, Formatter},
    sync::{Arc, Mutex},
    time::Duration,
};

use datafusion_physical_plan::metrics::{
    Count, CustomMetricValue, ExecutionPlanMetricsSet, Gauge, Label, Metric, MetricCategory,
    MetricType, MetricValue, MetricsSet, PruningMetrics, RatioMergeStrategy, RatioMetrics, Time,
    Timestamp,
};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value};

/// An owned, serde-compatible snapshot of an [`ExecutionPlanMetricsSet`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableExecutionPlanMetricsSet {
    /// The legacy metrics representation used by SedonaDB 0.4.1.
    ///
    /// Keep emitting this field so older hosts can decode metrics produced by
    /// newer extensions. Newer hosts use `metrics` and only retain `display`
    /// for wire compatibility.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub display: String,
    /// Structured metrics added after SedonaDB 0.4.1.
    ///
    /// Defaulting this field lets newer hosts accept the legacy
    /// `{ "display": "..." }` representation without failing the entire
    /// metrics property decode.
    #[serde(default)]
    pub metrics: Vec<SerializableMetric>,
}

impl SerializableExecutionPlanMetricsSet {
    pub fn new(metrics: &ExecutionPlanMetricsSet) -> Self {
        Self::from_metrics_set(&metrics.clone_inner())
    }

    pub fn from_metrics_set(metrics: &MetricsSet) -> Self {
        let custom_aggregates = custom_metric_aggregates(metrics);
        Self {
            display: metrics.to_string(),
            metrics: metrics
                .iter()
                .map(|metric| {
                    SerializableMetric::from_metric(
                        metric.as_ref(),
                        custom_aggregates.get(metric.value().name()),
                    )
                })
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
        Self::from_metric(metric, None)
    }
}

impl SerializableMetric {
    fn from_metric(
        metric: &Metric,
        custom_aggregate: Option<&SerializableCustomMetricAggregate>,
    ) -> Self {
        Self {
            value: SerializableMetricValue::from_metric_value(metric.value(), custom_aggregate),
            labels: metric.labels().iter().map(Into::into).collect(),
            partition: metric.partition(),
            metric_type: metric.metric_type().into(),
            category: metric.metric_category().map(Into::into),
        }
    }

    fn into_metric(self) -> Metric {
        let labels = self.labels.into_iter().map(Into::into).collect();
        let mut metric = Metric::new_with_labels(self.value.into(), self.partition, labels)
            .with_type(self.metric_type.into());
        if let Some(category) = self.category.and_then(SerializableMetricCategory::known) {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializableMetricType {
    Known(KnownMetricType),
    Unknown(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownMetricType {
    Summary,
    Dev,
}

impl From<MetricType> for SerializableMetricType {
    fn from(value: MetricType) -> Self {
        match value {
            MetricType::Summary => Self::Known(KnownMetricType::Summary),
            MetricType::Dev => Self::Known(KnownMetricType::Dev),
        }
    }
}

impl From<SerializableMetricType> for MetricType {
    fn from(value: SerializableMetricType) -> Self {
        match value {
            SerializableMetricType::Known(KnownMetricType::Summary) => Self::Summary,
            SerializableMetricType::Known(KnownMetricType::Dev)
            | SerializableMetricType::Unknown(_) => Self::Dev,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializableMetricCategory {
    Known(KnownMetricCategory),
    Unknown(String),
}

impl SerializableMetricCategory {
    fn known(self) -> Option<KnownMetricCategory> {
        match self {
            Self::Known(category) => Some(category),
            Self::Unknown(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnownMetricCategory {
    Rows,
    Bytes,
    Timing,
    Uncategorized,
}

impl From<MetricCategory> for SerializableMetricCategory {
    fn from(value: MetricCategory) -> Self {
        match value {
            MetricCategory::Rows => Self::Known(KnownMetricCategory::Rows),
            MetricCategory::Bytes => Self::Known(KnownMetricCategory::Bytes),
            MetricCategory::Timing => Self::Known(KnownMetricCategory::Timing),
            MetricCategory::Uncategorized => Self::Known(KnownMetricCategory::Uncategorized),
        }
    }
}

impl From<KnownMetricCategory> for MetricCategory {
    fn from(value: KnownMetricCategory) -> Self {
        match value {
            KnownMetricCategory::Rows => Self::Rows,
            KnownMetricCategory::Bytes => Self::Bytes,
            KnownMetricCategory::Timing => Self::Timing,
            KnownMetricCategory::Uncategorized => Self::Uncategorized,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SerializableMetricValue {
    Known(KnownMetricValue),
    /// A metric variant introduced by a newer producer. Its complete JSON
    /// object is retained until conversion to DataFusion's current types.
    Unknown(Map<String, Value>),
}

impl From<KnownMetricValue> for SerializableMetricValue {
    fn from(value: KnownMetricValue) -> Self {
        Self::Known(value)
    }
}

impl Serialize for SerializableMetricValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Known(value) => value.serialize(serializer),
            Self::Unknown(value) => value.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for SerializableMetricValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let object = Map::<String, Value>::deserialize(deserializer)?;
        let type_name = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| D::Error::missing_field("type"))?;

        if KnownMetricValue::is_known_type(type_name) {
            serde_json::from_value(Value::Object(object))
                .map(Self::Known)
                .map_err(D::Error::custom)
        } else {
            Ok(Self::Unknown(object))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnownMetricValue {
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
        #[serde(default, skip_serializing_if = "Option::is_none")]
        aggregate: Option<SerializableCustomMetricAggregate>,
    },
}

/// The result of aggregating all producer-side custom metrics with the same
/// name, while their concrete implementation is still available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializableCustomMetricAggregate {
    pub display: String,
    pub value: usize,
    pub input_count: usize,
}

impl KnownMetricValue {
    fn is_known_type(type_name: &str) -> bool {
        matches!(
            type_name,
            "output_rows"
                | "elapsed_compute"
                | "spill_count"
                | "spilled_bytes"
                | "output_bytes"
                | "output_batches"
                | "spilled_rows"
                | "current_memory_usage"
                | "count"
                | "gauge"
                | "time"
                | "start_timestamp"
                | "end_timestamp"
                | "pruning_metrics"
                | "ratio"
                | "custom"
        )
    }
}

impl From<&MetricValue> for SerializableMetricValue {
    fn from(value: &MetricValue) -> Self {
        Self::from_metric_value(value, None)
    }
}

impl SerializableMetricValue {
    fn from_metric_value(
        value: &MetricValue,
        custom_aggregate: Option<&SerializableCustomMetricAggregate>,
    ) -> Self {
        let value = match value {
            MetricValue::OutputRows(value) => KnownMetricValue::OutputRows {
                value: value.value(),
            },
            MetricValue::ElapsedCompute(value) => KnownMetricValue::ElapsedCompute {
                nanos: value.value(),
            },
            MetricValue::SpillCount(value) => KnownMetricValue::SpillCount {
                value: value.value(),
            },
            MetricValue::SpilledBytes(value) => KnownMetricValue::SpilledBytes {
                value: value.value(),
            },
            MetricValue::OutputBytes(value) => KnownMetricValue::OutputBytes {
                value: value.value(),
            },
            MetricValue::OutputBatches(value) => KnownMetricValue::OutputBatches {
                value: value.value(),
            },
            MetricValue::SpilledRows(value) => KnownMetricValue::SpilledRows {
                value: value.value(),
            },
            MetricValue::CurrentMemoryUsage(value) => KnownMetricValue::CurrentMemoryUsage {
                value: value.value(),
            },
            MetricValue::Count { name, count } => KnownMetricValue::Count {
                name: name.to_string(),
                value: count.value(),
            },
            MetricValue::Gauge { name, gauge } => KnownMetricValue::Gauge {
                name: name.to_string(),
                value: gauge.value(),
            },
            MetricValue::Time { name, time } => KnownMetricValue::Time {
                name: name.to_string(),
                nanos: time.value(),
            },
            MetricValue::StartTimestamp(value) => KnownMetricValue::StartTimestamp {
                value: value.value().map(|value| value.to_rfc3339()),
            },
            MetricValue::EndTimestamp(value) => KnownMetricValue::EndTimestamp {
                value: value.value().map(|value| value.to_rfc3339()),
            },
            MetricValue::PruningMetrics {
                name,
                pruning_metrics,
            } => KnownMetricValue::PruningMetrics {
                name: name.to_string(),
                pruned: pruning_metrics.pruned(),
                matched: pruning_metrics.matched(),
                fully_matched: pruning_metrics.fully_matched(),
            },
            MetricValue::Ratio {
                name,
                ratio_metrics,
            } => KnownMetricValue::Ratio {
                name: name.to_string(),
                part: ratio_metrics.part(),
                total: ratio_metrics.total(),
                merge_strategy: ratio_metrics.merge_strategy().into(),
                display_raw_values: ratio_metrics.display_raw_values(),
            },
            MetricValue::Custom { name, value } => KnownMetricValue::Custom {
                name: name.to_string(),
                display: value.to_string(),
                value: value.as_usize(),
                aggregate: custom_aggregate.cloned(),
            },
        };
        Self::Known(value)
    }
}

impl From<SerializableMetricValue> for MetricValue {
    fn from(value: SerializableMetricValue) -> Self {
        let value = match value {
            SerializableMetricValue::Known(value) => value,
            SerializableMetricValue::Unknown(object) => return unknown_metric_value(object),
        };
        match value {
            KnownMetricValue::OutputRows { value } => Self::OutputRows(count(value)),
            KnownMetricValue::ElapsedCompute { nanos } => Self::ElapsedCompute(time(nanos)),
            KnownMetricValue::SpillCount { value } => Self::SpillCount(count(value)),
            KnownMetricValue::SpilledBytes { value } => Self::SpilledBytes(count(value)),
            KnownMetricValue::OutputBytes { value } => Self::OutputBytes(count(value)),
            KnownMetricValue::OutputBatches { value } => Self::OutputBatches(count(value)),
            KnownMetricValue::SpilledRows { value } => Self::SpilledRows(count(value)),
            KnownMetricValue::CurrentMemoryUsage { value } => {
                Self::CurrentMemoryUsage(gauge(value))
            }
            KnownMetricValue::Count { name, value } => Self::Count {
                name: name.into(),
                count: count(value),
            },
            KnownMetricValue::Gauge { name, value } => Self::Gauge {
                name: name.into(),
                gauge: gauge(value),
            },
            KnownMetricValue::Time { name, nanos } => Self::Time {
                name: name.into(),
                time: time(nanos),
            },
            KnownMetricValue::StartTimestamp { value } => Self::StartTimestamp(timestamp(value)),
            KnownMetricValue::EndTimestamp { value } => Self::EndTimestamp(timestamp(value)),
            KnownMetricValue::PruningMetrics {
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
            KnownMetricValue::Ratio {
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
            KnownMetricValue::Custom {
                name,
                display,
                value,
                aggregate,
            } => Self::Custom {
                name: name.into(),
                value: Arc::new(SerializedCustomMetric::new(display, value, aggregate)),
            },
        }
    }
}

fn custom_metric_aggregates(
    metrics: &MetricsSet,
) -> HashMap<String, SerializableCustomMetricAggregate> {
    let mut accumulators: HashMap<String, (Arc<dyn CustomMetricValue>, usize)> = HashMap::new();

    for metric in metrics.iter() {
        let MetricValue::Custom { name, value } = metric.value() else {
            continue;
        };

        let (accumulator, input_count) = accumulators
            .entry(name.to_string())
            .or_insert_with(|| (value.new_empty(), 0));
        accumulator.aggregate(Arc::clone(value));
        *input_count += 1;
    }

    accumulators
        .into_iter()
        .filter_map(|(name, (value, input_count))| {
            (input_count > 1).then(|| {
                (
                    name,
                    SerializableCustomMetricAggregate {
                        display: value.to_string(),
                        value: value.as_usize(),
                        input_count,
                    },
                )
            })
        })
        .collect()
}

fn unknown_metric_value(object: Map<String, Value>) -> MetricValue {
    let type_name = object
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("unknown_metric");
    let name = object
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or(type_name)
        .to_owned();
    let numeric_value = object
        .get("value")
        .or_else(|| object.get("as_usize_value"))
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(0);
    let display = Value::Object(object).to_string();

    MetricValue::Custom {
        name: name.into(),
        value: Arc::new(SerializedCustomMetric::new(display, numeric_value, None)),
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct SerializedCustomMetricState {
    display: String,
    value: usize,
    aggregate: Option<SerializableCustomMetricAggregate>,
    input_count: usize,
}

impl SerializedCustomMetric {
    fn new(
        display: String,
        value: usize,
        aggregate: Option<SerializableCustomMetricAggregate>,
    ) -> Self {
        Self {
            state: Mutex::new(SerializedCustomMetricState {
                display,
                value,
                aggregate,
                input_count: 1,
            }),
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
        let aggregate = self.snapshot().aggregate;
        Arc::new(Self {
            state: Mutex::new(SerializedCustomMetricState {
                display: String::new(),
                value: 0,
                aggregate,
                input_count: 0,
            }),
        })
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
        if state.aggregate.is_none() {
            state.aggregate = other.aggregate.clone();
        }
        state.value = state.value.saturating_add(other.value);
        state.input_count = state.input_count.saturating_add(other.input_count);

        let aggregate = state
            .aggregate
            .as_ref()
            .filter(|aggregate| aggregate.input_count == state.input_count)
            .cloned();
        if let Some(aggregate) = aggregate {
            state.display = aggregate.display;
            state.value = aggregate.value;
        } else if state.input_count == 1 {
            state.display = other.display;
        } else {
            // The concrete custom metric implementation is unavailable here,
            // so its formatting cannot be combined safely. Display the numeric
            // projection instead of retaining a stale per-input display.
            state.display = state.value.to_string();
        }
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
            this == other
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug)]
    struct NativeCustomCounter {
        value: AtomicUsize,
    }

    impl NativeCustomCounter {
        fn new(value: usize) -> Self {
            Self {
                value: AtomicUsize::new(value),
            }
        }
    }

    impl Display for NativeCustomCounter {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{} rows", self.as_usize())
        }
    }

    impl CustomMetricValue for NativeCustomCounter {
        fn new_empty(&self) -> Arc<dyn CustomMetricValue> {
            Arc::new(Self::new(0))
        }

        fn aggregate(&self, other: Arc<dyn CustomMetricValue>) {
            let other = other
                .as_any()
                .downcast_ref::<Self>()
                .expect("matching custom metric type");
            self.value.fetch_add(other.as_usize(), Ordering::Relaxed);
        }

        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_usize(&self) -> usize {
            self.value.load(Ordering::Relaxed)
        }

        fn is_eq(&self, other: &Arc<dyn CustomMetricValue>) -> bool {
            other
                .as_any()
                .downcast_ref::<Self>()
                .is_some_and(|other| self.as_usize() == other.as_usize())
        }
    }

    #[test]
    fn serde_json_roundtrip_preserves_all_metric_variants() {
        let values = vec![
            KnownMetricValue::OutputRows { value: 1 },
            KnownMetricValue::ElapsedCompute { nanos: 2 },
            KnownMetricValue::SpillCount { value: 3 },
            KnownMetricValue::SpilledBytes { value: 4 },
            KnownMetricValue::OutputBytes { value: 5 },
            KnownMetricValue::OutputBatches { value: 6 },
            KnownMetricValue::SpilledRows { value: 7 },
            KnownMetricValue::CurrentMemoryUsage { value: 8 },
            KnownMetricValue::Count {
                name: "count".to_owned(),
                value: 9,
            },
            KnownMetricValue::Gauge {
                name: "gauge".to_owned(),
                value: 10,
            },
            KnownMetricValue::Time {
                name: "time".to_owned(),
                nanos: 11,
            },
            KnownMetricValue::StartTimestamp {
                value: Some("2026-09-21T12:34:56.123456789+00:00".to_owned()),
            },
            KnownMetricValue::EndTimestamp { value: None },
            KnownMetricValue::PruningMetrics {
                name: "pruning".to_owned(),
                pruned: 12,
                matched: 13,
                fully_matched: 14,
            },
            KnownMetricValue::Ratio {
                name: "ratio".to_owned(),
                part: 15,
                total: 16,
                merge_strategy: SerializableRatioMergeStrategy::AddPartSetTotal,
                display_raw_values: false,
            },
            KnownMetricValue::Custom {
                name: "custom".to_owned(),
                display: "custom display".to_owned(),
                value: 17,
                aggregate: None,
            },
        ];
        let mut snapshot = SerializableExecutionPlanMetricsSet {
            display: String::new(),
            metrics: values
                .into_iter()
                .map(SerializableMetricValue::from)
                .enumerate()
                .map(|(partition, value)| SerializableMetric {
                    value,
                    labels: vec![SerializableLabel {
                        name: "source".to_owned(),
                        value: "test".to_owned(),
                    }],
                    partition: Some(partition),
                    metric_type: SerializableMetricType::Known(KnownMetricType::Summary),
                    category: Some(SerializableMetricCategory::Known(KnownMetricCategory::Rows)),
                })
                .collect(),
        };
        snapshot.display = snapshot.clone().into_metrics_set().to_string();

        let json = serde_json::to_string(&snapshot).unwrap();
        let decoded: SerializableExecutionPlanMetricsSet = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, snapshot);

        let metrics = decoded.into_execution_plan_metrics_set();
        let reconstructed = SerializableExecutionPlanMetricsSet::new(&metrics);
        assert_eq!(reconstructed, snapshot);
    }

    #[test]
    fn json_is_structured_for_external_consumers() {
        #[derive(Deserialize)]
        struct LegacyMetrics {
            display: String,
        }

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
        assert_eq!(
            json["display"],
            "files_scanned{partition=3, format=parquet}=42"
        );
        let legacy: LegacyMetrics = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(legacy.display, json["display"]);
        assert_eq!(json["metrics"][0]["value"]["type"], "count");
        assert_eq!(json["metrics"][0]["value"]["name"], "files_scanned");
        assert_eq!(json["metrics"][0]["value"]["value"], 42);
        assert_eq!(json["metrics"][0]["partition"], 3);
        assert_eq!(json["metrics"][0]["labels"][0]["name"], "format");
        assert_eq!(json["metrics"][0]["metric_type"], "summary");
        assert_eq!(json["metrics"][0]["category"], "rows");
    }

    #[test]
    fn legacy_metrics_repro() {
        let json = r#"{"display":"output_rows=42"}"#;
        let result =
            serde_json::from_str::<Option<SerializableExecutionPlanMetricsSet>>(json).unwrap();

        let snapshot = result.expect("legacy metrics should deserialize as Some");
        assert_eq!(snapshot.display, "output_rows=42");
        assert!(snapshot.metrics.is_empty());
        assert!(snapshot.into_metrics_set().iter().next().is_none());
    }

    #[test]
    fn future_metric_and_classifications_are_handled_gracefully() {
        let json = serde_json::json!({
            "metrics": [{
                "value": {
                    "type": "histogram",
                    "name": "scan_latency",
                    "value": 23,
                    "buckets": [1, 5, 10]
                },
                "metric_type": "diagnostic",
                "category": "latency"
            }]
        });

        let snapshot: SerializableExecutionPlanMetricsSet =
            serde_json::from_value(json.clone()).unwrap();
        let metric = &snapshot.metrics[0];
        assert!(matches!(
            metric.metric_type,
            SerializableMetricType::Unknown(ref value) if value == "diagnostic"
        ));
        assert!(matches!(
            metric.category,
            Some(SerializableMetricCategory::Unknown(ref value)) if value == "latency"
        ));
        assert!(matches!(
            metric.value,
            SerializableMetricValue::Unknown(ref value)
                if value.get("type") == Some(&Value::String("histogram".to_owned()))
        ));

        // Unknown data is retained exactly while it remains in the wire model.
        assert_eq!(serde_json::to_value(&snapshot).unwrap(), json);

        // DataFusion cannot represent unknown variants, so use a visible custom
        // metric, retain its numeric projection, and safely omit its category.
        let metrics = snapshot.into_metrics_set();
        let metric = metrics.iter().next().unwrap();
        assert_eq!(metric.metric_type(), MetricType::Dev);
        assert_eq!(metric.metric_category(), None);
        match metric.value() {
            MetricValue::Custom { name, value } => {
                assert_eq!(name, "scan_latency");
                assert_eq!(value.as_usize(), 23);
                let display: Value = serde_json::from_str(&value.to_string()).unwrap();
                assert_eq!(display, json["metrics"][0]["value"]);
            }
            value => panic!("expected custom fallback, got {value:?}"),
        }
    }

    #[test]
    fn extra_fields_on_known_metrics_are_ignored() {
        let value: SerializableMetricValue = serde_json::from_value(serde_json::json!({
            "type": "count",
            "name": "files",
            "value": 4,
            "future_field": true
        }))
        .unwrap();

        assert!(matches!(
            value,
            SerializableMetricValue::Known(KnownMetricValue::Count {
                ref name,
                value: 4
            }) if name == "files"
        ));
    }

    #[test]
    fn malformed_known_metric_is_still_an_error() {
        let error = serde_json::from_value::<SerializableMetricValue>(serde_json::json!({
            "type": "count",
            "name": "missing_value"
        }))
        .unwrap_err();

        assert!(error.to_string().contains("missing field `value`"));
    }

    #[test]
    fn custom_metric_display_repro() {
        let snapshot: SerializableExecutionPlanMetricsSet =
            serde_json::from_value(serde_json::json!({"metrics": [
                {"partition": 0, "metric_type": "summary",
                 "value": {"type": "custom", "name": "custom_count", "display": "10", "value": 10}},
                {"partition": 1, "metric_type": "summary",
                 "value": {"type": "custom", "name": "custom_count", "display": "20", "value": 20}}
            ]}))
            .unwrap();
        let metrics = snapshot.into_metrics_set().aggregate_by_name();
        assert_eq!(metrics.iter().next().unwrap().value().as_usize(), 30);
        assert_eq!(metrics.to_string(), "custom_count=30");
    }

    #[test]
    fn custom_metric_preserves_native_aggregate_display() {
        let metrics = ExecutionPlanMetricsSet::new();
        for (partition, value) in [10, 20].into_iter().enumerate() {
            metrics.register(Arc::new(
                Metric::new(
                    MetricValue::Custom {
                        name: "custom_count".into(),
                        value: Arc::new(NativeCustomCounter::new(value)),
                    },
                    Some(partition),
                )
                .with_type(MetricType::Summary),
            ));
        }

        assert_eq!(
            metrics.clone_inner().aggregate_by_name().to_string(),
            "custom_count=30 rows"
        );

        let json =
            serde_json::to_string(&SerializableExecutionPlanMetricsSet::new(&metrics)).unwrap();
        let snapshot: SerializableExecutionPlanMetricsSet = serde_json::from_str(&json).unwrap();
        let reconstructed = snapshot.into_metrics_set().aggregate_by_name();

        assert_eq!(reconstructed.iter().next().unwrap().value().as_usize(), 30);
        assert_eq!(reconstructed.to_string(), "custom_count=30 rows");
    }
}
