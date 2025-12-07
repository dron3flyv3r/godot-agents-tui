use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PolicyMetrics {
    reward_mean: Option<f64>,
    reward_min: Option<f64>,
    reward_max: Option<f64>,
    episode_len_mean: Option<f64>,
    completed_episodes: Option<u64>,
    learner_stats: BTreeMap<String, f64>,
    custom_metrics: BTreeMap<String, f64>,
}

impl PolicyMetrics {
    pub fn reward_mean(&self) -> Option<f64> {
        self.reward_mean
    }

    pub fn reward_min(&self) -> Option<f64> {
        self.reward_min
    }

    pub fn reward_max(&self) -> Option<f64> {
        self.reward_max
    }

    pub fn episode_len_mean(&self) -> Option<f64> {
        self.episode_len_mean
    }

    pub fn completed_episodes(&self) -> Option<u64> {
        self.completed_episodes
    }

    pub fn learner_stats(&self) -> &BTreeMap<String, f64> {
        &self.learner_stats
    }

    pub fn custom_metrics(&self) -> &BTreeMap<String, f64> {
        &self.custom_metrics
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricSample {
    timestamp: Option<String>,
    date: Option<String>,
    trial_id: Option<String>,
    experiment_id: Option<String>,
    experiment_tag: Option<String>,
    training_iteration: Option<u64>,
    timesteps_total: Option<u64>,
    episodes_total: Option<u64>,
    episodes_this_iter: Option<u64>,
    episode_reward_mean: Option<f64>,
    episode_reward_min: Option<f64>,
    episode_reward_max: Option<f64>,
    episode_len_mean: Option<f64>,
    time_this_iter_s: Option<f64>,
    time_total_s: Option<f64>,
    env_steps_this_iter: Option<u64>,
    env_throughput: Option<f64>,
    num_env_steps_sampled: Option<u64>,
    num_env_steps_trained: Option<u64>,
    num_agent_steps_sampled: Option<u64>,
    num_agent_steps_trained: Option<u64>,
    custom_metrics: BTreeMap<String, f64>,
    policies: BTreeMap<String, PolicyMetrics>,
    checkpoints: Option<u64>,
}

impl MetricSample {
    pub(crate) fn from_value(value: &Value, checkpoint_frequency: u64) -> Option<Self> {
        let kind = value.get("kind").and_then(Value::as_str);
        if let Some(kind_str) = kind {
            if kind_str != "iteration" {
                return None;
            }
        } else if value.get("training_iteration").and_then(value_as_u64).is_none() {
            // Accept lines without "kind" as long as they include iteration info.
            return None;
        }

        let training_iteration = value.get("training_iteration").and_then(value_as_u64);
        let timesteps_total = value.get("timesteps_total").and_then(value_as_u64);
        let checkpoints = training_iteration.and_then(|iteration| {
            if checkpoint_frequency == 0 {
                None
            } else {
                Some(iteration / checkpoint_frequency)
            }
        });

        Some(Self {
            timestamp: value
                .get("timestamp")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            date: value
                .get("date")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            trial_id: value
                .get("trial_id")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            experiment_id: value
                .get("experiment_id")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            experiment_tag: value
                .get("experiment_tag")
                .and_then(Value::as_str)
                .map(|s| s.to_string()),
            training_iteration,
            timesteps_total,
            episodes_total: value.get("episodes_total").and_then(value_as_u64),
            episodes_this_iter: value.get("episodes_this_iter").and_then(value_as_u64),
            episode_reward_mean: value.get("episode_reward_mean").and_then(value_as_f64),
            episode_reward_min: value.get("episode_reward_min").and_then(value_as_f64),
            episode_reward_max: value.get("episode_reward_max").and_then(value_as_f64),
            episode_len_mean: value.get("episode_len_mean").and_then(value_as_f64),
            time_this_iter_s: value.get("time_this_iter_s").and_then(value_as_f64),
            time_total_s: value.get("time_total_s").and_then(value_as_f64),
            env_steps_this_iter: value.get("env_steps_this_iter").and_then(value_as_u64),
            env_throughput: value.get("env_throughput").and_then(value_as_f64),
            num_env_steps_sampled: value.get("num_env_steps_sampled").and_then(value_as_u64),
            num_env_steps_trained: value.get("num_env_steps_trained").and_then(value_as_u64),
            num_agent_steps_sampled: value.get("num_agent_steps_sampled").and_then(value_as_u64),
            num_agent_steps_trained: value.get("num_agent_steps_trained").and_then(value_as_u64),
            custom_metrics: value
                .get("custom_metrics")
                .and_then(value_as_f64_map)
                .unwrap_or_default(),
            policies: value
                .get("policies")
                .and_then(value_as_policy_map)
                .unwrap_or_default(),
            checkpoints,
        })
    }

    pub fn timestamp(&self) -> Option<&str> {
        self.timestamp.as_deref()
    }

    pub fn trial_id(&self) -> Option<&str> {
        self.trial_id.as_deref()
    }

    pub fn training_iteration(&self) -> Option<u64> {
        self.training_iteration
    }

    pub fn timesteps_total(&self) -> Option<u64> {
        self.timesteps_total
    }

    pub fn episode_reward_mean(&self) -> Option<f64> {
        self.episode_reward_mean
    }

    pub fn episode_reward_min(&self) -> Option<f64> {
        self.episode_reward_min
    }

    pub fn episode_reward_max(&self) -> Option<f64> {
        self.episode_reward_max
    }

    pub fn episodes_total(&self) -> Option<u64> {
        self.episodes_total
    }

    pub fn episodes_this_iter(&self) -> Option<u64> {
        self.episodes_this_iter
    }

    pub fn episode_len_mean(&self) -> Option<f64> {
        self.episode_len_mean
    }

    pub fn time_this_iter_s(&self) -> Option<f64> {
        self.time_this_iter_s
    }

    pub fn time_total_s(&self) -> Option<f64> {
        self.time_total_s
    }

    pub fn env_steps_this_iter(&self) -> Option<u64> {
        self.env_steps_this_iter
    }

    pub fn env_throughput(&self) -> Option<f64> {
        self.env_throughput
    }

    pub fn num_env_steps_sampled(&self) -> Option<u64> {
        self.num_env_steps_sampled
    }

    pub fn num_env_steps_trained(&self) -> Option<u64> {
        self.num_env_steps_trained
    }

    pub fn num_agent_steps_sampled(&self) -> Option<u64> {
        self.num_agent_steps_sampled
    }

    pub fn num_agent_steps_trained(&self) -> Option<u64> {
        self.num_agent_steps_trained
    }

    pub fn custom_metrics(&self) -> &BTreeMap<String, f64> {
        &self.custom_metrics
    }

    pub fn policies(&self) -> &BTreeMap<String, PolicyMetrics> {
        &self.policies
    }

    pub fn checkpoints(&self) -> Option<u64> {
        self.checkpoints
    }

    pub(crate) fn set_time_total_s(&mut self, value: f64) {
        self.time_total_s = Some(value);
    }

    pub(crate) fn set_time_this_iter_s(&mut self, value: f64) {
        self.time_this_iter_s = Some(value);
    }
}

impl Default for MetricSample {
    fn default() -> Self {
        Self {
            timestamp: None,
            date: None,
            trial_id: None,
            experiment_id: None,
            experiment_tag: None,
            training_iteration: None,
            timesteps_total: None,
            episodes_total: None,
            episodes_this_iter: None,
            episode_reward_mean: None,
            episode_reward_min: None,
            episode_reward_max: None,
            episode_len_mean: None,
            time_this_iter_s: None,
            time_total_s: None,
            env_steps_this_iter: None,
            env_throughput: None,
            num_env_steps_sampled: None,
            num_env_steps_trained: None,
            num_agent_steps_sampled: None,
            num_agent_steps_trained: None,
            custom_metrics: BTreeMap::new(),
            policies: BTreeMap::new(),
            checkpoints: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ChartMetricKind {
    EpisodeRewardMean,
    EpisodeLenMean,
    EnvThroughput,
    CustomMetric(String),
    PolicyRewardMean,
    PolicyEpisodeLenMean,
    PolicyLearnerStat(String),
    PolicyCustomMetric(String),
    AllPoliciesRewardMean,
    AllPoliciesEpisodeLenMean,
    AllPoliciesLearnerStat(String),
}

#[derive(Debug, Clone)]
pub struct ChartMetricOption {
    label: String,
    kind: ChartMetricKind,
    policy_id: Option<String>,
}

impl ChartMetricOption {
    pub(crate) fn new(label: impl Into<String>, kind: ChartMetricKind) -> Self {
        Self {
            label: label.into(),
            kind,
            policy_id: None,
        }
    }

    pub(crate) fn with_policy(
        label: impl Into<String>,
        policy_id: impl Into<String>,
        kind: ChartMetricKind,
    ) -> Self {
        Self {
            label: label.into(),
            kind,
            policy_id: Some(policy_id.into()),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn policy_id(&self) -> Option<&str> {
        self.policy_id.as_deref()
    }

    pub fn kind(&self) -> &ChartMetricKind {
        &self.kind
    }
}

#[derive(Debug, Clone)]
pub struct ChartData {
    pub label: String,
    pub points: Vec<(f64, f64)>,
}

fn value_as_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Number(number) => {
            if let Some(u) = number.as_u64() {
                Some(u)
            } else if let Some(i) = number.as_i64() {
                (i >= 0).then(|| i as u64)
            } else {
                number.as_f64().and_then(|f| {
                    if f >= 0.0 {
                        Some(f.trunc() as u64)
                    } else {
                        None
                    }
                })
            }
        }
        Value::String(string) => string.parse::<u64>().ok(),
        _ => None,
    }
}

fn value_as_f64_map(value: &Value) -> Option<BTreeMap<String, f64>> {
    let object = value.as_object()?;
    let mut map = BTreeMap::new();
    for (key, val) in object {
        if let Some(number) = value_as_f64(val) {
            map.insert(key.clone(), number);
        }
    }
    Some(map)
}

fn value_as_policy_map(value: &Value) -> Option<BTreeMap<String, PolicyMetrics>> {
    let object = value.as_object()?;
    let mut map = BTreeMap::new();
    for (policy_id, metrics) in object {
        if let Some(policy_metrics) = value_as_policy_metrics(metrics) {
            map.insert(policy_id.clone(), policy_metrics);
        }
    }
    Some(map)
}

fn value_as_policy_metrics(value: &Value) -> Option<PolicyMetrics> {
    let object = value.as_object()?;
    let mut policy = PolicyMetrics::default();
    if let Some(mean) = object.get("reward_mean").and_then(value_as_f64) {
        policy.reward_mean = Some(mean);
    }
    if let Some(min) = object.get("reward_min").and_then(value_as_f64) {
        policy.reward_min = Some(min);
    }
    if let Some(max) = object.get("reward_max").and_then(value_as_f64) {
        policy.reward_max = Some(max);
    }
    if let Some(len) = object.get("episode_len_mean").and_then(value_as_f64) {
        policy.episode_len_mean = Some(len);
    }
    if let Some(completed) = object.get("completed_episodes").and_then(value_as_u64) {
        policy.completed_episodes = Some(completed);
    }
    if let Some(custom) = object.get("custom_metrics").and_then(value_as_f64_map) {
        policy.custom_metrics = custom;
    }
    if let Some(learner) = object.get("learner").and_then(value_as_f64_map_recursive) {
        policy.learner_stats = learner;
    }
    Some(policy)
}

fn value_as_f64_map_recursive(value: &Value) -> Option<BTreeMap<String, f64>> {
    let mut map = BTreeMap::new();
    collect_numeric_values("", value, &mut map);
    if map.is_empty() {
        None
    } else {
        Some(map)
    }
}

fn collect_numeric_values(prefix: &str, value: &Value, out: &mut BTreeMap<String, f64>) {
    match value {
        Value::Object(obj) => {
            for (key, val) in obj {
                let new_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                collect_numeric_values(&new_prefix, val, out);
            }
        }
        Value::Array(items) => {
            for (index, item) in items.iter().enumerate() {
                let new_prefix = if prefix.is_empty() {
                    format!("[{index}]")
                } else {
                    format!("{prefix}[{index}]")
                };
                collect_numeric_values(&new_prefix, item, out);
            }
        }
        _ => {
            if let Some(number) = value_as_f64(value) {
                let key = if prefix.is_empty() {
                    "value".to_string()
                } else {
                    prefix.to_string()
                };
                out.insert(key, number);
            }
        }
    }
}

fn value_as_f64(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .or_else(|| number.as_i64().map(|i| i as f64)),
        Value::String(string) => string.parse::<f64>().ok(),
        _ => None,
    }
}
