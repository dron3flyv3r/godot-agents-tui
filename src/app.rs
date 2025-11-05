use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::ops::Index;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Instant;

use color_eyre::{
    eyre::{bail, WrapErr},
    Result,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::project::{ProjectInfo, ProjectManager};

const TRAINING_BUFFER_LIMIT: usize = 500;
const EXPORT_BUFFER_LIMIT: usize = 500;
const METRIC_PREFIX: &str = "@METRIC ";
const TRAINING_METRIC_HISTORY_LIMIT: usize = 2000;

#[derive(Debug, Clone, Default)]
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

#[derive(Debug, Clone)]
pub struct MetricSample {
    timestamp: Option<String>,
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
    fn from_value(value: &Value, checkpoint_frequency: u64) -> Option<Self> {
        let kind = value.get("kind").and_then(Value::as_str);
        if kind != Some("iteration") {
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
            checkpoints
        })
    }

    pub fn timestamp(&self) -> Option<&str> {
        self.timestamp.as_deref()
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
    // Multi-policy overlays
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
    fn new(label: impl Into<String>, kind: ChartMetricKind) -> Self {
        Self {
            label: label.into(),
            kind,
            policy_id: None,
        }
    }

    fn with_policy(
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
const TRAINING_CONFIG_FILENAME: &str = "training_config.json";
const EXPORT_CONFIG_FILENAME: &str = "export_config.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    SingleAgent, // Stable Baselines 3
    MultiAgent,  // RLlib
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RllibStopMode {
    TimeSeconds,
    Timesteps,
}

impl Default for RllibStopMode {
    fn default() -> Self {
        Self::TimeSeconds
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub mode: TrainingMode,
    pub env_path: String,
    pub timesteps: u64,
    pub experiment_name: String,
    // SB3 specific
    pub sb3_policy_layers: Vec<usize>,
    pub sb3_viz: bool,
    pub sb3_speedup: u32,
    pub sb3_n_parallel: u32,
    pub sb3_learning_rate: f64,
    pub sb3_batch_size: u32,
    pub sb3_n_steps: u32,
    pub sb3_gamma: f64,
    pub sb3_gae_lambda: f64,
    pub sb3_ent_coef: f64,
    pub sb3_clip_range: f64,
    pub sb3_vf_coef: f64,
    pub sb3_max_grad_norm: f64,
    // RLlib specific
    pub rllib_config_file: String,
    pub rllib_show_window: bool,
    pub rllib_num_workers: u32,
    pub rllib_num_envs_per_worker: u32,
    pub rllib_train_batch_size: u32,
    pub rllib_sgd_minibatch_size: u32,
    pub rllib_num_sgd_iter: u32,
    pub rllib_lr: f64,
    pub rllib_gamma: f64,
    pub rllib_lambda: f64,
    pub rllib_clip_param: f64,
    pub rllib_entropy_coeff: f64,
    pub rllib_vf_loss_coeff: f64,
    pub rllib_grad_clip: f64,
    pub rllib_framework: String,
    pub rllib_activation: String,
    pub rllib_batch_mode: String,
    pub rllib_rollout_fragment_length: u32,
    pub rllib_fcnet_hiddens: Vec<usize>,
    pub rllib_checkpoint_frequency: u32,
    pub rllib_resume_from: String,
    pub rllib_stop_mode: RllibStopMode,
    pub rllib_stop_time_seconds: u64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::SingleAgent,
            env_path: String::new(),
            timesteps: 1_000_000,
            experiment_name: "training".to_string(),
            sb3_policy_layers: vec![256, 256],
            sb3_viz: false,
            sb3_speedup: 1,
            sb3_n_parallel: 1,
            sb3_learning_rate: 3e-4,
            sb3_batch_size: 64,
            sb3_n_steps: 2048,
            sb3_gamma: 0.99,
            sb3_gae_lambda: 0.95,
            sb3_ent_coef: 0.01,
            sb3_clip_range: 0.2,
            sb3_vf_coef: 0.5,
            sb3_max_grad_norm: 0.5,
            rllib_config_file: "rllib_config.yaml".to_string(),
            rllib_show_window: false,
            rllib_num_workers: 4,
            rllib_num_envs_per_worker: 1,
            rllib_train_batch_size: 4000,
            rllib_sgd_minibatch_size: 128,
            rllib_num_sgd_iter: 30,
            rllib_lr: 3e-4,
            rllib_gamma: 0.99,
            rllib_lambda: 0.95,
            rllib_clip_param: 0.2,
            rllib_entropy_coeff: 0.01,
            rllib_vf_loss_coeff: 0.5,
            rllib_grad_clip: 0.5,
            rllib_framework: "torch".to_string(),
            rllib_activation: "relu".to_string(),
            rllib_batch_mode: "truncate_episodes".to_string(),
            rllib_rollout_fragment_length: 200,
            rllib_fcnet_hiddens: vec![256, 256],
            rllib_checkpoint_frequency: 20,
            rllib_resume_from: String::new(),
            rllib_stop_mode: RllibStopMode::TimeSeconds,
            rllib_stop_time_seconds: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabId {
    Home,
    Train,
    Metrics,
    Export,
    Settings,
}

#[derive(Debug, Clone)]
pub struct Tab {
    pub title: &'static str,
    pub id: TabId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusKind {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
pub struct StatusMessage {
    pub text: String,
    pub kind: StatusKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputMode {
    Normal,
    CreatingProject,
    EditingConfig,
    EditingAdvancedConfig,
    AdvancedConfig,
    BrowsingFiles,
    Help,
    ConfirmQuit,
    EditingExport,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricsFocus {
    History,
    Summary,
    Policies,
    Chart,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigField {
    EnvPath,
    Timesteps,
    ExperimentName,
    Sb3Speedup,
    Sb3NParallel,
    Sb3Viz,
    Sb3PolicyLayers,
    Sb3LearningRate,
    Sb3BatchSize,
    Sb3NSteps,
    Sb3Gamma,
    Sb3GaeLambda,
    Sb3EntCoef,
    Sb3ClipRange,
    Sb3VfCoef,
    Sb3MaxGradNorm,
    RllibConfigFile,
    RllibShowWindow,
    RllibNumWorkers,
    RllibNumEnvWorkers,
    RllibTrainBatchSize,
    RllibSgdMinibatchSize,
    RllibNumSgdIter,
    RllibLr,
    RllibGamma,
    RllibLambda,
    RllibClipParam,
    RllibEntropyCoeff,
    RllibVfLossCoeff,
    RllibGradClip,
    RllibFramework,
    RllibActivation,
    RllibBatchMode,
    RllibRolloutFragmentLength,
    RllibFcnetHiddens,
    RllibCheckpointFrequency,
    RllibResumeFrom,
    RllibStopMode,
    RllibStopTimeSeconds,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExportMode {
    StableBaselines3,
    Rllib,
}

impl Default for ExportMode {
    fn default() -> Self {
        ExportMode::StableBaselines3
    }
}

impl ExportMode {
    pub fn label(self) -> &'static str {
        match self {
            ExportMode::StableBaselines3 => "Stable-Baselines3",
            ExportMode::Rllib => "RLlib",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportField {
    Sb3ModelPath,
    Sb3OutputPath,
    Sb3Algo,
    Sb3Opset,
    Sb3IrVersion,
    Sb3UseObsArray,
    Sb3SkipVerify,
    RllibCheckpointPath,
    RllibCheckpointNumber,
    RllibOutputDir,
    RllibPolicyId,
    RllibOpset,
    RllibIrVersion,
    RllibMultiagent,
}

impl ExportField {
    pub fn label(self) -> &'static str {
        match self {
            ExportField::Sb3ModelPath => "SB3 Model (.zip)",
            ExportField::Sb3OutputPath => "SB3 Output (.onnx)",
            ExportField::Sb3Algo => "SB3 Algorithm",
            ExportField::Sb3Opset => "SB3 Opset",
            ExportField::Sb3IrVersion => "SB3 IR Version",
            ExportField::Sb3UseObsArray => "SB3 Use Obs Array",
            ExportField::Sb3SkipVerify => "SB3 Skip Verification",
            ExportField::RllibCheckpointPath => "RLlib Checkpoint Dir",
            ExportField::RllibCheckpointNumber => "RLlib Checkpoint Number",
            ExportField::RllibOutputDir => "RLlib Output Dir",
            ExportField::RllibPolicyId => "RLlib Policy ID",
            ExportField::RllibOpset => "RLlib Opset",
            ExportField::RllibIrVersion => "RLlib IR Version",
            ExportField::RllibMultiagent => "RLlib Multi-agent",
        }
    }

    pub fn is_toggle(self) -> bool {
        matches!(
            self,
            ExportField::Sb3UseObsArray | ExportField::Sb3SkipVerify | ExportField::RllibMultiagent
        )
    }

    pub fn uses_file_browser(self) -> bool {
        matches!(
            self,
            ExportField::Sb3ModelPath
                | ExportField::Sb3OutputPath
                | ExportField::RllibCheckpointPath
                | ExportField::RllibOutputDir
        )
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBrowserTarget {
    Config(ConfigField),
    Export(ExportField),
}

#[derive(Debug, Clone)]
pub enum FileBrowserKind {
    Directory {
        allow_create: bool,
        require_checkpoints: bool,
    },
    ExistingFile {
        extensions: Vec<String>,
    },
    OutputFile {
        extension: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileBrowserState {
    Browsing,
    NamingFolder,
    NamingFile,
}

#[derive(Debug, Clone)]
pub enum FileBrowserEntry {
    Parent(PathBuf),
    Directory(PathBuf),
    File(PathBuf),
}

impl FileBrowserEntry {
    pub fn path(&self) -> &Path {
        match self {
            FileBrowserEntry::Parent(path)
            | FileBrowserEntry::Directory(path)
            | FileBrowserEntry::File(path) => path,
        }
    }

    // pub fn is_dir(&self) -> bool {
    //     matches!(
    //         self,
    //         FileBrowserEntry::Parent(_) | FileBrowserEntry::Directory(_)
    //     )
    // }

    pub fn is_parent(&self) -> bool {
        matches!(self, FileBrowserEntry::Parent(_))
    }

    pub fn display_name(&self) -> String {
        if self.is_parent() {
            return String::from("[..]");
        }

        self.path()
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.path().display().to_string())
    }

    // pub fn extension(&self) -> Option<&str> {
    //     if matches!(self, FileBrowserEntry::File(_)) {
    //         self.path().extension().and_then(|s| s.to_str())
    //     } else {
    //         None
    //     }
    // }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportConfig {
    pub sb3_model_path: String,
    pub sb3_output_path: String,
    pub sb3_algo: String,
    pub sb3_opset: u32,
    pub sb3_ir_version: u32,
    pub sb3_use_obs_array: bool,
    pub sb3_skip_verify: bool,
    pub rllib_checkpoint_path: String,
    pub rllib_checkpoint_number: Option<u32>,
    pub rllib_output_dir: String,
    pub rllib_policy_id: String,
    pub rllib_opset: u32,
    pub rllib_ir_version: u32,
    pub rllib_multiagent: bool,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            sb3_model_path: String::new(),
            sb3_output_path: String::new(),
            sb3_algo: String::new(),
            sb3_opset: 13,
            sb3_ir_version: 9,
            sb3_use_obs_array: false,
            sb3_skip_verify: false,
            rllib_checkpoint_path: String::new(),
            rllib_checkpoint_number: None,
            rllib_output_dir: String::new(),
            rllib_policy_id: String::new(),
            rllib_opset: 13,
            rllib_ir_version: 9,
            rllib_multiagent: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct ExportState {
    mode: ExportMode,
    config: ExportConfig,
}

impl Default for ExportState {
    fn default() -> Self {
        Self {
            mode: ExportMode::default(),
            config: ExportConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFocus {
    Fields,
    Output,
}

impl ConfigField {
    pub fn label(self) -> &'static str {
        match self {
            ConfigField::EnvPath => "Environment Path",
            ConfigField::Timesteps => "Timesteps",
            ConfigField::ExperimentName => "Experiment Name",
            ConfigField::Sb3Speedup => "SB3 Speedup",
            ConfigField::Sb3NParallel => "SB3 Parallel Envs",
            ConfigField::Sb3Viz => "SB3 Visualization",
            ConfigField::Sb3PolicyLayers => "SB3 Policy Layers",
            ConfigField::Sb3LearningRate => "SB3 Learning Rate",
            ConfigField::Sb3BatchSize => "SB3 Batch Size",
            ConfigField::Sb3NSteps => "SB3 n_steps",
            ConfigField::Sb3Gamma => "SB3 Gamma",
            ConfigField::Sb3GaeLambda => "SB3 GAE Lambda",
            ConfigField::Sb3EntCoef => "SB3 Entropy Coef",
            ConfigField::Sb3ClipRange => "SB3 Clip Range",
            ConfigField::Sb3VfCoef => "SB3 VF Coef",
            ConfigField::Sb3MaxGradNorm => "SB3 Max Grad Norm",
            ConfigField::RllibConfigFile => "RLlib Config File",
            ConfigField::RllibShowWindow => "RLlib Show Window",
            ConfigField::RllibNumWorkers => "RLlib Workers",
            ConfigField::RllibNumEnvWorkers => "RLlib Envs/Worker",
            ConfigField::RllibTrainBatchSize => "RLlib Train Batch",
            ConfigField::RllibSgdMinibatchSize => "RLlib Minibatch",
            ConfigField::RllibNumSgdIter => "RLlib SGD Iterations",
            ConfigField::RllibLr => "RLlib Learning Rate",
            ConfigField::RllibGamma => "RLlib Gamma",
            ConfigField::RllibLambda => "RLlib Lambda",
            ConfigField::RllibClipParam => "RLlib Clip Param",
            ConfigField::RllibEntropyCoeff => "RLlib Entropy Coef",
            ConfigField::RllibVfLossCoeff => "RLlib VF Loss Coef",
            ConfigField::RllibGradClip => "RLlib Grad Clip",
            ConfigField::RllibFramework => "RLlib Framework",
            ConfigField::RllibActivation => "RLlib Activation",
            ConfigField::RllibBatchMode => "RLlib Batch Mode",
            ConfigField::RllibRolloutFragmentLength => "RLlib Rollout Fragment",
            ConfigField::RllibFcnetHiddens => "RLlib FC Layers",
            ConfigField::RllibCheckpointFrequency => "RLlib Checkpoint Frequency",
            ConfigField::RllibResumeFrom => "RLlib Resume Directory",
            ConfigField::RllibStopMode => "RLlib Stop Mode",
            ConfigField::RllibStopTimeSeconds => "RLlib Time Limit (s)",
            // ConfigField::RllibCheckpointFrequency => {
            //     self.training_config.rllib_checkpoint_frequency.to_string()
            // }
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            ConfigField::EnvPath => {
                "Executable path to the exported Godot environment used for training."
            }
            ConfigField::Timesteps => {
                "Total environment timesteps to run before stopping training."
            }
            ConfigField::ExperimentName => {
                "Name used for experiment folders, checkpoints, and logs."
            }
            ConfigField::Sb3Speedup => "Simulation speed multiplier applied while SB3 is running.",
            ConfigField::Sb3NParallel => {
                "Number of parallel environments to launch for SB3 training."
            }
            ConfigField::Sb3Viz => "Enable to render the Godot window during SB3 training.",
            ConfigField::Sb3PolicyLayers => {
                "Comma-separated hidden layer sizes for the SB3 policy network."
            }
            ConfigField::Sb3LearningRate => "Learning rate used by the SB3 optimizer.",
            ConfigField::Sb3BatchSize => "Batch size sampled when performing SB3 updates.",
            ConfigField::Sb3NSteps => "Steps collected per rollout before each SB3 update.",
            ConfigField::Sb3Gamma => "Discount factor (gamma) applied to future rewards in SB3.",
            ConfigField::Sb3GaeLambda => {
                "GAE lambda balancing bias and variance for SB3 advantages."
            }
            ConfigField::Sb3EntCoef => "Entropy bonus encouraging exploration in SB3.",
            ConfigField::Sb3ClipRange => "PPO clip range limiting SB3 policy updates.",
            ConfigField::Sb3VfCoef => "Weight applied to the SB3 value function loss.",
            ConfigField::Sb3MaxGradNorm => "Maximum gradient norm before SB3 clips updates.",
            ConfigField::RllibConfigFile => {
                "Relative path where the generated RLlib config YAML is saved."
            }
            ConfigField::RllibShowWindow => {
                "Show the Godot window during RLlib training when enabled."
            }
            ConfigField::RllibNumWorkers => "Number of RLlib rollout workers to spawn.",
            ConfigField::RllibNumEnvWorkers => "Environments created per RLlib worker.",
            ConfigField::RllibTrainBatchSize => {
                "Total batch size RLlib gathers before each optimization step."
            }
            ConfigField::RllibSgdMinibatchSize => "Minibatch size used inside RLlib's SGD loop.",
            ConfigField::RllibNumSgdIter => "How many SGD passes RLlib runs per collected batch.",
            ConfigField::RllibLr => "Learning rate for the RLlib optimizer.",
            ConfigField::RllibGamma => {
                "Discount factor (gamma) applied to future rewards in RLlib."
            }
            ConfigField::RllibLambda => "GAE lambda used for RLlib advantage estimates.",
            ConfigField::RllibClipParam => {
                "PPO clip parameter limiting RLlib policy update magnitude."
            }
            ConfigField::RllibEntropyCoeff => "Entropy bonus encouraging exploration in RLlib.",
            ConfigField::RllibVfLossCoeff => "Weight applied to RLlib's value function loss.",
            ConfigField::RllibGradClip => "Maximum gradient norm before RLlib clips updates.",
            ConfigField::RllibFramework => "Deep learning backend RLlib should use (torch or tf).",
            ConfigField::RllibActivation => "Activation function for RLlib fully connected layers.",
            ConfigField::RllibBatchMode => {
                "Choose between episode or fragment based RLlib batching."
            }
            ConfigField::RllibRolloutFragmentLength => {
                "Steps per rollout fragment collected by each RLlib worker."
            }
            ConfigField::RllibFcnetHiddens => {
                "Comma-separated hidden layer sizes for RLlib's fully connected net."
            }
            ConfigField::RllibCheckpointFrequency => {
                "Number of iterations between RLlib checkpoints."
            }
            ConfigField::RllibResumeFrom => {
                "Optional checkpoint directory to resume RLlib training from."
            }
            ConfigField::RllibStopMode => {
                "Select whether RLlib stops by elapsed time or total timesteps."
            }
            ConfigField::RllibStopTimeSeconds => {
                "Target duration in seconds when using time-based stopping."
            }
        }
    }
}

#[derive(Debug)]
enum TrainingEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

#[derive(Debug)]
enum ExportEvent {
    Line(String),
    Error(String),
    Finished(Option<i32>),
}

pub struct App {
    tabs: Vec<Tab>,
    active_tab_index: usize,
    should_quit: bool,

    project_manager: ProjectManager,
    projects: Vec<ProjectInfo>,
    active_project: Option<ProjectInfo>,
    selected_project: usize,

    input_mode: InputMode,
    project_name_buffer: String,
    config_edit_buffer: String,
    active_config_field: Option<ConfigField>,
    config_return_mode: Option<InputMode>,
    advanced_fields: Vec<ConfigField>,
    advanced_selection: usize,
    file_browser_path: PathBuf,
    file_browser_entries: Vec<FileBrowserEntry>,
    file_browser_selected: usize,
    file_browser_target: Option<FileBrowserTarget>,
    file_browser_kind: FileBrowserKind,
    file_browser_state: FileBrowserState,
    file_browser_default_name: Option<String>,
    file_browser_input: String,

    status: Option<StatusMessage>,

    training_config: TrainingConfig,
    training_config_valid: bool,
    training_output: Vec<String>,
    training_output_scroll: usize,
    training_receiver: Option<Receiver<TrainingEvent>>,
    training_cancel: Option<Sender<()>>,
    training_running: bool,
    training_metrics: Vec<MetricSample>,
    metrics_history_index: usize,
    metrics_chart_index: usize,
    metrics_focus: MetricsFocus,
    metrics_history_scroll: usize,
    metrics_summary_scroll: usize,
    metrics_policies_scroll: usize,
    metrics_policies_expanded: bool,
    metrics_policies_horizontal_scroll: usize,
    metric_timer_start: Option<Instant>,
    metric_last_sample_time: Option<Instant>,

    export_mode: ExportMode,
    export_config: ExportConfig,
    export_focus: ExportFocus,
    export_fields: Vec<ExportField>,
    export_selection: usize,
    export_edit_buffer: String,
    active_export_field: Option<ExportField>,
    export_return_mode: Option<InputMode>,
    export_output: Vec<String>,
    export_output_scroll: usize,
    export_receiver: Option<Receiver<ExportEvent>>,
    export_cancel: Option<Sender<()>>,
    export_running: bool,

    // Python environment check results
    python_sb3_available: Option<bool>,
    python_ray_available: Option<bool>,
}

impl App {
    pub fn new() -> Result<Self> {
        let project_root = default_projects_root()?;
        let project_manager = ProjectManager::new(project_root)?;
        let projects = project_manager.list_projects()?;

        let tabs = vec![
            Tab {
                title: "Home",
                id: TabId::Home,
            },
            Tab {
                title: "Train",
                id: TabId::Train,
            },
            Tab {
                title: "Metrics",
                id: TabId::Metrics,
            },
            Tab {
                title: "Export",
                id: TabId::Export,
            },
            Tab {
                title: "Settings",
                id: TabId::Settings,
            },
        ];

        let mut app = Self {
            tabs,
            active_tab_index: 0,
            should_quit: false,
            project_manager,
            projects,
            active_project: None,
            selected_project: 0,
            input_mode: InputMode::Normal,
            project_name_buffer: String::new(),
            config_edit_buffer: String::new(),
            active_config_field: None,
            config_return_mode: None,
            advanced_fields: Vec::new(),
            advanced_selection: 0,
            file_browser_path: std::env::current_dir().unwrap_or_default(),
            file_browser_entries: Vec::new(),
            file_browser_selected: 0,
            file_browser_target: None,
            file_browser_kind: FileBrowserKind::Directory {
                allow_create: true,
                require_checkpoints: false,
            },
            file_browser_state: FileBrowserState::Browsing,
            file_browser_default_name: None,
            file_browser_input: String::new(),
            status: None,
            training_config: TrainingConfig::default(),
            training_config_valid: false,
            training_output: Vec::new(),
            training_output_scroll: 0,
            training_receiver: None,
            training_running: false,
            training_cancel: None,
            training_metrics: Vec::new(),
            metrics_history_index: 0,
            metrics_chart_index: 0,
            metrics_focus: MetricsFocus::History,
            metrics_history_scroll: 0,
            metrics_summary_scroll: 0,
            metrics_policies_scroll: 0,
            metrics_policies_expanded: false,
            metrics_policies_horizontal_scroll: 0,
            metric_timer_start: None,
            metric_last_sample_time: None,

            export_mode: ExportMode::StableBaselines3,
            export_config: ExportConfig::default(),
            export_focus: ExportFocus::Fields,
            export_fields: Vec::new(),
            export_selection: 0,
            export_edit_buffer: String::new(),
            active_export_field: None,
            export_return_mode: None,
            export_output: Vec::new(),
            export_output_scroll: 0,
            export_receiver: None,
            export_cancel: None,
            export_running: false,

            python_sb3_available: None,
            python_ray_available: None,
        };

        app.ensure_selection_valid();
        app.rebuild_export_fields();
        app.check_python_environment();

        Ok(app)
    }

    pub fn tabs(&self) -> &[Tab] {
        &self.tabs
    }

    pub fn active_index(&self) -> usize {
        self.active_tab_index
    }

    pub fn active_tab(&self) -> &Tab {
        &self.tabs[self.active_tab_index]
    }

    pub fn activate(&mut self, tab: TabId) {
        if let Some(index) = self.tabs.iter().position(|t| t.id == tab) {
            self.active_tab_index = index;
        }
    }

    pub fn next_tab(&mut self) {
        self.active_tab_index = (self.active_tab_index + 1) % self.tabs.len();
    }

    pub fn previous_tab(&mut self) {
        if self.active_tab_index == 0 {
            self.active_tab_index = self.tabs.len() - 1;
        } else {
            self.active_tab_index -= 1;
        }
    }

    pub fn should_quit(&self) -> bool {
        self.should_quit
    }

    // pub fn quit(&mut self) {
    //     self.should_quit = true;
    // }

    pub fn projects(&self) -> &[ProjectInfo] {
        &self.projects
    }

    pub fn selected_project_index(&self) -> Option<usize> {
        if self.projects.is_empty() {
            None
        } else {
            Some(self.selected_project)
        }
    }

    pub fn selected_project(&self) -> Option<&ProjectInfo> {
        self.selected_project_index()
            .and_then(|index| self.projects.get(index))
    }

    pub fn active_project(&self) -> Option<&ProjectInfo> {
        self.active_project.as_ref()
    }

    pub fn input_mode(&self) -> InputMode {
        self.input_mode
    }

    pub fn project_name_buffer(&self) -> &str {
        &self.project_name_buffer
    }

    pub fn status(&self) -> Option<&StatusMessage> {
        self.status.as_ref()
    }

    pub fn python_sb3_available(&self) -> Option<bool> {
        self.python_sb3_available
    }

    pub fn python_ray_available(&self) -> Option<bool> {
        self.python_ray_available
    }

    fn check_python_environment(&mut self) {
        let python_cmd = determine_python_command();
        let base_dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        
        let check_script = match find_script(&base_dir, "check_py_env.py") {
            Ok(script) => script,
            Err(_) => {
                // Script not found, mark as unknown
                self.python_sb3_available = None;
                self.python_ray_available = None;
                return;
            }
        };

        let output = Command::new(&python_cmd)
            .arg(&check_script)
            .output();

        match output {
            Ok(result) => {
                match result.status.code() {
                    Some(0) => {
                        // Both available
                        self.python_sb3_available = Some(true);
                        self.python_ray_available = Some(true);
                    }
                    Some(2) => {
                        // Only SB3 available
                        self.python_sb3_available = Some(true);
                        self.python_ray_available = Some(false);
                    }
                    Some(3) => {
                        // Only Ray available
                        self.python_sb3_available = Some(false);
                        self.python_ray_available = Some(true);
                    }
                    Some(4) => {
                        // Neither available
                        self.python_sb3_available = Some(false);
                        self.python_ray_available = Some(false);
                    }
                    _ => {
                        // Unknown error
                        self.python_sb3_available = None;
                        self.python_ray_available = None;
                    }
                }
            }
            Err(_) => {
                // Failed to run check
                self.python_sb3_available = None;
                self.python_ray_available = None;
            }
        }
    }

    pub fn training_output(&self) -> &[String] {
        &self.training_output
    }

    pub fn training_output_scroll(&self) -> usize {
        self.training_output_scroll
    }

    pub fn scroll_training_output_up(&mut self, lines: usize) {
        if self.training_output.is_empty() {
            self.training_output_scroll = 0;
            return;
        }
        let max_offset = self.training_output.len().saturating_sub(1);
        self.training_output_scroll = self
            .training_output_scroll
            .saturating_add(lines)
            .min(max_offset);
    }

    pub fn scroll_training_output_down(&mut self, lines: usize) {
        if lines >= self.training_output_scroll {
            self.training_output_scroll = 0;
        } else {
            self.training_output_scroll -= lines;
        }
    }

    pub fn reset_training_output_scroll(&mut self) {
        self.training_output_scroll = 0;
    }

    pub fn training_metrics_history(&self) -> &[MetricSample] {
        &self.training_metrics
    }

    pub fn metrics_history_selected_index(&self) -> usize {
        if self.training_metrics.is_empty() {
            0
        } else {
            self.metrics_history_index
                .min(self.training_metrics.len().saturating_sub(1))
        }
    }

    pub fn metrics_sample_at(&self, offset_from_latest: usize) -> Option<&MetricSample> {
        if self.training_metrics.is_empty() {
            return None;
        }
        let len = self.training_metrics.len();
        if offset_from_latest >= len {
            None
        } else {
            self.training_metrics.get(len - 1 - offset_from_latest)
        }
    }

    pub fn selected_metric_sample(&self) -> Option<&MetricSample> {
        let index = self.metrics_history_selected_index();
        self.metrics_sample_at(index)
    }

    pub fn metrics_history_move_newer(&mut self) {
        if self.metrics_history_index > 0 {
            self.metrics_history_index -= 1;
        }
    }

    pub fn metrics_history_move_older(&mut self) {
        if self.metrics_history_index + 1 < self.training_metrics.len() {
            self.metrics_history_index += 1;
        }
    }

    pub fn metrics_history_page_newer(&mut self, count: usize) {
        if count >= self.metrics_history_index {
            self.metrics_history_index = 0;
        } else {
            self.metrics_history_index -= count;
        }
    }

    pub fn metrics_history_page_older(&mut self, count: usize) {
        if self.training_metrics.is_empty() {
            return;
        }
        let max_index = self.training_metrics.len() - 1;
        let new_index = self.metrics_history_index.saturating_add(count);
        self.metrics_history_index = new_index.min(max_index);
    }

    pub fn metrics_history_to_latest(&mut self) {
        self.metrics_history_index = 0;
    }

    pub fn metrics_history_to_oldest(&mut self) {
        if !self.training_metrics.is_empty() {
            self.metrics_history_index = self.training_metrics.len() - 1;
        }
    }

    pub fn available_chart_metrics(&self) -> Vec<ChartMetricOption> {
        let mut options = Vec::new();
        options.push(ChartMetricOption::new(
            "Episode reward mean",
            ChartMetricKind::EpisodeRewardMean,
        ));
        options.push(ChartMetricOption::new(
            "Episode length mean",
            ChartMetricKind::EpisodeLenMean,
        ));

        if let Some(latest) = self.latest_training_metric() {
            if latest.env_throughput().is_some() || latest.env_steps_this_iter().is_some() {
                options.push(ChartMetricOption::new(
                    "Env throughput (steps/s)",
                    ChartMetricKind::EnvThroughput,
                ));
            }

            for (name, _) in latest.custom_metrics().iter() {
                options.push(ChartMetricOption::new(
                    format!("Custom metric: {name}"),
                    ChartMetricKind::CustomMetric(name.clone()),
                ));
            }

            // Add overlay options if there are multiple policies
            let num_policies = latest.policies().len();
            if num_policies > 1 {
                options.push(ChartMetricOption::new(
                    format!("All Policies - Reward ({num_policies} policies)"),
                    ChartMetricKind::AllPoliciesRewardMean,
                ));
                options.push(ChartMetricOption::new(
                    format!("All Policies - Episode Length ({num_policies} policies)"),
                    ChartMetricKind::AllPoliciesEpisodeLenMean,
                ));
                
                // Collect all unique learner stat keys
                let mut learner_stat_keys = std::collections::HashSet::new();
                for (_, metrics) in latest.policies() {
                    for key in metrics.learner_stats().keys() {
                        learner_stat_keys.insert(key.clone());
                    }
                }
                
                // Add overlay options for each learner stat
                for key in learner_stat_keys {
                    options.push(ChartMetricOption::new(
                        format!("All Policies - {} ({num_policies} policies)", key),
                        ChartMetricKind::AllPoliciesLearnerStat(key),
                    ));
                }
            }

            for (policy_id, metrics) in latest.policies() {
                if metrics.reward_mean().is_some() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} reward mean"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyRewardMean,
                    ));
                }
                if metrics.episode_len_mean().is_some() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} episode length mean"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyEpisodeLenMean,
                    ));
                }
                for (key, _) in metrics.learner_stats().iter() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} {key}"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyLearnerStat(key.clone()),
                    ));
                }
                for (key, _) in metrics.custom_metrics().iter() {
                    options.push(ChartMetricOption::with_policy(
                        format!("{policy_id} custom {key}"),
                        policy_id.clone(),
                        ChartMetricKind::PolicyCustomMetric(key.clone()),
                    ));
                }
            }
        }

        options
    }

    fn ensure_chart_metric_index(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
        } else if self.metrics_chart_index >= options.len() {
            self.metrics_chart_index = options.len() - 1;
        }
    }

    pub fn current_chart_metric(&self) -> Option<ChartMetricOption> {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            None
        } else {
            let index = self
                .metrics_chart_index
                .min(options.len().saturating_sub(1));
            Some(options[index].clone())
        }
    }

    pub fn cycle_chart_metric_next(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
            return;
        }
        self.metrics_chart_index = (self.metrics_chart_index + 1) % options.len();
    }

    pub fn cycle_chart_metric_previous(&mut self) {
        let options = self.available_chart_metrics();
        if options.is_empty() {
            self.metrics_chart_index = 0;
            return;
        }
        if self.metrics_chart_index == 0 {
            self.metrics_chart_index = options.len() - 1;
        } else {
            self.metrics_chart_index -= 1;
        }
    }

    pub fn chart_data(&self, max_points: usize) -> Option<ChartData> {
        let metric = self.current_chart_metric()?;
        if self.training_metrics.is_empty() {
            return None;
        }

        let len = self.training_metrics.len();
        let start = len.saturating_sub(max_points);
        let mut points = Vec::new();

        for (idx, sample) in self.training_metrics.iter().enumerate().skip(start) {
            if let Some(value) = App::chart_value_for_sample(sample, &metric) {
                let x = sample
                    .training_iteration()
                    .map(|iter| iter as f64)
                    .unwrap_or_else(|| idx as f64);
                points.push((x, value));
            }
        }

        if points.is_empty() {
            return None;
        }

        Some(ChartData {
            label: metric.label().to_string(),
            points,
        })
    }

    pub fn selected_chart_value(&self) -> Option<f64> {
        let sample = self.selected_metric_sample()?;
        let metric = self.current_chart_metric()?;
        App::chart_value_for_sample(sample, &metric)
    }

    fn chart_value_for_sample(sample: &MetricSample, option: &ChartMetricOption) -> Option<f64> {
        match option.kind() {
            ChartMetricKind::EpisodeRewardMean => sample.episode_reward_mean(),
            ChartMetricKind::EpisodeLenMean => sample.episode_len_mean(),
            ChartMetricKind::EnvThroughput => sample.env_throughput().or_else(|| {
                match (sample.env_steps_this_iter(), sample.time_this_iter_s()) {
                    (Some(steps), Some(time)) if time > 0.0 => Some(steps as f64 / time.max(1e-9)),
                    _ => None,
                }
            }),
            ChartMetricKind::CustomMetric(name) => sample.custom_metrics().get(name).copied(),
            ChartMetricKind::PolicyRewardMean => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.reward_mean()),
            ChartMetricKind::PolicyEpisodeLenMean => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.episode_len_mean()),
            ChartMetricKind::PolicyLearnerStat(key) => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.learner_stats().get(key).copied()),
            ChartMetricKind::PolicyCustomMetric(key) => option
                .policy_id()
                .and_then(|id| sample.policies().get(id))
                .and_then(|metrics| metrics.custom_metrics().get(key).copied()),
            // Multi-policy overlays return None for single-value lookups
            ChartMetricKind::AllPoliciesRewardMean
            | ChartMetricKind::AllPoliciesEpisodeLenMean
            | ChartMetricKind::AllPoliciesLearnerStat(_) => None,
        }
    }

    /// Get multi-series chart data for overlay charts
    pub fn chart_multi_series_data(&self, option: &ChartMetricOption) -> Vec<(String, Vec<(f64, f64)>)> {
        let samples = self.training_metrics_history();
        
        match option.kind() {
            ChartMetricKind::AllPoliciesRewardMean => {
                // Collect all policy IDs
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }
                
                // Sort policy IDs
                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b)));
                
                // Build series for each policy
                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.reward_mean())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            ChartMetricKind::AllPoliciesEpisodeLenMean => {
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }
                
                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b)));
                
                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.episode_len_mean())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            ChartMetricKind::AllPoliciesLearnerStat(stat_key) => {
                let mut policy_ids = std::collections::HashSet::new();
                for sample in samples {
                    for id in sample.policies().keys() {
                        policy_ids.insert(id.clone());
                    }
                }
                
                let mut sorted_ids: Vec<_> = policy_ids.into_iter().collect();
                sorted_ids.sort_by(|a, b| crate::ui::alphanumeric_sort_key(a).cmp(&crate::ui::alphanumeric_sort_key(b)));
                
                sorted_ids
                    .into_iter()
                    .map(|policy_id| {
                        let data: Vec<(f64, f64)> = samples
                            .iter()
                            .filter_map(|sample| {
                                let x = sample.training_iteration()? as f64;
                                let y = sample
                                    .policies()
                                    .get(&policy_id)
                                    .and_then(|m| m.learner_stats().get(stat_key).copied())?;
                                Some((x, y))
                            })
                            .collect();
                        (policy_id, data)
                    })
                    .collect()
            }
            // Non-overlay types return empty
            _ => Vec::new(),
        }
    }

    // Metrics panel focus and scrolling
    pub fn metrics_focus(&self) -> MetricsFocus {
        self.metrics_focus
    }

    pub fn metrics_cycle_focus_next(&mut self) {

        self.metrics_focus = match self.metrics_focus {
            MetricsFocus::History => MetricsFocus::Policies,
            MetricsFocus::Policies => MetricsFocus::History,
            MetricsFocus::Chart => MetricsFocus::History,
            MetricsFocus::Summary => MetricsFocus::History
        };

        // self.metrics_focus = match self.metrics_focus {
        //     MetricsFocus::History => MetricsFocus::Summary,
        //     MetricsFocus::Summary => MetricsFocus::Policies,
        //     MetricsFocus::Policies => MetricsFocus::Chart,
        //     MetricsFocus::Chart => MetricsFocus::History,
        // };
    }

    pub fn metrics_cycle_focus_previous(&mut self) {
        self.metrics_focus = match self.metrics_focus {
            MetricsFocus::History => MetricsFocus::Policies,
            MetricsFocus::Policies => MetricsFocus::History,
            MetricsFocus::Chart => MetricsFocus::History,
            MetricsFocus::Summary => MetricsFocus::History
        };


        // self.metrics_focus = match self.metrics_focus {
        //     MetricsFocus::History => MetricsFocus::Chart,
        //     MetricsFocus::Chart => MetricsFocus::Policies,
        //     MetricsFocus::Policies => MetricsFocus::Summary,
        //     MetricsFocus::Summary => MetricsFocus::History,
        // };
    }

    pub fn metrics_summary_scroll(&self) -> usize {
        self.metrics_summary_scroll
    }

    pub fn metrics_policies_scroll(&self) -> usize {
        self.metrics_policies_scroll
    }

    pub fn clamp_metrics_summary_scroll(&mut self, total_lines: usize, visible_lines: usize) {
        let max_scroll = total_lines.saturating_sub(visible_lines);
        self.metrics_summary_scroll = self.metrics_summary_scroll.min(max_scroll);
    }

    pub fn clamp_metrics_policies_scroll(&mut self, total_items: usize, visible_items: usize) {
        let max_scroll = total_items.saturating_sub(visible_items);
        self.metrics_policies_scroll = self.metrics_policies_scroll.min(max_scroll);
    }

    pub fn clamp_metrics_policies_horizontal_scroll(&mut self, total_policies: usize, visible_policies: usize) {
        let max_scroll = total_policies.saturating_sub(visible_policies);
        self.metrics_policies_horizontal_scroll = self.metrics_policies_horizontal_scroll.min(max_scroll);
    }

    pub fn clamp_all_metrics_scrolls(&mut self) {
        // This gets called from the event loop to ensure scroll values stay reasonable
        // Individual panels will still do their own clamping, but this prevents
        // scroll values from growing too large when content changes
        
        // We can't know the exact bounds without the UI context, but we can
        // at least ensure they don't grow unbounded. The UI will apply final clamping.
        
        // Summary scroll - arbitrary max to prevent extreme values
        if self.metrics_summary_scroll > 1000 {
            self.metrics_summary_scroll = 100;
        }
        
        // Policies scroll - same logic
        if self.metrics_policies_scroll > 1000 {
            self.metrics_policies_scroll = 100;
        }
        
        // Horizontal scroll - limit to reasonable policy count
        if let Some(sample) = self.training_metrics.last() {
            let num_policies = sample.policies().len();
            if num_policies > 0 {
                self.metrics_policies_horizontal_scroll = 
                    self.metrics_policies_horizontal_scroll.min(num_policies.saturating_sub(1));
            }
        }
    }

    pub fn metrics_scroll_up(&mut self, amount: usize) {
        match self.metrics_focus {
            MetricsFocus::History => {
                self.metrics_history_scroll = self.metrics_history_scroll.saturating_sub(amount);
            }
            MetricsFocus::Summary => {
                self.metrics_summary_scroll = self.metrics_summary_scroll.saturating_sub(amount);
            }
            MetricsFocus::Policies => {
                self.metrics_policies_scroll = self.metrics_policies_scroll.saturating_sub(amount);
            }
            MetricsFocus::Chart => {
                // Chart doesn't scroll, but we can use it for zoom or other controls later
            }
        }
    }

    pub fn metrics_scroll_down(&mut self, amount: usize) {
        match self.metrics_focus {
            MetricsFocus::History => {
                self.metrics_history_scroll = self.metrics_history_scroll.saturating_add(amount);
            }
            MetricsFocus::Summary => {
                self.metrics_summary_scroll = self.metrics_summary_scroll.saturating_add(amount);
            }
            MetricsFocus::Policies => {
                self.metrics_policies_scroll = self.metrics_policies_scroll.saturating_add(amount);
            }
            MetricsFocus::Chart => {
                // Chart doesn't scroll
            }
        }
    }

    pub fn metrics_policies_expanded(&self) -> bool {
        self.metrics_policies_expanded
    }

    pub fn metrics_toggle_policies_expanded(&mut self) {
        self.metrics_policies_expanded = !self.metrics_policies_expanded;
        // Reset scrolls when toggling
        if self.metrics_policies_expanded {
            self.metrics_policies_horizontal_scroll = 0;
            self.metrics_policies_scroll = 0;
        }
    }

    pub fn metrics_policies_horizontal_scroll(&self) -> usize {
        self.metrics_policies_horizontal_scroll
    }

    pub fn metrics_scroll_policies_left(&mut self) {
        self.metrics_policies_horizontal_scroll = self.metrics_policies_horizontal_scroll.saturating_sub(1);
    }

    pub fn metrics_scroll_policies_right(&mut self) {
        self.metrics_policies_horizontal_scroll = self.metrics_policies_horizontal_scroll.saturating_add(1);
    }

    pub fn export_output(&self) -> &[String] {
        &self.export_output
    }

    pub fn export_output_scroll(&self) -> usize {
        self.export_output_scroll
    }

    pub fn scroll_export_output_up(&mut self, lines: usize) {
        if self.export_output.is_empty() {
            self.export_output_scroll = 0;
            return;
        }
        let max_offset = self.export_output.len().saturating_sub(1);
        self.export_output_scroll = self
            .export_output_scroll
            .saturating_add(lines)
            .min(max_offset);
    }

    pub fn scroll_export_output_down(&mut self, lines: usize) {
        if lines >= self.export_output_scroll {
            self.export_output_scroll = 0;
        } else {
            self.export_output_scroll -= lines;
        }
    }

    pub fn reset_export_output_scroll(&mut self) {
        self.export_output_scroll = 0;
    }

    pub fn latest_training_metric(&self) -> Option<&MetricSample> {
        self.training_metrics.last()
    }

    pub fn is_training_running(&self) -> bool {
        self.training_running
    }

    pub fn is_export_running(&self) -> bool {
        self.export_running
    }

    pub fn training_config(&self) -> &TrainingConfig {
        &self.training_config
    }

    // pub fn training_config_mut(&mut self) -> &mut TrainingConfig {
    //     &mut self.training_config
    // }

    pub fn export_config(&self) -> &ExportConfig {
        &self.export_config
    }

    pub fn is_training_config_valid(&self) -> bool {
        self.training_config_valid
    }

    pub fn update_validation_status(&mut self) {
        self.training_config_valid = self.validate_training_config().is_ok();
    }

    pub fn toggle_training_mode(&mut self) {
        self.training_config.mode = match self.training_config.mode {
            TrainingMode::SingleAgent => TrainingMode::MultiAgent,
            TrainingMode::MultiAgent => TrainingMode::SingleAgent,
        };
        if let Err(error) = self.persist_training_config() {
            self.set_status(
                format!("Failed to save training config: {error}"),
                StatusKind::Error,
            );
        }
        self.rebuild_advanced_fields();
        self.update_validation_status();
    }

    pub fn show_help(&mut self) {
        self.input_mode = InputMode::Help;
    }

    pub fn hide_help(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn is_help_visible(&self) -> bool {
        self.input_mode == InputMode::Help
    }

    pub fn request_quit(&mut self) {
        if self.training_running {
            self.set_status(
                "Cannot quit while training is running. Cancel training first.",
                StatusKind::Warning,
            );
        } else {
            self.input_mode = InputMode::ConfirmQuit;
        }
    }

    pub fn cancel_quit(&mut self) {
        self.input_mode = InputMode::Normal;
    }

    pub fn confirm_quit(&mut self) {
        self.should_quit = true;
    }

    pub fn start_config_edit(&mut self, field: ConfigField) {
        let origin_mode = self.input_mode;
        self.config_return_mode = Some(origin_mode);
        self.input_mode = match origin_mode {
            InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig => {
                InputMode::EditingAdvancedConfig
            }
            _ => InputMode::EditingConfig,
        };
        self.active_config_field = Some(field);
        self.config_edit_buffer = self.config_field_value(field);
    }

    pub fn cancel_config_edit(&mut self) {
        let return_mode = self.config_return_mode.take().unwrap_or(InputMode::Normal);
        self.input_mode = return_mode;
        self.active_config_field = None;
        self.config_edit_buffer.clear();
        if matches!(
            return_mode,
            InputMode::AdvancedConfig | InputMode::EditingAdvancedConfig
        ) {
            self.rebuild_advanced_fields();
        }
    }

    pub fn push_config_char(&mut self, ch: char) {
        if self.config_edit_buffer.len() >= 256 {
            return;
        }
        self.config_edit_buffer.push(ch);
    }

    pub fn pop_config_char(&mut self) {
        self.config_edit_buffer.pop();
    }

    pub fn confirm_config_edit(&mut self) {
        let field = match self.active_config_field {
            Some(f) => f,
            None => return,
        };

        let value = self.config_edit_buffer.clone();
        match self
            .set_config_field_value(field, &value)
            .and_then(|_| self.persist_training_config())
        {
            Ok(_) => {
                self.set_status("Configuration updated", StatusKind::Success);
                self.cancel_config_edit();
                self.update_validation_status();
            }
            Err(e) => {
                self.set_status(format!("Failed to update config: {}", e), StatusKind::Error);
            }
        }
    }

    pub fn active_config_field(&self) -> Option<ConfigField> {
        self.active_config_field
    }

    pub fn config_edit_buffer(&self) -> &str {
        &self.config_edit_buffer
    }

    pub fn open_advanced_config(&mut self) {
        self.rebuild_advanced_fields();
        if self.advanced_fields.is_empty() {
            self.set_status(
                "No advanced settings available for this mode yet.",
                StatusKind::Info,
            );
            return;
        }
        if self.advanced_selection >= self.advanced_fields.len() {
            self.advanced_selection = 0;
        }
        self.input_mode = InputMode::AdvancedConfig;
    }

    pub fn close_advanced_config(&mut self) {
        self.input_mode = InputMode::Normal;
        self.config_return_mode = None;
        self.active_config_field = None;
        self.config_edit_buffer.clear();
        self.advanced_fields.clear();
        self.advanced_selection = 0;
    }

    pub fn advanced_fields(&self) -> &[ConfigField] {
        &self.advanced_fields
    }

    pub fn advanced_selection(&self) -> usize {
        self.advanced_selection
    }

    pub fn selected_advanced_field(&self) -> Option<ConfigField> {
        self.advanced_fields.get(self.advanced_selection).copied()
    }

    pub fn select_next_advanced_field(&mut self) {
        if !self.advanced_fields.is_empty() {
            self.advanced_selection = (self.advanced_selection + 1) % self.advanced_fields.len();
        }
    }

    pub fn select_previous_advanced_field(&mut self) {
        if !self.advanced_fields.is_empty() {
            if self.advanced_selection == 0 {
                self.advanced_selection = self.advanced_fields.len() - 1;
            } else {
                self.advanced_selection -= 1;
            }
        }
    }

    pub fn edit_selected_advanced_field(&mut self) {
        if let Some(field) = self.selected_advanced_field() {
            if field == ConfigField::RllibResumeFrom {
                self.start_config_file_browser(field);
            } else {
                self.start_config_edit(field);
            }
        }
    }

    pub fn clear_selected_advanced_field(&mut self) {
        if self.is_training_running() {
            self.set_status(
                "Cannot modify settings while training is running",
                StatusKind::Warning,
            );
            return;
        }

        if let Some(field) = self.selected_advanced_field() {
            match self.reset_config_field_to_default(field) {
                Ok(true) => {
                    self.set_status("Field reset to default", StatusKind::Success);
                }
                Ok(false) => {
                    self.set_status("Field already at default value", StatusKind::Info);
                }
                Err(error) => {
                    self.set_status(format!("Failed to reset field: {error}"), StatusKind::Error);
                }
            }
        }
    }

    pub fn export_mode(&self) -> ExportMode {
        self.export_mode
    }

    pub fn toggle_export_mode(&mut self) -> Result<()> {
        self.export_mode = match self.export_mode {
            ExportMode::StableBaselines3 => ExportMode::Rllib,
            ExportMode::Rllib => ExportMode::StableBaselines3,
        };
        self.rebuild_export_fields();
        self.persist_export_state()
    }

    pub fn export_focus(&self) -> ExportFocus {
        self.export_focus
    }

    pub fn toggle_export_focus(&mut self) {
        self.export_focus = match self.export_focus {
            ExportFocus::Fields => ExportFocus::Output,
            ExportFocus::Output => ExportFocus::Fields,
        };
    }

    pub fn export_fields(&self) -> &[ExportField] {
        &self.export_fields
    }

    pub fn export_selection(&self) -> usize {
        self.export_selection
    }

    pub fn selected_export_field(&self) -> Option<ExportField> {
        self.export_fields.get(self.export_selection).copied()
    }

    pub fn select_next_export_field(&mut self) {
        if !self.export_fields.is_empty() {
            self.export_selection = (self.export_selection + 1) % self.export_fields.len();
        }
    }

    pub fn select_previous_export_field(&mut self) {
        if !self.export_fields.is_empty() {
            if self.export_selection == 0 {
                self.export_selection = self.export_fields.len() - 1;
            } else {
                self.export_selection -= 1;
            }
        }
    }

    pub fn edit_selected_export_field(&mut self) {
        if let Some(field) = self.selected_export_field() {
            if field.is_toggle() {
                self.toggle_export_field(field);
            } else if field.uses_file_browser() {
                self.start_export_file_browser(field);
            } else {
                self.start_export_edit(field);
            }
        }
    }

    pub fn clear_selected_export_field(&mut self) {
        if self.export_focus != ExportFocus::Fields {
            return;
        }

        if self.is_export_running() {
            self.set_status(
                "Cannot edit export options during an active export",
                StatusKind::Warning,
            );
            return;
        }

        if let Some(field) = self.selected_export_field() {
            match self.reset_export_field_to_default(field) {
                Ok(true) => {
                    self.set_status("Export option reset to default", StatusKind::Success);
                }
                Ok(false) => {
                    self.set_status("Export option already at default value", StatusKind::Info);
                }
                Err(error) => {
                    self.set_status(
                        format!("Failed to reset export option: {error}"),
                        StatusKind::Error,
                    );
                }
            }
            self.rebuild_export_fields();
        }
    }

    pub fn active_export_field(&self) -> Option<ExportField> {
        self.active_export_field
    }

    pub fn export_edit_buffer(&self) -> &str {
        &self.export_edit_buffer
    }

    pub fn config_field_value(&self, field: ConfigField) -> String {
        match field {
            ConfigField::EnvPath => self.training_config.env_path.clone(),
            ConfigField::Timesteps => self.training_config.timesteps.to_string(),
            ConfigField::ExperimentName => self.training_config.experiment_name.clone(),
            ConfigField::Sb3Speedup => self.training_config.sb3_speedup.to_string(),
            ConfigField::Sb3NParallel => self.training_config.sb3_n_parallel.to_string(),
            ConfigField::Sb3Viz => if self.training_config.sb3_viz {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ConfigField::Sb3PolicyLayers => {
                format_usize_list(&self.training_config.sb3_policy_layers)
            }
            ConfigField::Sb3LearningRate => format_f64(self.training_config.sb3_learning_rate),
            ConfigField::Sb3BatchSize => self.training_config.sb3_batch_size.to_string(),
            ConfigField::Sb3NSteps => self.training_config.sb3_n_steps.to_string(),
            ConfigField::Sb3Gamma => format_f64(self.training_config.sb3_gamma),
            ConfigField::Sb3GaeLambda => format_f64(self.training_config.sb3_gae_lambda),
            ConfigField::Sb3EntCoef => format_f64(self.training_config.sb3_ent_coef),
            ConfigField::Sb3ClipRange => format_f64(self.training_config.sb3_clip_range),
            ConfigField::Sb3VfCoef => format_f64(self.training_config.sb3_vf_coef),
            ConfigField::Sb3MaxGradNorm => format_f64(self.training_config.sb3_max_grad_norm),
            ConfigField::RllibConfigFile => self.training_config.rllib_config_file.clone(),
            ConfigField::RllibShowWindow => if self.training_config.rllib_show_window {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ConfigField::RllibNumWorkers => self.training_config.rllib_num_workers.to_string(),
            ConfigField::RllibNumEnvWorkers => {
                self.training_config.rllib_num_envs_per_worker.to_string()
            }
            ConfigField::RllibTrainBatchSize => {
                self.training_config.rllib_train_batch_size.to_string()
            }
            ConfigField::RllibSgdMinibatchSize => {
                self.training_config.rllib_sgd_minibatch_size.to_string()
            }
            ConfigField::RllibNumSgdIter => self.training_config.rllib_num_sgd_iter.to_string(),
            ConfigField::RllibLr => format_f64(self.training_config.rllib_lr),
            ConfigField::RllibGamma => format_f64(self.training_config.rllib_gamma),
            ConfigField::RllibLambda => format_f64(self.training_config.rllib_lambda),
            ConfigField::RllibClipParam => format_f64(self.training_config.rllib_clip_param),
            ConfigField::RllibEntropyCoeff => format_f64(self.training_config.rllib_entropy_coeff),
            ConfigField::RllibVfLossCoeff => format_f64(self.training_config.rllib_vf_loss_coeff),
            ConfigField::RllibGradClip => format_f64(self.training_config.rllib_grad_clip),
            ConfigField::RllibFramework => self.training_config.rllib_framework.clone(),
            ConfigField::RllibActivation => self.training_config.rllib_activation.clone(),
            ConfigField::RllibBatchMode => self.training_config.rllib_batch_mode.clone(),
            ConfigField::RllibRolloutFragmentLength => self
                .training_config
                .rllib_rollout_fragment_length
                .to_string(),
            ConfigField::RllibFcnetHiddens => {
                format_usize_list(&self.training_config.rllib_fcnet_hiddens)
            }
            ConfigField::RllibCheckpointFrequency => {
                self.training_config.rllib_checkpoint_frequency.to_string()
            }
            ConfigField::RllibResumeFrom => self.training_config.rllib_resume_from.clone(),
            ConfigField::RllibStopMode => match self.training_config.rllib_stop_mode {
                RllibStopMode::TimeSeconds => "time_seconds".to_string(),
                RllibStopMode::Timesteps => "timesteps".to_string(),
            },
            ConfigField::RllibStopTimeSeconds => {
                self.training_config.rllib_stop_time_seconds.to_string()
            }
        }
    }

    pub fn export_field_value(&self, field: ExportField) -> String {
        match field {
            ExportField::Sb3ModelPath => self.export_config.sb3_model_path.clone(),
            ExportField::Sb3OutputPath => self.export_config.sb3_output_path.clone(),
            ExportField::Sb3Algo => self.export_config.sb3_algo.clone(),
            ExportField::Sb3Opset => self.export_config.sb3_opset.to_string(),
            ExportField::Sb3IrVersion => self.export_config.sb3_ir_version.to_string(),
            ExportField::Sb3UseObsArray => if self.export_config.sb3_use_obs_array {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ExportField::Sb3SkipVerify => if self.export_config.sb3_skip_verify {
                "true"
            } else {
                "false"
            }
            .to_string(),
            ExportField::RllibCheckpointPath => self.export_config.rllib_checkpoint_path.clone(),
            ExportField::RllibCheckpointNumber => self
                .export_config
                .rllib_checkpoint_number
                .map(|value| value.to_string())
                .unwrap_or_default(),
            ExportField::RllibOutputDir => self.export_config.rllib_output_dir.clone(),
            ExportField::RllibPolicyId => self.export_config.rllib_policy_id.clone(),
            ExportField::RllibOpset => self.export_config.rllib_opset.to_string(),
            ExportField::RllibIrVersion => self.export_config.rllib_ir_version.to_string(),
            ExportField::RllibMultiagent => if self.export_config.rllib_multiagent {
                "true"
            } else {
                "false"
            }
            .to_string(),
        }
    }

    pub fn toggle_export_field(&mut self, field: ExportField) {
        match field {
            ExportField::Sb3UseObsArray => {
                self.export_config.sb3_use_obs_array = !self.export_config.sb3_use_obs_array;
                let state = if self.export_config.sb3_use_obs_array {
                    "enabled"
                } else {
                    "disabled"
                };
                self.set_status(format!("SB3 obs-array export {state}"), StatusKind::Info);
            }
            ExportField::Sb3SkipVerify => {
                self.export_config.sb3_skip_verify = !self.export_config.sb3_skip_verify;
                let state = if self.export_config.sb3_skip_verify {
                    "Skipping verification"
                } else {
                    "Running verification"
                };
                self.set_status(state, StatusKind::Info);
            }
            ExportField::RllibMultiagent => {
                self.export_config.rllib_multiagent = !self.export_config.rllib_multiagent;
                let state = if self.export_config.rllib_multiagent {
                    "Multi-agent export"
                } else {
                    "Single-agent export"
                };
                self.set_status(state, StatusKind::Info);
            }
            _ => {}
        }
        if let Err(error) = self.persist_export_state() {
            self.set_status(
                format!("Failed to save export settings: {error}"),
                StatusKind::Error,
            );
        }
        self.rebuild_export_fields();
    }

    pub fn start_export_edit(&mut self, field: ExportField) {
        let origin_mode = self.input_mode;
        self.export_return_mode = Some(origin_mode);
        self.input_mode = InputMode::EditingExport;
        self.active_export_field = Some(field);
        self.export_edit_buffer = self.export_field_value(field);
    }

    pub fn cancel_export_edit(&mut self) {
        let return_mode = self.export_return_mode.take().unwrap_or(InputMode::Normal);
        self.input_mode = return_mode;
        self.active_export_field = None;
        self.export_edit_buffer.clear();
    }

    pub fn push_export_char(&mut self, ch: char) {
        if self.export_edit_buffer.len() >= 256 {
            return;
        }
        self.export_edit_buffer.push(ch);
    }

    pub fn pop_export_char(&mut self) {
        self.export_edit_buffer.pop();
    }

    pub fn confirm_export_edit(&mut self) {
        let field = match self.active_export_field {
            Some(field) => field,
            None => return,
        };
        let value = self.export_edit_buffer.clone();
        match self.set_export_field_value(field, &value) {
            Ok(()) => {
                self.set_status("Export option updated", StatusKind::Success);
                self.cancel_export_edit();
                self.rebuild_export_fields();
            }
            Err(error) => {
                self.set_status(
                    format!("Failed to update export option: {error}"),
                    StatusKind::Error,
                );
            }
        }
    }

    fn set_export_field_value(&mut self, field: ExportField, value: &str) -> Result<()> {
        let trimmed = value.trim();
        match field {
            ExportField::Sb3ModelPath => {
                self.export_config.sb3_model_path = trimmed.to_string();
            }
            ExportField::Sb3OutputPath => {
                self.export_config.sb3_output_path = trimmed.to_string();
            }
            ExportField::Sb3Algo => {
                self.export_config.sb3_algo = trimmed.to_string();
            }
            ExportField::Sb3Opset => {
                let opset: u32 = trimmed
                    .parse()
                    .wrap_err("Opset must be a positive integer")?;
                if opset == 0 {
                    bail!("Opset must be greater than 0");
                }
                self.export_config.sb3_opset = opset;
            }
            ExportField::Sb3IrVersion => {
                let ir: u32 = trimmed
                    .parse()
                    .wrap_err("IR version must be a positive integer")?;
                if ir == 0 {
                    bail!("IR version must be greater than 0");
                }
                self.export_config.sb3_ir_version = ir;
            }
            ExportField::Sb3UseObsArray | ExportField::Sb3SkipVerify => {
                bail!("Toggle values cannot be edited as text");
            }
            ExportField::RllibCheckpointPath => {
                if trimmed.is_empty() {
                    self.export_config.rllib_checkpoint_path.clear();
                    self.export_config.rllib_checkpoint_number = None;
                } else {
                    self.export_config.rllib_checkpoint_path = trimmed.to_string();
                    self.sync_checkpoint_number_from_path(trimmed);
                }
            }
            ExportField::RllibCheckpointNumber => {
                if trimmed.is_empty() {
                    self.export_config.rllib_checkpoint_number = None;
                } else {
                    let number: u32 = trimmed
                        .parse()
                        .wrap_err("Checkpoint number must be a non-negative integer")?;
                    self.export_config.rllib_checkpoint_number = Some(number);
                }
            }
            ExportField::RllibOutputDir => {
                self.export_config.rllib_output_dir = trimmed.to_string();
            }
            ExportField::RllibPolicyId => {
                self.export_config.rllib_policy_id = trimmed.to_string();
            }
            ExportField::RllibOpset => {
                let opset: u32 = trimmed
                    .parse()
                    .wrap_err("Opset must be a positive integer")?;
                if opset == 0 {
                    bail!("Opset must be greater than 0");
                }
                self.export_config.rllib_opset = opset;
            }
            ExportField::RllibIrVersion => {
                let ir: u32 = trimmed
                    .parse()
                    .wrap_err("IR version must be a positive integer")?;
                if ir == 0 {
                    bail!("IR version must be greater than 0");
                }
                self.export_config.rllib_ir_version = ir;
            }
            ExportField::RllibMultiagent => {
                bail!("Toggle values cannot be edited as text");
            }
        }
        self.persist_export_state()?;
        Ok(())
    }

    fn reset_export_field_to_default(&mut self, field: ExportField) -> Result<bool> {
        let defaults = ExportConfig::default();
        let changed = match field {
            ExportField::Sb3ModelPath => {
                if self.export_config.sb3_model_path == defaults.sb3_model_path {
                    false
                } else {
                    self.export_config.sb3_model_path = defaults.sb3_model_path.clone();
                    true
                }
            }
            ExportField::Sb3OutputPath => {
                if self.export_config.sb3_output_path == defaults.sb3_output_path {
                    false
                } else {
                    self.export_config.sb3_output_path = defaults.sb3_output_path.clone();
                    true
                }
            }
            ExportField::Sb3Algo => {
                if self.export_config.sb3_algo == defaults.sb3_algo {
                    false
                } else {
                    self.export_config.sb3_algo = defaults.sb3_algo.clone();
                    true
                }
            }
            ExportField::Sb3Opset => {
                if self.export_config.sb3_opset == defaults.sb3_opset {
                    false
                } else {
                    self.export_config.sb3_opset = defaults.sb3_opset;
                    true
                }
            }
            ExportField::Sb3IrVersion => {
                if self.export_config.sb3_ir_version == defaults.sb3_ir_version {
                    false
                } else {
                    self.export_config.sb3_ir_version = defaults.sb3_ir_version;
                    true
                }
            }
            ExportField::Sb3UseObsArray => {
                if self.export_config.sb3_use_obs_array == defaults.sb3_use_obs_array {
                    false
                } else {
                    self.export_config.sb3_use_obs_array = defaults.sb3_use_obs_array;
                    true
                }
            }
            ExportField::Sb3SkipVerify => {
                if self.export_config.sb3_skip_verify == defaults.sb3_skip_verify {
                    false
                } else {
                    self.export_config.sb3_skip_verify = defaults.sb3_skip_verify;
                    true
                }
            }
            ExportField::RllibCheckpointPath => {
                if self.export_config.rllib_checkpoint_path == defaults.rllib_checkpoint_path {
                    false
                } else {
                    self.export_config.rllib_checkpoint_path =
                        defaults.rllib_checkpoint_path.clone();
                    self.export_config.rllib_checkpoint_number = defaults.rllib_checkpoint_number;
                    true
                }
            }
            ExportField::RllibCheckpointNumber => {
                if self.export_config.rllib_checkpoint_number == defaults.rllib_checkpoint_number {
                    false
                } else {
                    self.export_config.rllib_checkpoint_number = defaults.rllib_checkpoint_number;
                    true
                }
            }
            ExportField::RllibOutputDir => {
                if self.export_config.rllib_output_dir == defaults.rllib_output_dir {
                    false
                } else {
                    self.export_config.rllib_output_dir = defaults.rllib_output_dir.clone();
                    true
                }
            }
            ExportField::RllibPolicyId => {
                if self.export_config.rllib_policy_id == defaults.rllib_policy_id {
                    false
                } else {
                    self.export_config.rllib_policy_id = defaults.rllib_policy_id.clone();
                    true
                }
            }
            ExportField::RllibOpset => {
                if self.export_config.rllib_opset == defaults.rllib_opset {
                    false
                } else {
                    self.export_config.rllib_opset = defaults.rllib_opset;
                    true
                }
            }
            ExportField::RllibIrVersion => {
                if self.export_config.rllib_ir_version == defaults.rllib_ir_version {
                    false
                } else {
                    self.export_config.rllib_ir_version = defaults.rllib_ir_version;
                    true
                }
            }
            ExportField::RllibMultiagent => {
                if self.export_config.rllib_multiagent == defaults.rllib_multiagent {
                    false
                } else {
                    self.export_config.rllib_multiagent = defaults.rllib_multiagent;
                    true
                }
            }
        };

        if changed {
            self.persist_export_state()?;
        }
        Ok(changed)
    }

    fn rebuild_export_fields(&mut self) {
        self.export_fields = self.build_export_fields();
        if self.export_fields.is_empty() {
            self.export_selection = 0;
        } else if self.export_selection >= self.export_fields.len() {
            self.export_selection = self.export_fields.len() - 1;
        }
    }

    fn build_export_fields(&self) -> Vec<ExportField> {
        match self.export_mode {
            ExportMode::StableBaselines3 => vec![
                ExportField::Sb3ModelPath,
                ExportField::Sb3OutputPath,
                ExportField::Sb3Algo,
                ExportField::Sb3Opset,
                ExportField::Sb3IrVersion,
                ExportField::Sb3UseObsArray,
                ExportField::Sb3SkipVerify,
            ],
            ExportMode::Rllib => vec![
                ExportField::RllibCheckpointPath,
                ExportField::RllibCheckpointNumber,
                ExportField::RllibOutputDir,
                ExportField::RllibPolicyId,
                ExportField::RllibOpset,
                ExportField::RllibIrVersion,
                ExportField::RllibMultiagent,
            ],
        }
    }

    fn reset_config_field_to_default(&mut self, field: ConfigField) -> Result<bool> {
        let defaults = TrainingConfig::default();
        let changed = match field {
            ConfigField::EnvPath => {
                if self.training_config.env_path == defaults.env_path {
                    false
                } else {
                    self.training_config.env_path = defaults.env_path.clone();
                    true
                }
            }
            ConfigField::Timesteps => {
                if self.training_config.timesteps == defaults.timesteps {
                    false
                } else {
                    self.training_config.timesteps = defaults.timesteps;
                    true
                }
            }
            ConfigField::ExperimentName => {
                if self.training_config.experiment_name == defaults.experiment_name {
                    false
                } else {
                    self.training_config.experiment_name = defaults.experiment_name.clone();
                    true
                }
            }
            ConfigField::Sb3Speedup => {
                if self.training_config.sb3_speedup == defaults.sb3_speedup {
                    false
                } else {
                    self.training_config.sb3_speedup = defaults.sb3_speedup;
                    true
                }
            }
            ConfigField::Sb3NParallel => {
                if self.training_config.sb3_n_parallel == defaults.sb3_n_parallel {
                    false
                } else {
                    self.training_config.sb3_n_parallel = defaults.sb3_n_parallel;
                    true
                }
            }
            ConfigField::Sb3Viz => {
                if self.training_config.sb3_viz == defaults.sb3_viz {
                    false
                } else {
                    self.training_config.sb3_viz = defaults.sb3_viz;
                    true
                }
            }
            ConfigField::Sb3PolicyLayers => {
                if self.training_config.sb3_policy_layers == defaults.sb3_policy_layers {
                    false
                } else {
                    self.training_config.sb3_policy_layers = defaults.sb3_policy_layers.clone();
                    true
                }
            }
            ConfigField::Sb3LearningRate => {
                if (self.training_config.sb3_learning_rate - defaults.sb3_learning_rate).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_learning_rate = defaults.sb3_learning_rate;
                    true
                }
            }
            ConfigField::Sb3BatchSize => {
                if self.training_config.sb3_batch_size == defaults.sb3_batch_size {
                    false
                } else {
                    self.training_config.sb3_batch_size = defaults.sb3_batch_size;
                    true
                }
            }
            ConfigField::Sb3NSteps => {
                if self.training_config.sb3_n_steps == defaults.sb3_n_steps {
                    false
                } else {
                    self.training_config.sb3_n_steps = defaults.sb3_n_steps;
                    true
                }
            }
            ConfigField::Sb3Gamma => {
                if (self.training_config.sb3_gamma - defaults.sb3_gamma).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.sb3_gamma = defaults.sb3_gamma;
                    true
                }
            }
            ConfigField::Sb3GaeLambda => {
                if (self.training_config.sb3_gae_lambda - defaults.sb3_gae_lambda).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_gae_lambda = defaults.sb3_gae_lambda;
                    true
                }
            }
            ConfigField::Sb3EntCoef => {
                if (self.training_config.sb3_ent_coef - defaults.sb3_ent_coef).abs() < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_ent_coef = defaults.sb3_ent_coef;
                    true
                }
            }
            ConfigField::Sb3ClipRange => {
                if (self.training_config.sb3_clip_range - defaults.sb3_clip_range).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_clip_range = defaults.sb3_clip_range;
                    true
                }
            }
            ConfigField::Sb3VfCoef => {
                if (self.training_config.sb3_vf_coef - defaults.sb3_vf_coef).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.sb3_vf_coef = defaults.sb3_vf_coef;
                    true
                }
            }
            ConfigField::Sb3MaxGradNorm => {
                if (self.training_config.sb3_max_grad_norm - defaults.sb3_max_grad_norm).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.sb3_max_grad_norm = defaults.sb3_max_grad_norm;
                    true
                }
            }
            ConfigField::RllibConfigFile => {
                if self.training_config.rllib_config_file == defaults.rllib_config_file {
                    false
                } else {
                    self.training_config.rllib_config_file = defaults.rllib_config_file.clone();
                    true
                }
            }
            ConfigField::RllibShowWindow => {
                if self.training_config.rllib_show_window == defaults.rllib_show_window {
                    false
                } else {
                    self.training_config.rllib_show_window = defaults.rllib_show_window;
                    true
                }
            }
            ConfigField::RllibNumWorkers => {
                if self.training_config.rllib_num_workers == defaults.rllib_num_workers {
                    false
                } else {
                    self.training_config.rllib_num_workers = defaults.rllib_num_workers;
                    true
                }
            }
            ConfigField::RllibNumEnvWorkers => {
                if self.training_config.rllib_num_envs_per_worker
                    == defaults.rllib_num_envs_per_worker
                {
                    false
                } else {
                    self.training_config.rllib_num_envs_per_worker =
                        defaults.rllib_num_envs_per_worker;
                    true
                }
            }
            ConfigField::RllibTrainBatchSize => {
                if self.training_config.rllib_train_batch_size == defaults.rllib_train_batch_size {
                    false
                } else {
                    self.training_config.rllib_train_batch_size = defaults.rllib_train_batch_size;
                    true
                }
            }
            ConfigField::RllibSgdMinibatchSize => {
                if self.training_config.rllib_sgd_minibatch_size
                    == defaults.rllib_sgd_minibatch_size
                {
                    false
                } else {
                    self.training_config.rllib_sgd_minibatch_size =
                        defaults.rllib_sgd_minibatch_size;
                    true
                }
            }
            ConfigField::RllibNumSgdIter => {
                if self.training_config.rllib_num_sgd_iter == defaults.rllib_num_sgd_iter {
                    false
                } else {
                    self.training_config.rllib_num_sgd_iter = defaults.rllib_num_sgd_iter;
                    true
                }
            }
            ConfigField::RllibLr => {
                if (self.training_config.rllib_lr - defaults.rllib_lr).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.rllib_lr = defaults.rllib_lr;
                    true
                }
            }
            ConfigField::RllibGamma => {
                if (self.training_config.rllib_gamma - defaults.rllib_gamma).abs() < f64::EPSILON {
                    false
                } else {
                    self.training_config.rllib_gamma = defaults.rllib_gamma;
                    true
                }
            }
            ConfigField::RllibLambda => {
                if (self.training_config.rllib_lambda - defaults.rllib_lambda).abs() < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_lambda = defaults.rllib_lambda;
                    true
                }
            }
            ConfigField::RllibClipParam => {
                if (self.training_config.rllib_clip_param - defaults.rllib_clip_param).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_clip_param = defaults.rllib_clip_param;
                    true
                }
            }
            ConfigField::RllibEntropyCoeff => {
                if (self.training_config.rllib_entropy_coeff - defaults.rllib_entropy_coeff).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_entropy_coeff = defaults.rllib_entropy_coeff;
                    true
                }
            }
            ConfigField::RllibVfLossCoeff => {
                if (self.training_config.rllib_vf_loss_coeff - defaults.rllib_vf_loss_coeff).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_vf_loss_coeff = defaults.rllib_vf_loss_coeff;
                    true
                }
            }
            ConfigField::RllibGradClip => {
                if (self.training_config.rllib_grad_clip - defaults.rllib_grad_clip).abs()
                    < f64::EPSILON
                {
                    false
                } else {
                    self.training_config.rllib_grad_clip = defaults.rllib_grad_clip;
                    true
                }
            }
            ConfigField::RllibFramework => {
                if self.training_config.rllib_framework == defaults.rllib_framework {
                    false
                } else {
                    self.training_config.rllib_framework = defaults.rllib_framework.clone();
                    true
                }
            }
            ConfigField::RllibActivation => {
                if self.training_config.rllib_activation == defaults.rllib_activation {
                    false
                } else {
                    self.training_config.rllib_activation = defaults.rllib_activation.clone();
                    true
                }
            }
            ConfigField::RllibBatchMode => {
                if self.training_config.rllib_batch_mode == defaults.rllib_batch_mode {
                    false
                } else {
                    self.training_config.rllib_batch_mode = defaults.rllib_batch_mode.clone();
                    true
                }
            }
            ConfigField::RllibRolloutFragmentLength => {
                if self.training_config.rllib_rollout_fragment_length
                    == defaults.rllib_rollout_fragment_length
                {
                    false
                } else {
                    self.training_config.rllib_rollout_fragment_length =
                        defaults.rllib_rollout_fragment_length;
                    true
                }
            }
            ConfigField::RllibFcnetHiddens => {
                if self.training_config.rllib_fcnet_hiddens == defaults.rllib_fcnet_hiddens {
                    false
                } else {
                    self.training_config.rllib_fcnet_hiddens = defaults.rllib_fcnet_hiddens.clone();
                    true
                }
            }
            ConfigField::RllibCheckpointFrequency => {
                if self.training_config.rllib_checkpoint_frequency
                    == defaults.rllib_checkpoint_frequency
                {
                    false
                } else {
                    self.training_config.rllib_checkpoint_frequency =
                        defaults.rllib_checkpoint_frequency;
                    true
                }
            }
            ConfigField::RllibResumeFrom => {
                if self.training_config.rllib_resume_from == defaults.rllib_resume_from {
                    false
                } else {
                    self.training_config.rllib_resume_from = defaults.rllib_resume_from.clone();
                    true
                }
            }
            ConfigField::RllibStopMode => {
                if self.training_config.rllib_stop_mode == defaults.rllib_stop_mode {
                    false
                } else {
                    self.training_config.rllib_stop_mode = defaults.rllib_stop_mode;
                    true
                }
            }
            ConfigField::RllibStopTimeSeconds => {
                if self.training_config.rllib_stop_time_seconds == defaults.rllib_stop_time_seconds
                {
                    false
                } else {
                    self.training_config.rllib_stop_time_seconds = defaults.rllib_stop_time_seconds;
                    true
                }
            }
        };

        if changed {
            self.persist_training_config()?;
            self.update_validation_status();
        }
        Ok(changed)
    }

    fn set_config_field_value(&mut self, field: ConfigField, value: &str) -> Result<()> {
        let trimmed = value.trim();
        match field {
            ConfigField::EnvPath => {
                self.training_config.env_path = trimmed.to_string();
            }
            ConfigField::Timesteps => {
                self.training_config.timesteps = trimmed
                    .parse()
                    .wrap_err("Timesteps must be a positive number")?;
            }
            ConfigField::ExperimentName => {
                if trimmed.is_empty() {
                    bail!("Experiment name cannot be empty");
                }
                self.training_config.experiment_name = trimmed.to_string();
            }
            ConfigField::Sb3Speedup => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Speedup must be a positive integer")?;
                if val == 0 {
                    bail!("Speedup must be at least 1");
                }
                self.training_config.sb3_speedup = val;
            }
            ConfigField::Sb3NParallel => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("N parallel must be a positive integer")?;
                if val == 0 {
                    bail!("N parallel must be at least 1");
                }
                self.training_config.sb3_n_parallel = val;
            }
            ConfigField::Sb3Viz => {
                self.training_config.sb3_viz =
                    matches!(trimmed.to_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            ConfigField::Sb3PolicyLayers => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.sb3_policy_layers = layers;
            }
            ConfigField::Sb3LearningRate => {
                let val: f64 = trimmed.parse().wrap_err("Learning rate must be a number")?;
                if val <= 0.0 {
                    bail!("Learning rate must be greater than 0");
                }
                self.training_config.sb3_learning_rate = val;
            }
            ConfigField::Sb3BatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Batch size must be a positive integer")?;
                if val == 0 {
                    bail!("Batch size must be at least 1");
                }
                self.training_config.sb3_batch_size = val;
            }
            ConfigField::Sb3NSteps => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("n_steps must be a positive integer")?;
                if val == 0 {
                    bail!("n_steps must be at least 1");
                }
                self.training_config.sb3_n_steps = val;
            }
            ConfigField::Sb3Gamma => {
                let val: f64 = trimmed.parse().wrap_err("Gamma must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Gamma must be between 0 and 1");
                }
                self.training_config.sb3_gamma = val;
            }
            ConfigField::Sb3GaeLambda => {
                let val: f64 = trimmed.parse().wrap_err("GAE Lambda must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("GAE Lambda must be between 0 and 1");
                }
                self.training_config.sb3_gae_lambda = val;
            }
            ConfigField::Sb3EntCoef => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Entropy coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Entropy coefficient cannot be negative");
                }
                self.training_config.sb3_ent_coef = val;
            }
            ConfigField::Sb3ClipRange => {
                let val: f64 = trimmed.parse().wrap_err("Clip range must be a number")?;
                if val <= 0.0 {
                    bail!("Clip range must be greater than 0");
                }
                self.training_config.sb3_clip_range = val;
            }
            ConfigField::Sb3VfCoef => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Value function coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Value function coefficient cannot be negative");
                }
                self.training_config.sb3_vf_coef = val;
            }
            ConfigField::Sb3MaxGradNorm => {
                let val: f64 = trimmed.parse().wrap_err("Max grad norm must be a number")?;
                if val <= 0.0 {
                    bail!("Max grad norm must be greater than 0");
                }
                self.training_config.sb3_max_grad_norm = val;
            }
            ConfigField::RllibConfigFile => {
                if trimmed.is_empty() {
                    bail!("Config file path cannot be empty");
                }
                self.training_config.rllib_config_file = trimmed.to_string();
            }
            ConfigField::RllibShowWindow => {
                self.training_config.rllib_show_window =
                    matches!(trimmed.to_lowercase().as_str(), "true" | "yes" | "1" | "on");
            }
            ConfigField::RllibNumWorkers => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Number of workers must be a positive integer")?;
                if val == 0 {
                    bail!("Number of workers must be at least 1");
                }
                self.training_config.rllib_num_workers = val;
            }
            ConfigField::RllibNumEnvWorkers => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Environments per worker must be a positive integer")?;
                if val == 0 {
                    bail!("Environments per worker must be at least 1");
                }
                self.training_config.rllib_num_envs_per_worker = val;
            }
            ConfigField::RllibTrainBatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Train batch size must be a positive integer")?;
                if val == 0 {
                    bail!("Train batch size must be at least 1");
                }
                self.training_config.rllib_train_batch_size = val;
            }
            ConfigField::RllibSgdMinibatchSize => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("SGD minibatch size must be a positive integer")?;
                if val == 0 {
                    bail!("SGD minibatch size must be at least 1");
                }
                self.training_config.rllib_sgd_minibatch_size = val;
            }
            ConfigField::RllibNumSgdIter => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Number of SGD iterations must be a positive integer")?;
                if val == 0 {
                    bail!("Number of SGD iterations must be at least 1");
                }
                self.training_config.rllib_num_sgd_iter = val;
            }
            ConfigField::RllibLr => {
                let val: f64 = trimmed.parse().wrap_err("Learning rate must be a number")?;
                if val <= 0.0 {
                    bail!("Learning rate must be greater than 0");
                }
                self.training_config.rllib_lr = val;
            }
            ConfigField::RllibGamma => {
                let val: f64 = trimmed.parse().wrap_err("Gamma must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Gamma must be between 0 and 1");
                }
                self.training_config.rllib_gamma = val;
            }
            ConfigField::RllibLambda => {
                let val: f64 = trimmed.parse().wrap_err("Lambda must be a number")?;
                if !(0.0..=1.0).contains(&val) {
                    bail!("Lambda must be between 0 and 1");
                }
                self.training_config.rllib_lambda = val;
            }
            ConfigField::RllibClipParam => {
                let val: f64 = trimmed.parse().wrap_err("Clip param must be a number")?;
                if val <= 0.0 {
                    bail!("Clip param must be greater than 0");
                }
                self.training_config.rllib_clip_param = val;
            }
            ConfigField::RllibEntropyCoeff => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("Entropy coefficient must be a number")?;
                if val < 0.0 {
                    bail!("Entropy coefficient cannot be negative");
                }
                self.training_config.rllib_entropy_coeff = val;
            }
            ConfigField::RllibVfLossCoeff => {
                let val: f64 = trimmed
                    .parse()
                    .wrap_err("VF loss coefficient must be a number")?;
                if val < 0.0 {
                    bail!("VF loss coefficient cannot be negative");
                }
                self.training_config.rllib_vf_loss_coeff = val;
            }
            ConfigField::RllibGradClip => {
                let val: f64 = trimmed.parse().wrap_err("Grad clip must be a number")?;
                if val <= 0.0 {
                    bail!("Grad clip must be greater than 0");
                }
                self.training_config.rllib_grad_clip = val;
            }
            ConfigField::RllibFramework => {
                if trimmed.is_empty() {
                    bail!("Framework cannot be empty");
                }
                self.training_config.rllib_framework = trimmed.to_string();
            }
            ConfigField::RllibActivation => {
                if trimmed.is_empty() {
                    bail!("Activation cannot be empty");
                }
                self.training_config.rllib_activation = trimmed.to_string();
            }
            ConfigField::RllibBatchMode => {
                if trimmed.is_empty() {
                    bail!("Batch mode cannot be empty");
                }
                let lower = trimmed.to_lowercase();
                if lower != "truncate_episodes" && lower != "complete_episodes" {
                    bail!("Batch mode must be 'truncate_episodes' or 'complete_episodes'");
                }
                self.training_config.rllib_batch_mode = lower;
            }
            ConfigField::RllibRolloutFragmentLength => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Rollout fragment length must be a positive integer")?;
                if val == 0 {
                    bail!("Rollout fragment length must be at least 1");
                }
                self.training_config.rllib_rollout_fragment_length = val;
            }
            ConfigField::RllibFcnetHiddens => {
                let layers = parse_usize_list(trimmed)?;
                self.training_config.rllib_fcnet_hiddens = layers;
            }
            ConfigField::RllibCheckpointFrequency => {
                let val: u32 = trimmed
                    .parse()
                    .wrap_err("Checkpoint frequency must be a positive integer")?;
                if val == 0 {
                    bail!("Checkpoint frequency must be at least 1");
                }
                self.training_config.rllib_checkpoint_frequency = val;
            }
            ConfigField::RllibResumeFrom => {
                self.training_config.rllib_resume_from = trimmed.to_string();
            }
            ConfigField::RllibStopMode => {
                let mode = trimmed.to_lowercase();
                let stop_mode = match mode.as_str() {
                    "time" | "time_seconds" | "seconds" | "s" => RllibStopMode::TimeSeconds,
                    "timesteps" | "steps" | "timesteps_total" | "t" => RllibStopMode::Timesteps,
                    _ => bail!("Stop mode must be 'time' or 'timesteps'"),
                };
                self.training_config.rllib_stop_mode = stop_mode;
            }
            ConfigField::RllibStopTimeSeconds => {
                let val: u64 = trimmed
                    .parse()
                    .wrap_err("Time limit must be a positive integer of seconds")?;
                if val == 0 {
                    bail!("Time limit must be at least 1 second");
                }
                self.training_config.rllib_stop_time_seconds = val;
            }
        }
        Ok(())
    }

    fn persist_training_config(&mut self) -> Result<()> {
        if let Some(path) = self.training_config_path() {
            let json = serde_json::to_string_pretty(&self.training_config).wrap_err_with(|| {
                format!("failed to serialize training config for {}", path.display())
            })?;
            fs::write(&path, json).wrap_err_with(|| {
                format!("failed to write training config to {}", path.display())
            })?;
        }
        Ok(())
    }

    fn training_config_path(&self) -> Option<PathBuf> {
        self.active_project
            .as_ref()
            .map(|project| project.path.join(TRAINING_CONFIG_FILENAME))
    }

    fn load_training_config_for_active_project(&mut self) -> bool {
        let mut had_error = false;
        if let Some(path) = self.training_config_path() {
            if path.exists() {
                match fs::read_to_string(&path) {
                    Ok(contents) => match serde_json::from_str::<TrainingConfig>(&contents) {
                        Ok(config) => {
                            self.training_config = config;
                        }
                        Err(error) => {
                            self.training_config = TrainingConfig::default();
                            self.set_status(
                                format!("Invalid training config, using defaults: {}", error),
                                StatusKind::Warning,
                            );
                            had_error = true;
                        }
                    },
                    Err(error) => {
                        self.training_config = TrainingConfig::default();
                        self.set_status(
                            format!("Failed to read training config, using defaults: {}", error),
                            StatusKind::Warning,
                        );
                        had_error = true;
                    }
                }
            } else {
                self.training_config = TrainingConfig::default();
                if let Err(error) = self.persist_training_config() {
                    self.set_status(
                        format!("Failed to create training config: {error}"),
                        StatusKind::Error,
                    );
                    had_error = true;
                }
            }
        } else {
            self.training_config = TrainingConfig::default();
        }
        self.rebuild_advanced_fields();
        self.update_validation_status();
        had_error
    }

    fn persist_export_state(&mut self) -> Result<()> {
        if let Some(path) = self.export_config_path() {
            let state = ExportState {
                mode: self.export_mode,
                config: self.export_config.clone(),
            };
            let json = serde_json::to_string_pretty(&state).wrap_err_with(|| {
                format!("failed to serialize export config for {}", path.display())
            })?;
            fs::write(&path, json)
                .wrap_err_with(|| format!("failed to write export config to {}", path.display()))?;
        }
        Ok(())
    }

    fn export_config_path(&self) -> Option<PathBuf> {
        self.active_project
            .as_ref()
            .map(|project| project.path.join(EXPORT_CONFIG_FILENAME))
    }

    fn load_export_state_for_active_project(&mut self) -> bool {
        let mut had_error = false;
        if let Some(path) = self.export_config_path() {
            if path.exists() {
                match fs::read_to_string(&path) {
                    Ok(contents) => {
                        let parsed_state =
                            serde_json::from_str::<ExportState>(&contents).or_else(|_| {
                                serde_json::from_str::<ExportConfig>(&contents).map(|config| {
                                    ExportState {
                                        mode: ExportMode::default(),
                                        config,
                                    }
                                })
                            });
                        match parsed_state {
                            Ok(state) => {
                                self.export_mode = state.mode;
                                self.export_config = state.config;
                                let checkpoint_path =
                                    self.export_config.rllib_checkpoint_path.clone();
                                self.sync_checkpoint_number_from_path(&checkpoint_path);
                            }
                            Err(error) => {
                                self.export_mode = ExportMode::default();
                                self.export_config = ExportConfig::default();
                                self.set_status(
                                    format!("Invalid export config, using defaults: {}", error),
                                    StatusKind::Warning,
                                );
                                had_error = true;
                            }
                        }
                    }
                    Err(error) => {
                        self.export_mode = ExportMode::default();
                        self.export_config = ExportConfig::default();
                        self.set_status(
                            format!("Failed to read export config, using defaults: {error}"),
                            StatusKind::Warning,
                        );
                        had_error = true;
                    }
                }
            } else {
                self.export_mode = ExportMode::default();
                self.export_config = ExportConfig::default();
                if let Err(error) = self.persist_export_state() {
                    self.set_status(
                        format!("Failed to create export config: {error}"),
                        StatusKind::Error,
                    );
                    had_error = true;
                }
            }
        } else {
            self.export_mode = ExportMode::default();
            self.export_config = ExportConfig::default();
        }
        self.rebuild_export_fields();
        had_error
    }

    fn rebuild_advanced_fields(&mut self) {
        self.advanced_fields = self.build_advanced_fields();
        if self.advanced_fields.is_empty() {
            self.advanced_selection = 0;
        } else if self.advanced_selection >= self.advanced_fields.len() {
            self.advanced_selection = self.advanced_fields.len() - 1;
        }
    }

    fn build_advanced_fields(&self) -> Vec<ConfigField> {
        match self.training_config.mode {
            TrainingMode::SingleAgent => vec![
                ConfigField::Sb3Speedup,
                ConfigField::Sb3NParallel,
                ConfigField::Sb3Viz,
                ConfigField::Sb3PolicyLayers,
                ConfigField::Sb3LearningRate,
                ConfigField::Sb3BatchSize,
                ConfigField::Sb3NSteps,
                ConfigField::Sb3Gamma,
                ConfigField::Sb3GaeLambda,
                ConfigField::Sb3EntCoef,
                ConfigField::Sb3ClipRange,
                ConfigField::Sb3VfCoef,
                ConfigField::Sb3MaxGradNorm,
            ],
            TrainingMode::MultiAgent => vec![
                ConfigField::RllibConfigFile,
                ConfigField::RllibStopMode,
                ConfigField::RllibStopTimeSeconds,
                ConfigField::RllibResumeFrom,
                ConfigField::RllibShowWindow,
                ConfigField::RllibNumWorkers,
                ConfigField::RllibNumEnvWorkers,
                ConfigField::RllibTrainBatchSize,
                ConfigField::RllibSgdMinibatchSize,
                ConfigField::RllibNumSgdIter,
                ConfigField::RllibLr,
                ConfigField::RllibGamma,
                ConfigField::RllibLambda,
                ConfigField::RllibClipParam,
                ConfigField::RllibEntropyCoeff,
                ConfigField::RllibVfLossCoeff,
                ConfigField::RllibGradClip,
                ConfigField::RllibFramework,
                ConfigField::RllibActivation,
                ConfigField::RllibBatchMode,
                ConfigField::RllibRolloutFragmentLength,
                ConfigField::RllibFcnetHiddens,
                ConfigField::RllibCheckpointFrequency,
            ],
        }
    }

    pub fn start_config_file_browser(&mut self, field: ConfigField) {
        let kind = match field {
            ConfigField::EnvPath => FileBrowserKind::ExistingFile {
                extensions: Vec::new(),
            },
            ConfigField::RllibConfigFile => FileBrowserKind::ExistingFile {
                extensions: vec!["yaml".into(), "yml".into(), "json".into()],
            },
            ConfigField::RllibResumeFrom => FileBrowserKind::Directory {
                allow_create: false,
                require_checkpoints: false,
            },
            _ => FileBrowserKind::ExistingFile {
                extensions: Vec::new(),
            },
        };
        self.start_file_browser(FileBrowserTarget::Config(field), kind, None);
    }

    pub fn start_export_file_browser(&mut self, field: ExportField) {
        let (kind, suggested_name) = match field {
            ExportField::Sb3ModelPath => (
                FileBrowserKind::ExistingFile {
                    extensions: vec!["zip".into()],
                },
                None,
            ),
            ExportField::Sb3OutputPath => (
                FileBrowserKind::OutputFile {
                    extension: Some("onnx".into()),
                },
                self.default_sb3_output_name(),
            ),
            ExportField::RllibCheckpointPath => (
                FileBrowserKind::Directory {
                    allow_create: false,
                    require_checkpoints: true,
                },
                None,
            ),
            ExportField::RllibOutputDir => (
                FileBrowserKind::Directory {
                    allow_create: true,
                    require_checkpoints: false,
                },
                None,
            ),
            _ => (
                FileBrowserKind::ExistingFile {
                    extensions: Vec::new(),
                },
                None,
            ),
        };
        self.start_file_browser(FileBrowserTarget::Export(field), kind, suggested_name);
    }

    fn default_sb3_output_name(&self) -> Option<String> {
        if !self.export_config.sb3_output_path.trim().is_empty() {
            let path = PathBuf::from(self.export_config.sb3_output_path.trim());
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                return Some(name.to_string());
            }
        }

        if !self.export_config.sb3_model_path.trim().is_empty() {
            let path = PathBuf::from(self.export_config.sb3_model_path.trim());
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                return Some(format!("{stem}.onnx"));
            }
        }

        Some(String::from("export.onnx"))
    }

    fn start_file_browser(
        &mut self,
        target: FileBrowserTarget,
        kind: FileBrowserKind,
        suggested_name: Option<String>,
    ) {
        self.input_mode = InputMode::BrowsingFiles;
        self.file_browser_target = Some(target);
        self.file_browser_kind = kind;
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_default_name = suggested_name;
        self.file_browser_input.clear();
        self.file_browser_selected = 0;

        self.file_browser_path = self.determine_browser_start_path(&target);
        self.refresh_file_browser();
    }

    fn determine_browser_start_path(&self, target: &FileBrowserTarget) -> PathBuf {
        let project_root = self
            .active_project
            .as_ref()
            .map(|project| project.path.clone())
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("/"));

        match target {
            FileBrowserTarget::Config(ConfigField::EnvPath) => self
                .resolve_existing_path(&self.training_config.env_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(ConfigField::RllibConfigFile) => self
                .resolve_existing_path(&self.training_config.rllib_config_file)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(ConfigField::RllibResumeFrom) => self
                .resolve_existing_path(&self.training_config.rllib_resume_from)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Config(_) => project_root.clone(),
            FileBrowserTarget::Export(ExportField::Sb3ModelPath) => self
                .resolve_existing_path(&self.export_config.sb3_model_path)
                .unwrap_or_else(|| project_root.clone()),
            FileBrowserTarget::Export(ExportField::Sb3OutputPath) => {
                let current = self
                    .resolve_existing_path(&self.export_config.sb3_output_path)
                    .and_then(|path| {
                        if path.is_file() {
                            path.parent().map(|p| p.to_path_buf())
                        } else {
                            Some(path)
                        }
                    });
                current
                    .or_else(|| {
                        self.resolve_existing_path(&self.export_config.sb3_model_path)
                            .and_then(|p| p.parent().map(|parent| parent.to_path_buf()))
                    })
                    .unwrap_or_else(|| project_root.clone())
            }
            FileBrowserTarget::Export(ExportField::RllibCheckpointPath) => self
                .resolve_existing_path(&self.export_config.rllib_checkpoint_path)
                .unwrap_or_else(|| project_root.join("logs")),
            FileBrowserTarget::Export(ExportField::RllibOutputDir) => self
                .resolve_existing_path(&self.export_config.rllib_output_dir)
                .unwrap_or_else(|| project_root.join("onnx_exports")),
            FileBrowserTarget::Export(_) => project_root.clone(),
        }
    }

    fn resolve_existing_path(&self, value: &str) -> Option<PathBuf> {
        if value.trim().is_empty() {
            return None;
        }
        let path = PathBuf::from(value.trim());
        if path.is_absolute() {
            Some(path)
        } else if let Some(project) = &self.active_project {
            Some(project.path.join(path))
        } else {
            std::env::current_dir().ok().map(|cwd| cwd.join(path))
        }
    }

    pub fn cancel_file_browser(&mut self) {
        self.input_mode = InputMode::Normal;
        self.file_browser_target = None;
        self.file_browser_entries.clear();
        self.file_browser_selected = 0;
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.file_browser_default_name = None;
    }

    pub fn file_browser_path(&self) -> &Path {
        &self.file_browser_path
    }

    pub fn file_browser_entries(&self) -> &[FileBrowserEntry] {
        &self.file_browser_entries
    }

    pub fn file_browser_selected(&self) -> usize {
        self.file_browser_selected
    }

    pub fn file_browser_state(&self) -> FileBrowserState {
        self.file_browser_state
    }

    pub fn file_browser_input(&self) -> &str {
        &self.file_browser_input
    }

    pub fn file_browser_kind(&self) -> &FileBrowserKind {
        &self.file_browser_kind
    }

    pub fn file_browser_select_next(&mut self) {
        if !self.file_browser_entries.is_empty() {
            self.file_browser_selected =
                (self.file_browser_selected + 1) % self.file_browser_entries.len();
        }
    }

    pub fn file_browser_select_previous(&mut self) {
        if !self.file_browser_entries.is_empty() {
            if self.file_browser_selected == 0 {
                self.file_browser_selected = self.file_browser_entries.len() - 1;
            } else {
                self.file_browser_selected -= 1;
            }
        }
    }

    pub fn file_browser_enter(&mut self) {
        if let Some(entry) = self
            .file_browser_entries
            .get(self.file_browser_selected)
            .cloned()
        {
            match entry {
                FileBrowserEntry::Parent(path) | FileBrowserEntry::Directory(path) => {
                    self.file_browser_path = path;
                    self.file_browser_selected = 0;
                    self.refresh_file_browser();
                }
                FileBrowserEntry::File(path) => {
                    if matches!(self.file_browser_kind, FileBrowserKind::ExistingFile { .. })
                        || matches!(self.file_browser_kind, FileBrowserKind::OutputFile { .. })
                    {
                        self.finalize_file_selection(path);
                    }
                }
            }
        }
    }

    pub fn file_browser_go_up(&mut self) {
        if let Some(parent) = self.file_browser_path.parent() {
            self.file_browser_path = parent.to_path_buf();
            self.file_browser_selected = 0;
            self.refresh_file_browser();
        }
    }

    pub fn file_browser_finalize_selection(&mut self) {
        match self.file_browser_kind.clone() {
            FileBrowserKind::Directory { .. } => {
                let target_path = self.file_browser_path.clone();
                self.finalize_directory_selection(target_path);
            }
            FileBrowserKind::ExistingFile { extensions } => {
                if let Some(entry) = self.file_browser_entries.get(self.file_browser_selected) {
                    if let FileBrowserEntry::File(path) = entry {
                        if self.extension_allowed(path, &extensions) {
                            self.finalize_file_selection(path.clone());
                            return;
                        }
                    }
                }
                self.set_status(
                    "Select a file that matches the required type",
                    StatusKind::Warning,
                );
            }
            FileBrowserKind::OutputFile { extension } => {
                if let Some(entry) = self.file_browser_entries.get(self.file_browser_selected) {
                    match entry {
                        FileBrowserEntry::File(path) => {
                            if extension
                                .as_ref()
                                .map(|ext| self.path_has_extension(path, ext))
                                .unwrap_or(true)
                            {
                                self.finalize_file_selection(path.clone());
                                return;
                            }
                        }
                        FileBrowserEntry::Directory(path) | FileBrowserEntry::Parent(path) => {
                            self.file_browser_path = path.clone();
                            self.file_browser_selected = 0;
                            self.refresh_file_browser();
                            self.begin_file_naming(extension);
                            return;
                        }
                    }
                }
                self.begin_file_naming(extension);
            }
        }
    }

    fn begin_file_naming(&mut self, extension: Option<String>) {
        self.file_browser_state = FileBrowserState::NamingFile;
        self.file_browser_input =
            self.file_browser_default_name
                .clone()
                .unwrap_or_else(|| match extension.as_deref() {
                    Some(ext) => format!("export.{ext}"),
                    None => String::from("export"),
                });
        if let Some(ext) = extension {
            if !self.file_browser_input.ends_with(&format!(".{ext}")) {
                self.file_browser_input.push('.');
                self.file_browser_input.push_str(&ext);
            }
        }
        self.set_status(
            "Type a file name and press Enter to confirm",
            StatusKind::Info,
        );
    }

    pub fn file_browser_begin_new_folder(&mut self) {
        self.file_browser_state = FileBrowserState::NamingFolder;
        self.file_browser_input.clear();
        self.set_status("Enter a name for the new folder", StatusKind::Info);
    }

    pub fn file_browser_push_char(&mut self, ch: char) {
        if matches!(
            self.file_browser_state,
            FileBrowserState::NamingFolder | FileBrowserState::NamingFile
        ) && !ch.is_control()
            && self.file_browser_input.len() < 128
        {
            self.file_browser_input.push(ch);
        }
    }

    pub fn file_browser_pop_char(&mut self) {
        self.file_browser_input.pop();
    }

    pub fn file_browser_cancel_input(&mut self) {
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.set_status("Selection cancelled", StatusKind::Info);
    }

    pub fn file_browser_confirm_input(&mut self) {
        match self.file_browser_state {
            FileBrowserState::NamingFolder => self.confirm_new_folder(),
            FileBrowserState::NamingFile => self.confirm_new_file(),
            FileBrowserState::Browsing => {}
        }
    }

    fn confirm_new_folder(&mut self) {
        let name = self.file_browser_input.trim();
        if name.is_empty() {
            self.set_status("Folder name cannot be empty", StatusKind::Warning);
            return;
        }
        if name.contains(std::path::MAIN_SEPARATOR) || name.contains('/') {
            self.set_status(
                "Folder name cannot contain path separators",
                StatusKind::Warning,
            );
            return;
        }
        let new_path = self.file_browser_path.join(name);
        if new_path.exists() {
            self.set_status(
                "A folder with that name already exists",
                StatusKind::Warning,
            );
            return;
        }
        if let Err(error) = fs::create_dir_all(&new_path) {
            self.set_status(
                format!("Failed to create folder: {error}"),
                StatusKind::Error,
            );
            return;
        }

        self.set_status("Folder created", StatusKind::Success);
        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.refresh_file_browser();
        if let Some(index) = self.file_browser_entries.iter().position(
            |entry| matches!(entry, FileBrowserEntry::Directory(path) if path == &new_path),
        ) {
            self.file_browser_selected = index;
        }
    }

    fn confirm_new_file(&mut self) {
        let name = self.file_browser_input.trim();
        if name.is_empty() {
            self.set_status("File name cannot be empty", StatusKind::Warning);
            return;
        }
        let extension_check =
            if let FileBrowserKind::OutputFile { extension } = &self.file_browser_kind {
                if let Some(ext) = extension {
                    if !name.ends_with(&format!(".{ext}")) {
                        self.set_status(
                            format!("File name must end with .{ext}"),
                            StatusKind::Warning,
                        );
                        return;
                    }
                }
                true
            } else {
                true
            };

        if !extension_check {
            return;
        }

        let path = self.file_browser_path.join(name);
        if let Some(parent) = path.parent() {
            if let Err(error) = fs::create_dir_all(parent) {
                self.set_status(
                    format!("Failed to prepare directory: {error}"),
                    StatusKind::Error,
                );
                return;
            }
        }

        self.file_browser_state = FileBrowserState::Browsing;
        self.file_browser_input.clear();
        self.finalize_file_selection(path);
    }

    fn finalize_directory_selection(&mut self, path: PathBuf) {
        if let Some(FileBrowserTarget::Export(ExportField::RllibCheckpointPath)) =
            self.file_browser_target
        {
            if let Err(error) = self.validate_rllib_checkpoint_dir(&path) {
                self.set_status(
                    format!("Invalid RLlib checkpoint directory: {error}"),
                    StatusKind::Warning,
                );
                return;
            }
        }

        self.apply_browser_selection(path);
    }

    fn finalize_file_selection(&mut self, path: PathBuf) {
        match self.file_browser_kind {
            FileBrowserKind::ExistingFile { ref extensions } => {
                if !self.extension_allowed(&path, extensions) {
                    self.set_status("File type not allowed", StatusKind::Warning);
                    return;
                }
            }
            FileBrowserKind::OutputFile { ref extension } => {
                if let Some(ext) = extension {
                    if !self.path_has_extension(&path, ext) {
                        self.set_status(
                            format!("File must have .{ext} extension"),
                            StatusKind::Warning,
                        );
                        return;
                    }
                }
            }
            FileBrowserKind::Directory { .. } => {}
        }

        self.apply_browser_selection(path);
    }

    fn apply_browser_selection(&mut self, path: PathBuf) {
        match self.file_browser_target {
            Some(FileBrowserTarget::Config(field)) => {
                let stored_value = self.stringify_for_storage(&path);
                if let Err(error) = self.apply_config_browser_selection(field, stored_value) {
                    self.set_status(
                        format!("Failed to apply selection: {error}"),
                        StatusKind::Error,
                    );
                    return;
                }
                self.update_validation_status();
                self.set_status("Configuration updated", StatusKind::Success);
            }
            Some(FileBrowserTarget::Export(field)) => {
                let stored_value = self.stringify_for_export(&path);
                if let Err(error) = self.apply_export_browser_selection(field, stored_value) {
                    self.set_status(
                        format!("Failed to apply selection: {error}"),
                        StatusKind::Error,
                    );
                    return;
                }
                self.set_status("Export option updated", StatusKind::Success);
                self.rebuild_export_fields();
            }
            None => {}
        }

        self.cancel_file_browser();
    }

    fn apply_config_browser_selection(&mut self, field: ConfigField, value: String) -> Result<()> {
        match field {
            ConfigField::EnvPath => {
                self.training_config.env_path = value;
            }
            ConfigField::RllibConfigFile => {
                self.training_config.rllib_config_file = value;
            }
            ConfigField::RllibResumeFrom => {
                self.training_config.rllib_resume_from = value;
            }
            _ => {}
        }
        self.persist_training_config()
    }

    fn apply_export_browser_selection(&mut self, field: ExportField, value: String) -> Result<()> {
        match field {
            ExportField::Sb3ModelPath => {
                if !value.to_lowercase().ends_with(".zip") {
                    bail!("SB3 model must be a .zip file");
                }
                self.export_config.sb3_model_path = value;
            }
            ExportField::Sb3OutputPath => {
                if !value.to_lowercase().ends_with(".onnx") {
                    bail!("SB3 output must end with .onnx");
                }
                self.export_config.sb3_output_path = value;
            }
            ExportField::RllibCheckpointPath => {
                self.validate_rllib_checkpoint_dir(&PathBuf::from(&value))?;
                self.sync_checkpoint_number_from_path(&value);
                self.export_config.rllib_checkpoint_path = value;
            }
            ExportField::RllibOutputDir => {
                self.export_config.rllib_output_dir = value;
            }
            ExportField::RllibPolicyId => {
                self.export_config.rllib_policy_id = value;
            }
            _ => {}
        }
        self.persist_export_state()?;
        Ok(())
    }

    fn stringify_for_storage(&self, path: &Path) -> String {
        if let Some(project) = &self.active_project {
            if let Ok(relative) = path.strip_prefix(&project.path) {
                return relative.to_string_lossy().to_string();
            }
        }
        path.to_string_lossy().to_string()
    }

    fn stringify_for_export(&self, path: &Path) -> String {
        if path.is_absolute() {
            return path.to_string_lossy().to_string();
        }

        if let Some(project) = &self.active_project {
            return project.path.join(path).to_string_lossy().to_string();
        }

        if let Ok(cwd) = std::env::current_dir() {
            return cwd.join(path).to_string_lossy().to_string();
        }

        path.to_string_lossy().to_string()
    }

    fn extension_allowed(&self, path: &Path, extensions: &[String]) -> bool {
        if extensions.is_empty() {
            return true;
        }
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| {
                extensions
                    .iter()
                    .any(|allowed| allowed.eq_ignore_ascii_case(ext))
            })
            .unwrap_or(false)
    }

    fn path_has_extension(&self, path: &Path, extension: &str) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case(extension))
            .unwrap_or(false)
    }

    fn refresh_file_browser(&mut self) {
        self.file_browser_entries.clear();

        if let Some(parent) = self.file_browser_path.parent() {
            self.file_browser_entries
                .push(FileBrowserEntry::Parent(parent.to_path_buf()));
        }

        if let Ok(entries) = fs::read_dir(&self.file_browser_path) {
            let mut dirs = Vec::new();
            let mut files = Vec::new();

            for entry in entries.flatten() {
                let path = entry.path();
                match entry.file_type() {
                    Ok(ft) if ft.is_dir() => dirs.push(path),
                    Ok(ft) if ft.is_file() => files.push(path),
                    _ => {}
                }
            }

            dirs.sort();
            files.sort();

            for dir in dirs {
                self.file_browser_entries
                    .push(FileBrowserEntry::Directory(dir));
            }

            match self.file_browser_kind {
                FileBrowserKind::ExistingFile { .. }
                | FileBrowserKind::OutputFile { .. }
                | FileBrowserKind::Directory { .. } => {
                    for file in files {
                        self.file_browser_entries.push(FileBrowserEntry::File(file));
                    }
                }
            }
        }

        if self.file_browser_selected >= self.file_browser_entries.len() {
            self.file_browser_selected = self.file_browser_entries.len().saturating_sub(1);
        }
    }

    pub fn validate_training_config(&self) -> Result<()> {
        if self.active_project.is_none() {
            bail!("No project selected");
        }

        if self.training_config.env_path.trim().is_empty() {
            bail!("Environment path is required");
        }

        let env_path = PathBuf::from(&self.training_config.env_path);
        if !env_path.exists() {
            bail!("Environment path does not exist: {}", env_path.display());
        }

        if !env_path.is_file() {
            bail!("Environment path must be a file: {}", env_path.display());
        }

        if self.training_config.timesteps == 0 {
            bail!("Timesteps must be greater than 0");
        }

        if self.training_config.experiment_name.trim().is_empty() {
            bail!("Experiment name cannot be empty");
        }

        if self.training_config.mode == TrainingMode::MultiAgent {
            let project_root = &self.active_project.as_ref().unwrap().path;
            let config_path = project_root.join(&self.training_config.rllib_config_file);
            if !config_path.exists() {
                bail!(
                    "RLlib config file not found: {}. Use 'g' to generate it.",
                    config_path.display()
                );
            }

            match self.training_config.rllib_stop_mode {
                RllibStopMode::TimeSeconds => {
                    if self.training_config.rllib_stop_time_seconds == 0 {
                        bail!("RLlib time limit must be at least 1 second");
                    }
                }
                RllibStopMode::Timesteps => {
                    // Already verified timesteps > 0 above
                }
            }

            let resume_trimmed = self.training_config.rllib_resume_from.trim();
            if !resume_trimmed.is_empty() {
                let resume_input = PathBuf::from(resume_trimmed);
                let resume_path = if resume_input.is_absolute() {
                    resume_input
                } else {
                    project_root.join(resume_input)
                };
                if !resume_path.exists() {
                    bail!("Resume directory not found: {}", resume_path.display());
                }
                if !resume_path.is_dir() {
                    bail!("Resume path must be a directory: {}", resume_path.display());
                }
                let tuner_file = resume_path.join("tuner.pkl");
                if !tuner_file.is_file() {
                    bail!(
                        "Resume directory is missing tuner.pkl: {}",
                        tuner_file.display()
                    );
                }
            }
        }

        Ok(())
    }

    pub fn generate_rllib_config(&mut self) -> Result<()> {
        if let Some(project) = &self.active_project {
            let config_path = project.path.join(&self.training_config.rllib_config_file);
            let existed = config_path.exists();
            match write_rllib_config(&config_path, &self.training_config) {
                Ok(()) => {
                    let message = if existed {
                        format!("Updated config: {}", config_path.display())
                    } else {
                        format!("Created config: {}", config_path.display())
                    };
                    self.set_status(message, StatusKind::Success);
                }
                Err(error) => {
                    self.set_status(
                        format!("Failed to write config: {}", error),
                        StatusKind::Error,
                    );
                    return Ok(());
                }
            }
            self.update_validation_status();
        } else {
            self.set_status("No project selected", StatusKind::Warning);
        }
        Ok(())
    }

    pub fn set_status<S: Into<String>>(&mut self, text: S, kind: StatusKind) {
        self.status = Some(StatusMessage {
            text: text.into(),
            kind,
        });
    }

    pub fn clear_status(&mut self) {
        self.status = None;
    }

    pub fn start_project_creation(&mut self) {
        self.input_mode = InputMode::CreatingProject;
        self.project_name_buffer.clear();
        self.set_status("Enter a name for the new project", StatusKind::Info);
    }

    pub fn cancel_project_creation(&mut self) {
        self.input_mode = InputMode::Normal;
        self.project_name_buffer.clear();
        self.clear_status();
    }

    pub fn push_project_name_char(&mut self, ch: char) {
        if self.project_name_buffer.len() >= 48 {
            return;
        }
        if ch.is_control() {
            return;
        }
        self.project_name_buffer.push(ch);
    }

    pub fn pop_project_name_char(&mut self) {
        self.project_name_buffer.pop();
    }

    pub fn confirm_project_creation(&mut self) -> Result<()> {
        let name = self.project_name_buffer.trim();
        if name.is_empty() {
            self.set_status("Project name cannot be empty", StatusKind::Warning);
            return Ok(());
        }

        match self.project_manager.create_project(name) {
            Ok(info) => {
                self.set_status(
                    format!("Project '{}' created", info.name),
                    StatusKind::Success,
                );
                self.input_mode = InputMode::Normal;
                self.project_name_buffer.clear();
                self.refresh_projects(Some(info.path.clone()))?;
                self.set_active_project_by_path(&info.path)?;
            }
            Err(error) => {
                self.set_status(
                    format!("Failed to create project: {error}"),
                    StatusKind::Error,
                );
            }
        }

        Ok(())
    }

    pub fn select_next_project(&mut self) {
        if self.projects.is_empty() {
            return;
        }
        self.selected_project = (self.selected_project + 1) % self.projects.len();
    }

    pub fn select_previous_project(&mut self) {
        if self.projects.is_empty() {
            return;
        }
        if self.selected_project == 0 {
            self.selected_project = self.projects.len() - 1;
        } else {
            self.selected_project -= 1;
        }
    }

    pub fn set_active_project(&mut self) -> Result<()> {
        if let Some(project) = self.selected_project().cloned() {
            self.set_active_project_inner(project)?;
        }
        Ok(())
    }

    fn set_active_project_by_path(&mut self, path: &PathBuf) -> Result<()> {
        if let Some(project) = self
            .projects
            .iter()
            .find(|info| &info.path == path)
            .cloned()
        {
            self.set_active_project_inner(project)?;
        }
        Ok(())
    }

    fn set_active_project_inner(&mut self, project: ProjectInfo) -> Result<()> {
        self.project_manager.mark_as_used(&project)?;
        self.active_project = Some(project.clone());
        let training_error = self.load_training_config_for_active_project();
        let export_error = self.load_export_state_for_active_project();
        self.refresh_projects(Some(project.path.clone()))?;
        if !training_error && !export_error {
            self.set_status(
                format!("Active project: {}", project.name),
                StatusKind::Success,
            );
        }
        Ok(())
    }

    pub fn force_refresh_projects(&mut self) -> Result<()> {
        self.refresh_projects(self.active_project.as_ref().map(|p| p.path.clone()))
    }

    pub fn start_training(&mut self) -> Result<()> {
        if self.training_running {
            self.set_status(
                "Training already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        // Validate configuration
        if let Err(e) = self.validate_training_config() {
            self.set_status(format!("Validation failed: {}", e), StatusKind::Error);
            return Ok(());
        }

        self.training_receiver = None;
        self.training_cancel = None;
        self.training_running = true;
        self.training_metrics.clear();
        self.metrics_history_index = 0;
        self.metrics_chart_index = 0;
        let now = Instant::now();
        self.metric_timer_start = Some(now);
        self.metric_last_sample_time = Some(now);
        self.reset_training_output_scroll();

        let mode_name = match self.training_config.mode {
            TrainingMode::SingleAgent => "Single-Agent (SB3)",
            TrainingMode::MultiAgent => "Multi-Agent (RLlib)",
        };
        self.set_status(
            format!("Starting {} training...", mode_name),
            StatusKind::Info,
        );

        let (tx, rx) = mpsc::channel();
        self.training_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.training_cancel = Some(cancel_tx);

        let command = determine_python_command();
        let cwd = self.active_project.as_ref().unwrap().path.clone();

        let (script_path, args) = match self.training_config.mode {
            TrainingMode::SingleAgent => {
                let script = find_script(&cwd, "stable_baselines3_training_script.py")?;
                let mut args = vec![
                    format!("--env_path={}", self.training_config.env_path),
                    format!("--experiment_dir=logs/sb3"),
                    format!("--experiment_name={}", self.training_config.experiment_name),
                    format!("--timesteps={}", self.training_config.timesteps),
                    format!("--speedup={}", self.training_config.sb3_speedup),
                    format!("--n_parallel={}", self.training_config.sb3_n_parallel),
                ];
                if self.training_config.sb3_viz {
                    args.push("--viz".to_string());
                }
                (script, args)
            }
            TrainingMode::MultiAgent => {
                let script = find_script(&cwd, "rllib_training_script.py")?;
                let mut args = vec![
                    format!("--config_file={}", self.training_config.rllib_config_file),
                    format!("--experiment_dir=logs/rllib"),
                ];
                if !self.training_config.rllib_resume_from.trim().is_empty() {
                    args.push(format!(
                        "--resume={}",
                        self.training_config.rllib_resume_from
                    ));
                }
                (script, args)
            }
        };

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        if !self.training_output.is_empty() {
            self.append_training_line(String::new());
        }
        self.append_training_line(format!("$ {}", display_cmd));

        spawn_training_task(tx, command, script_path, args, cwd, cancel_rx);

        Ok(())
    }

    pub fn start_demo_training(&mut self) -> Result<()> {
        if self.training_running {
            self.set_status(
                "Training already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        self.training_receiver = None;
        self.training_cancel = None;
        self.training_running = true;
        self.training_metrics.clear();
        self.metrics_history_index = 0;
        self.metrics_chart_index = 0;
        let now = Instant::now();
        self.metric_timer_start = Some(now);
        self.metric_last_sample_time = Some(now);
        self.reset_training_output_scroll();
        self.set_status("Starting demo training run...", StatusKind::Info);

        let (tx, rx) = mpsc::channel();
        self.training_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.training_cancel = Some(cancel_tx);

        let command = determine_python_command();
        let cwd = std::env::current_dir().wrap_err("failed to determine current directory")?;
        let mut script_path = cwd.join("demo.py");
        let mut workdir = cwd.clone();

        if !script_path.exists() {
            if let Some(parent) = cwd.parent() {
                let candidate = parent.join("demo.py");
                if candidate.exists() {
                    script_path = candidate;
                    workdir = parent.to_path_buf();
                }
            }
        }

        let script_path = script_path;
        let workdir = workdir;
        let args = Vec::new(); // No args for demo
        let display_cmd = format!("{} -u {}", command, script_path.display());

        if !self.training_output.is_empty() {
            self.append_training_line(String::new());
        }
        self.append_training_line(format!("$ {}", display_cmd));

        spawn_training_task(tx, command, script_path, args, workdir, cancel_rx);

        Ok(())
    }

    pub fn cancel_training(&mut self) {
        if let Some(cancel) = self.training_cancel.take() {
            let _ = cancel.send(());
            self.append_training_line("! Cancellation requested by user");
            self.set_status("Stopping training...", StatusKind::Info);
        }
    }

    pub fn start_export(&mut self) -> Result<()> {
        if self.export_running {
            self.set_status(
                "Export already running. Please wait for it to finish.",
                StatusKind::Warning,
            );
            return Ok(());
        }

        let project = match self.active_project.as_ref() {
            Some(project) => project.clone(),
            None => {
                self.set_status("No project selected", StatusKind::Warning);
                return Ok(());
            }
        };

        self.export_receiver = None;
        self.export_cancel = None;
        self.export_running = true;
        self.reset_export_output_scroll();
        self.export_output.clear();

        let command = determine_python_command();
        let workdir = project.path.clone();
        let (tx, rx) = mpsc::channel();
        self.export_receiver = Some(rx);
        let (cancel_tx, cancel_rx) = mpsc::channel();
        self.export_cancel = Some(cancel_tx);

        let (script_path, args) = match self.export_mode {
            ExportMode::StableBaselines3 => self.prepare_sb3_export_args(&project)?,
            ExportMode::Rllib => self.prepare_rllib_export_args(&project)?,
        };

        let display_cmd = format!(
            "{} -u {} {}",
            command,
            script_path.display(),
            args.join(" ")
        );

        self.append_export_line(format!("$ {display_cmd}"));
        spawn_export_task(tx, command, script_path, args, workdir, cancel_rx);
        self.set_status("Export started...", StatusKind::Info);

        Ok(())
    }

    pub fn cancel_export(&mut self) {
        if let Some(cancel) = self.export_cancel.take() {
            let _ = cancel.send(());
            self.append_export_line("! Cancellation requested by user");
            self.set_status("Stopping export...", StatusKind::Info);
        }
    }

    fn prepare_sb3_export_args(&self, project: &ProjectInfo) -> Result<(PathBuf, Vec<String>)> {
        let model_path_str = self.export_config.sb3_model_path.trim();
        if model_path_str.is_empty() {
            bail!("SB3 model path is required");
        }
        let model_path = self.resolve_project_path(project, model_path_str);
        if !model_path.exists() {
            bail!("Model file not found: {}", model_path.display());
        }

        let output_path = if self.export_config.sb3_output_path.trim().is_empty() {
            model_path.with_extension("onnx")
        } else {
            self.resolve_project_path(project, self.export_config.sb3_output_path.trim())
        };

        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent).wrap_err_with(|| {
                format!("failed to create export directory {}", parent.display())
            })?;
        }

        let script_path = find_script(&project.path, "convert_sb3_to_onnx.py")?;

        let mut args = Vec::new();
        args.push(model_path.to_string_lossy().to_string());
        args.push("--output".to_string());
        args.push(output_path.to_string_lossy().to_string());
        if !self.export_config.sb3_algo.trim().is_empty() {
            args.push("--algo".to_string());
            args.push(self.export_config.sb3_algo.trim().to_string());
        }
        args.push("--opset".to_string());
        args.push(self.export_config.sb3_opset.to_string());
        args.push("--ir-version".to_string());
        args.push(self.export_config.sb3_ir_version.to_string());
        if self.export_config.sb3_use_obs_array {
            args.push("--use-obs-array".to_string());
        }
        if self.export_config.sb3_skip_verify {
            args.push("--no-verify".to_string());
        }

        Ok((script_path, args))
    }

    fn prepare_rllib_export_args(&self, project: &ProjectInfo) -> Result<(PathBuf, Vec<String>)> {
        let checkpoint_path_str = self.export_config.rllib_checkpoint_path.trim();
        if checkpoint_path_str.is_empty() {
            bail!("RLlib checkpoint path is required");
        }
        let checkpoint_path = self.resolve_project_path(project, checkpoint_path_str);
        if !checkpoint_path.exists() {
            bail!("Checkpoint path not found: {}", checkpoint_path.display());
        }

        let is_direct_checkpoint = checkpoint_path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("checkpoint_"))
            .unwrap_or(false);

        if !is_direct_checkpoint {
            self.validate_rllib_checkpoint_dir(&checkpoint_path)?;
            if let Some(number) = self.export_config.rllib_checkpoint_number {
                self.ensure_rllib_checkpoint_number_exists(&checkpoint_path, number)?;
            }
        }

        let output_dir = if self.export_config.rllib_output_dir.trim().is_empty() {
            project.path.join("onnx_exports")
        } else {
            self.resolve_project_path(project, self.export_config.rllib_output_dir.trim())
        };
        fs::create_dir_all(&output_dir).wrap_err_with(|| {
            format!("failed to create export directory {}", output_dir.display())
        })?;

        let script_path = find_script(&project.path, "convert_rllib_to_onnx.py")?;

        let mut args = Vec::new();
        args.push(checkpoint_path.to_string_lossy().to_string());
        args.push("--output".to_string());
        args.push(output_dir.to_string_lossy().to_string());
        args.push("--opset".to_string());
        args.push(self.export_config.rllib_opset.to_string());
        args.push("--ir-version".to_string());
        args.push(self.export_config.rllib_ir_version.to_string());
        if !is_direct_checkpoint {
            if let Some(number) = self.export_config.rllib_checkpoint_number {
                args.push("--checkpoint-number".to_string());
                args.push(number.to_string());
            }
        }
        if !self.export_config.rllib_multiagent {
            args.push("--no-multiagent".to_string());
        }
        if !self.export_config.rllib_policy_id.trim().is_empty() {
            args.push("--policy".to_string());
            args.push(self.export_config.rllib_policy_id.trim().to_string());
        }

        Ok((script_path, args))
    }

    fn validate_rllib_checkpoint_dir(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            bail!("Path does not exist: {}", path.display());
        }
        if !path.is_dir() {
            bail!("Path is not a directory: {}", path.display());
        }

        let is_direct_checkpoint = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.starts_with("checkpoint_"))
            .unwrap_or(false);

        if is_direct_checkpoint {
            return Ok(());
        }

        let mut checkpoint_dirs = 0;
        let entries = fs::read_dir(path)
            .wrap_err_with(|| format!("failed to read checkpoint directory {}", path.display()))?;

        for entry in entries.flatten() {
            let entry_path = entry.path();
            if entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                if let Some(name) = entry_path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("checkpoint_") {
                        checkpoint_dirs += 1;
                    }
                }
            }
        }

        if checkpoint_dirs == 0 {
            bail!("No checkpoint_* directories found in {}", path.display());
        }

        Ok(())
    }

    fn ensure_rllib_checkpoint_number_exists(&self, path: &Path, number: u32) -> Result<()> {
        let padded_name = format!("checkpoint_{number:06}");
        let padded_path = path.join(&padded_name);
        if padded_path.is_dir() {
            return Ok(());
        }

        let simple_name = format!("checkpoint_{number}");
        let simple_path = path.join(&simple_name);
        if simple_path.is_dir() {
            return Ok(());
        }

        let entries = fs::read_dir(path)
            .wrap_err_with(|| format!("failed to read checkpoint directory {}", path.display()))?;

        for entry in entries.flatten() {
            if !entry.file_type().map(|ft| ft.is_dir()).unwrap_or(false) {
                continue;
            }
            let name_os = entry.file_name();
            let Some(name) = name_os.to_str() else {
                continue;
            };
            if let Some(suffix) = name.strip_prefix("checkpoint_") {
                if suffix.parse::<u32>().ok() == Some(number) {
                    return Ok(());
                }
            }
        }

        bail!(
            "Checkpoint checkpoint_{:06} not found in {}",
            number,
            path.display()
        );
    }

    fn sync_checkpoint_number_from_path(&mut self, value: &str) {
        let path = Path::new(value);
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if let Some(suffix) = name.strip_prefix("checkpoint_") {
                if let Ok(number) = suffix.parse::<u32>() {
                    self.export_config.rllib_checkpoint_number = Some(number);
                }
            }
        }
    }

    fn resolve_project_path(&self, project: &ProjectInfo, value: &str) -> PathBuf {
        let path = PathBuf::from(value);
        if path.is_absolute() {
            path
        } else {
            project.path.join(path)
        }
    }

    fn append_export_line(&mut self, line: impl Into<String>) {
        self.export_output.push(line.into());
        if self.export_output.len() > EXPORT_BUFFER_LIMIT {
            let overflow = self.export_output.len() - EXPORT_BUFFER_LIMIT;
            self.export_output.drain(0..overflow);
        }
        self.clamp_export_output_scroll();
    }

    fn clamp_export_output_scroll(&mut self) {
        if self.export_output.is_empty() {
            self.export_output_scroll = 0;
            return;
        }
        let max_offset = self.export_output.len().saturating_sub(1);
        if self.export_output_scroll > max_offset {
            self.export_output_scroll = max_offset;
        }
    }

    pub fn process_background_tasks(&mut self) {
        let mut events = Vec::new();
        let mut disconnected = false;
        let mut export_events = Vec::new();
        let mut export_disconnected = false;

        if let Some(rx) = self.training_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(TrainingEvent::Finished(code)) => {
                        events.push(TrainingEvent::Finished(code));
                        break;
                    }
                    Ok(event) => events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        disconnected = true;
                        break;
                    }
                }
            }
        }

        if let Some(rx) = self.export_receiver.as_ref() {
            loop {
                match rx.try_recv() {
                    Ok(ExportEvent::Finished(code)) => {
                        export_events.push(ExportEvent::Finished(code));
                        break;
                    }
                    Ok(event) => export_events.push(event),
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        export_disconnected = true;
                        break;
                    }
                }
            }
        }

        let mut finished = false;
        for event in events {
            match event {
                TrainingEvent::Line(line) => self.handle_training_line(line),
                TrainingEvent::Error(message) => {
                    self.append_training_line(format!("! {}", message));
                    self.set_status(message, StatusKind::Error);
                }
                TrainingEvent::Finished(code) => {
                    finished = true;
                    let message = match code {
                        Some(0) => "Training completed successfully.".to_string(),
                        Some(code) => format!("Training finished with exit code {code}."),
                        None => "Training finished.".to_string(),
                    };
                    self.set_status(message.clone(), StatusKind::Info);
                    self.append_training_line(message);
                }
            }
        }
        if finished || disconnected {
            self.training_running = false;
            self.training_receiver = None;
            self.training_cancel = None;
            if disconnected {
                self.set_status(
                    "Training task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
        }

        let mut export_finished = false;
        for event in export_events {
            match event {
                ExportEvent::Line(line) => self.append_export_line(line),
                ExportEvent::Error(message) => {
                    self.append_export_line(format!("! {message}"));
                    self.set_status(message, StatusKind::Error);
                }
                ExportEvent::Finished(code) => {
                    export_finished = true;
                    let message = match code {
                        Some(0) => "Export completed successfully.".to_string(),
                        Some(code) => format!("Export finished with exit code {code}."),
                        None => "Export finished.".to_string(),
                    };
                    self.append_export_line(message.clone());
                    self.set_status(message, StatusKind::Info);
                }
            }
        }
        if export_finished || export_disconnected {
            self.export_running = false;
            self.export_receiver = None;
            self.export_cancel = None;
            if export_disconnected {
                self.set_status(
                    "Export task disconnected unexpectedly.",
                    StatusKind::Warning,
                );
            }
        }
    }

    fn refresh_projects(&mut self, prefer_path: Option<PathBuf>) -> Result<()> {
        self.projects = self.project_manager.list_projects()?;
        if let Some(path) = prefer_path {
            if let Some(index) = self.projects.iter().position(|info| info.path == path) {
                self.selected_project = index;
            }
        }
        self.ensure_selection_valid();
        Ok(())
    }

    fn ensure_selection_valid(&mut self) {
        if self.projects.is_empty() {
            self.selected_project = 0;
        } else if self.selected_project >= self.projects.len() {
            self.selected_project = self.projects.len() - 1;
        }
    }

    fn append_training_line(&mut self, line: impl Into<String>) {
        self.training_output.push(line.into());
        if self.training_output.len() > TRAINING_BUFFER_LIMIT {
            let overflow = self.training_output.len() - TRAINING_BUFFER_LIMIT;
            self.training_output.drain(0..overflow);
        }
        self.clamp_training_output_scroll();
    }

    fn clamp_training_output_scroll(&mut self) {
        if self.training_output.is_empty() {
            self.training_output_scroll = 0;
            return;
        }
        let max_offset = self.training_output.len().saturating_sub(1);
        if self.training_output_scroll > max_offset {
            self.training_output_scroll = max_offset;
        }
    }

    fn handle_training_line(&mut self, line: String) {
        if self.try_parse_metric_line(&line) {
            return;
        }
        self.append_training_line(line);
    }

    fn try_parse_metric_line(&mut self, line: &str) -> bool {
        let payload = match line.strip_prefix(METRIC_PREFIX) {
            Some(payload) => payload,
            None => return false,
        };

        if let Ok(value) = serde_json::from_str::<Value>(payload) {
            if let Some(sample) = MetricSample::from_value(
                &value,
                self.training_config.rllib_checkpoint_frequency as u64,
            ) {
                self.record_metric_sample(sample);
            }
        }
        true
    }

    fn record_metric_sample(&mut self, mut sample: MetricSample) {
        let now = Instant::now();
        let start = if let Some(start) = self.metric_timer_start {
            start
        } else {
            self.metric_timer_start = Some(now);
            now
        };

        let total_duration = now.saturating_duration_since(start).as_secs_f64().max(0.0);
        let iter_duration = if let Some(previous) = self.metric_last_sample_time {
            now.saturating_duration_since(previous)
                .as_secs_f64()
                .max(0.0)
        } else {
            total_duration
        };

        sample.time_total_s = Some(total_duration);
        sample.time_this_iter_s = Some(iter_duration);

        self.metric_last_sample_time = Some(now);
        let previous_offset = self.metrics_history_index;
        self.training_metrics.push(sample);
        if self.training_metrics.len() > TRAINING_METRIC_HISTORY_LIMIT {
            let excess = self.training_metrics.len() - TRAINING_METRIC_HISTORY_LIMIT;
            self.training_metrics.drain(0..excess);
        }
        if previous_offset != 0 {
            self.metrics_history_index = previous_offset.saturating_add(1);
        }
        if self.training_metrics.is_empty() {
            self.metrics_history_index = 0;
        } else if self.metrics_history_index >= self.training_metrics.len() {
            self.metrics_history_index = self.training_metrics.len().saturating_sub(1);
        }
        self.ensure_chart_metric_index();
    }
}

fn format_f64(value: f64) -> String {
    let mut string = format!("{value:.6}");
    while string.contains('.') && string.ends_with('0') {
        string.pop();
    }
    if string.ends_with('.') {
        string.pop();
    }
    if string.is_empty() {
        string.push('0');
    }
    string
}

fn escape_single_quotes(input: &str) -> String {
    input.replace('\'', "''")
}

fn format_usize_list(values: &[usize]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn parse_usize_list(input: &str) -> Result<Vec<usize>> {
    let mut values = Vec::new();
    for token in input.split(|c| matches!(c, ',' | ';' | ' ' | '\t')) {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let value: usize = token
            .parse()
            .wrap_err_with(|| format!("Invalid layer size '{token}'"))?;
        if value == 0 {
            bail!("Layer sizes must be greater than zero");
        }
        values.push(value);
    }
    if values.is_empty() {
        bail!("At least one layer size is required");
    }
    Ok(values)
}

impl Index<TabId> for App {
    type Output = Tab;

    fn index(&self, index: TabId) -> &Self::Output {
        self.tabs
            .iter()
            .find(|tab| tab.id == index)
            .expect("tab exists")
    }
}

fn default_projects_root() -> Result<PathBuf> {
    let mut root = std::env::current_dir().wrap_err("failed to determine current directory")?;
    root.push("projects");
    Ok(root)
}

fn determine_python_command() -> String {
    std::env::var("PYTHON")
        .or_else(|_| std::env::var("PYTHON3"))
        .unwrap_or_else(|_| "python3".to_string())
}

fn find_script(base_dir: &PathBuf, script_name: &str) -> Result<PathBuf> {
    // First check in the project directory
    let project_script = base_dir.join(script_name);
    if project_script.exists() {
        return Ok(project_script);
    }

    // Then check in the controller root (parent of projects dir)
    if let Some(parent) = base_dir.parent() {
        if let Some(grandparent) = parent.parent() {
            let root_script = grandparent.join(script_name);
            if root_script.exists() {
                return Ok(root_script);
            }
        }
    }

    // Finally check current working directory
    let cwd = std::env::current_dir().wrap_err("failed to determine current directory")?;
    let cwd_script = cwd.join(script_name);
    if cwd_script.exists() {
        return Ok(cwd_script);
    }

    bail!("Training script '{}' not found", script_name)
}

fn spawn_export_task(
    tx: Sender<ExportEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(ExportEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(ExportEvent::Finished(None));
            return;
        }

        let mut cmd = Command::new(&command);
        cmd.arg("-u")
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(ExportEvent::Error(format!(
                    "Failed to start export command: {error}"
                )));
                let _ = tx.send(ExportEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(ExportEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {line}");
                    if tx_stderr.send(ExportEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        let cancel_thread = thread::spawn(move || {
            if cancel_rx.recv().is_ok() {
                #[cfg(unix)]
                {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                }
                #[cfg(not(unix))]
                {
                    // Fallback for non-Unix systems; the process will be terminated below.
                }
            }
        });

        let status = child.wait();

        drop(cancel_thread);

        match status {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(ExportEvent::Line(message));
                let _ = tx.send(ExportEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(ExportEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(ExportEvent::Finished(None));
            }
        }
    });
}

fn spawn_training_task(
    tx: Sender<TrainingEvent>,
    command: String,
    script_path: PathBuf,
    args: Vec<String>,
    workdir: PathBuf,
    cancel_rx: Receiver<()>,
) {
    thread::spawn(move || {
        if !script_path.exists() {
            let _ = tx.send(TrainingEvent::Error(format!(
                "Script not found: {}",
                script_path.display()
            )));
            let _ = tx.send(TrainingEvent::Finished(None));
            return;
        }

        // Use -u flag to force unbuffered output from Python
        let mut cmd = Command::new(&command);
        cmd.arg("-u") // Force Python to use unbuffered mode
            .arg(&script_path)
            .args(&args)
            .current_dir(&workdir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("CONTROLLER_METRICS", "1"); // Enable metrics output

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                let _ = tx.send(TrainingEvent::Error(format!(
                    "Failed to start training command: {error}"
                )));
                let _ = tx.send(TrainingEvent::Finished(None));
                return;
            }
        };

        let pid = child.id();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        if let Some(stdout) = stdout {
            let tx_stdout = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines().flatten() {
                    if tx_stdout.send(TrainingEvent::Line(line)).is_err() {
                        break;
                    }
                }
            });
        }

        if let Some(stderr) = stderr {
            let tx_stderr = tx.clone();
            thread::spawn(move || {
                let reader = BufReader::new(stderr);
                for line in reader.lines().flatten() {
                    let formatted = format!("! {}", line);
                    if tx_stderr.send(TrainingEvent::Line(formatted)).is_err() {
                        break;
                    }
                }
            });
        }

        // Handle cancellation with SIGINT (like Ctrl+C)
        let cancel_thread = thread::spawn(move || {
            if cancel_rx.recv().is_ok() {
                // On Unix, send SIGINT to emulate keyboard interrupt
                #[cfg(unix)]
                {
                    use nix::sys::signal::{kill, Signal};
                    use nix::unistd::Pid;
                    let _ = kill(Pid::from_raw(pid as i32), Signal::SIGINT);
                }
                // On Windows, we have to use kill
                #[cfg(not(unix))]
                {
                    // This is a fallback for non-Unix systems
                    // Windows doesn't have SIGINT, so we use terminate
                }
            }
        });

        let status = child.wait();

        // Make sure cancel thread completes
        drop(cancel_thread);

        match status {
            Ok(status) => {
                let code = status.code();
                let message = match code {
                    Some(code) => format!("Process exited with code {code}."),
                    None => "Process terminated by signal.".to_string(),
                };
                let _ = tx.send(TrainingEvent::Line(message));
                let _ = tx.send(TrainingEvent::Finished(code));
            }
            Err(error) => {
                let _ = tx.send(TrainingEvent::Error(format!(
                    "Failed to wait for process: {error}"
                )));
                let _ = tx.send(TrainingEvent::Finished(None));
            }
        }
    });
}

fn write_rllib_config(path: &Path, config: &TrainingConfig) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .wrap_err_with(|| format!("failed to create directory {}", parent.display()))?;
    }

    let env_path = config.env_path.trim();
    let escaped_env_path = if env_path.is_empty() {
        String::from("''")
    } else {
        format!("'{}'", escape_single_quotes(env_path))
    };

    let show_window = if config.rllib_show_window {
        "true"
    } else {
        "false"
    };
    let lr = format_f64(config.rllib_lr);
    let lambda = format_f64(config.rllib_lambda);
    let gamma = format_f64(config.rllib_gamma);
    let vf_loss_coeff = format_f64(config.rllib_vf_loss_coeff);
    let entropy_coeff = format_f64(config.rllib_entropy_coeff);
    let clip_param_comment = format_f64(config.rllib_clip_param);
    let grad_clip_comment = format_f64(config.rllib_grad_clip);
    let rollout_fragment_length = config.rllib_rollout_fragment_length;
    let sgd_minibatch_size = config.rllib_sgd_minibatch_size;
    let num_workers = config.rllib_num_workers;
    let num_envs_per_worker = config.rllib_num_envs_per_worker;
    let train_batch_size = config.rllib_train_batch_size;
    let num_sgd_iter = config.rllib_num_sgd_iter;
    let batch_mode = &config.rllib_batch_mode;
    let framework = &config.rllib_framework;
    let checkpoint_frequency = config.rllib_checkpoint_frequency;
    let stop_line = match config.rllib_stop_mode {
        RllibStopMode::TimeSeconds => {
            format!("    time_total_s: {}", config.rllib_stop_time_seconds)
        }
        RllibStopMode::Timesteps => format!("    timesteps_total: {}", config.timesteps),
    };

    let fcnet_hiddens = if config.rllib_fcnet_hiddens.is_empty() {
        String::from("[]")
    } else {
        format!(
            "[{}]",
            config
                .rllib_fcnet_hiddens
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };

    let content = format!(
        "algorithm: PPO\n\n# Multi-agent-env setting:\n# If true:\n# - Any AIController with done = true will receive zeroes as action values until all AIControllers are done, an episode ends at that point.\n# - ai_controller.needs_reset will also be set to true every time a new episode begins (but you can ignore it in your env if needed).\n# If false:\n# - AIControllers auto-reset in Godot and will receive actions after setting done = true.\n# - Each AIController has its own episodes that can end/reset at any point.\n# Set to false if you have a single policy name for all agents set in AIControllers\nenv_is_multiagent: true\n\ncheckpoint_frequency: {checkpoint_frequency}\n\n# You can set one or more stopping criteria\nstop:\n    #episode_reward_mean: 0\n    #training_iteration: 1000\n    #timesteps_total: 10000\n{stop_line}\n\nconfig:\n    env: godot\n    env_config:\n      env_path: {escaped_env_path} # Set your env path here (exported executable from Godot) - e.g. env_path: 'env_path.exe' on Windows\n      action_repeat: 2 # Doesn't need to be set here, you can set this in sync node in Godot editor as well\n      show_window: {show_window} # Displays game window while training. Might be faster when false in some cases, turning off also reduces GPU usage if you don't need rendering.\n      speedup: 30 # Speeds up Godot physics\n\n    framework: {framework} # ONNX models exported with torch are compatible with the current Godot RL Agents Plugin\n\n    lr: {lr}\n    lambda: {lambda}\n    gamma: {gamma}\n\n    vf_loss_coeff: {vf_loss_coeff}\n    vf_clip_param: .inf\n    #clip_param: {clip_param_comment}\n    entropy_coeff: {entropy_coeff}\n    entropy_coeff_schedule: null\n    #grad_clip: {grad_clip_comment}\n\n    normalize_actions: False\n    clip_actions: True # During onnx inference we simply clip the actions to [-1.0, 1.0] range, set here to match\n\n    rollout_fragment_length: {rollout_fragment_length}\n    sgd_minibatch_size: {sgd_minibatch_size}\n    num_workers: {num_workers}\n    num_envs_per_worker: {num_envs_per_worker} # This will be set automatically if not multi-agent. If multi-agent, changing this changes how many envs to launch per worker.\n    train_batch_size: {train_batch_size}\n\n    num_sgd_iter: {num_sgd_iter}\n    batch_mode: {batch_mode}\n\n    num_gpus: 0\n    model:\n        vf_share_layers: False\n        fcnet_hiddens: {fcnet_hiddens}\n"
    );

    fs::write(path, content)
        .wrap_err_with(|| format!("failed to write RLlib config to {}", path.display()))?;

    Ok(())
}
