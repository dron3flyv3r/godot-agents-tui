use serde::{Deserialize, Serialize};

use crate::domain::projects::PROJECT_CONFIG_DIR;

pub fn default_rllib_config_file() -> String {
    format!("{}/rllib_config.yaml", PROJECT_CONFIG_DIR)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyType {
    Mlp,
    Cnn,
    Lstm,
    Grn,
}

pub const POLICY_TYPE_LIST: [PolicyType; 4] = [
    PolicyType::Mlp,
    PolicyType::Cnn,
    PolicyType::Lstm,
    PolicyType::Grn,
];

impl Default for PolicyType {
    fn default() -> Self {
        PolicyType::Mlp
    }
}

impl PolicyType {
    pub fn label(self) -> &'static str {
        match self {
            PolicyType::Mlp => "MLP",
            PolicyType::Cnn => "CNN",
            PolicyType::Lstm => "LSTM",
            PolicyType::Grn => "GRN",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            PolicyType::Mlp => "mlp",
            PolicyType::Cnn => "cnn",
            PolicyType::Lstm => "lstm",
            PolicyType::Grn => "grn",
        }
    }

    pub fn summary(self) -> &'static str {
        match self {
            PolicyType::Mlp => {
                "Fully connected network for low-dimensional state vectors; fast but lacks spatial inductive biases."
            }
            PolicyType::Cnn => {
                "1-D convolutional stack better suited for grid/temporal observations; higher capacity than an MLP."
            }
            PolicyType::Lstm => {
                "Recurrent LSTM backbone that remembers previous observations; ideal for partial observability at higher compute cost."
            }
            PolicyType::Grn => {
                "Gated residual network mixing linear context with nonlinear gating; useful when features benefit from adaptive blending."
            }
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "mlp" => Some(PolicyType::Mlp),
            "cnn" => Some(PolicyType::Cnn),
            "lstm" => Some(PolicyType::Lstm),
            "grn" => Some(PolicyType::Grn),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RllibAlgorithm {
    Ppo,
    Dqn,
    Sac,
    Appo,
    Impala,
}

pub const RLLIB_ALGORITHM_LIST: [RllibAlgorithm; 5] = [
    RllibAlgorithm::Ppo,
    RllibAlgorithm::Dqn,
    RllibAlgorithm::Sac,
    RllibAlgorithm::Appo,
    RllibAlgorithm::Impala,
];

impl Default for RllibAlgorithm {
    fn default() -> Self {
        Self::Ppo
    }
}

impl RllibAlgorithm {
    pub fn trainer_name(self) -> &'static str {
        match self {
            Self::Ppo => "PPO",
            Self::Dqn => "DQN",
            Self::Sac => "SAC",
            Self::Appo => "APPO",
            Self::Impala => "IMPALA",
        }
    }

    pub fn summary(self) -> &'static str {
        match self {
            Self::Ppo => "Balanced on-policy baseline for both discrete and continuous actions; slower to adapt than off-policy methods when data is scarce.",
            Self::Dqn => "Value-based off-policy learner for discrete actions only; excellent with small action spaces but unusable for continuous control.",
            Self::Sac => "Off-policy actor-critic tuned for continuous control and noisy rewards; can handle discrete branches but shines with continuous vectors.",
            Self::Appo => "High-throughput PPO variant that scales to many workers for either action type; still inherits PPO's on-policy sample-efficiency limits.",
            Self::Impala => "Distributed importance-sampling trainer targeting massive discrete-action workloads; great for parallel simulations but incompatible with continuous actions.",
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ppo => "ppo",
            Self::Dqn => "dqn",
            Self::Sac => "sac",
            Self::Appo => "appo",
            Self::Impala => "impala",
        }
    }

    pub fn from_str(value: &str) -> Option<Self> {
        match value.trim().to_lowercase().as_str() {
            "ppo" => Some(Self::Ppo),
            "dqn" => Some(Self::Dqn),
            "sac" => Some(Self::Sac),
            "appo" => Some(Self::Appo),
            "impala" => Some(Self::Impala),
            _ => None,
        }
    }
}

const RLLIB_ALGO_DESCRIPTION: &str = "Select which RLlib trainer to run.\n\
- PPO: Stable on-policy baseline for both discrete and continuous actions; slower than off-policy methods when data is scarce.\n\
- DQN: Value-based off-policy learner for discrete actions only; great when the action space is small, unusable for continuous control.\n\
- SAC: Off-policy actor-critic that shines on continuous control and noisy rewards; heavier compute cost and can be finicky on tiny discrete tasks.\n\
- APPO: High-throughput PPO variant that supports both action types; good for scaling across many CPUs but still has on-policy sample efficiency limits.\n\
- IMPALA: Distributed importance-sampling trainer for large discrete problems; excellent for many parallel actors, cannot operate on continuous actions.";

pub const TRAINING_CONFIG_FILENAME: &str = "training_config.json";
pub const MARS_TRAINING_CONFIG_FILENAME: &str = "mars_training_config.json";
pub const EXPORT_CONFIG_FILENAME: &str = "export_config.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    SingleAgent,
    MultiAgent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MarsTrainingConfig {
    pub env_path: String,
    pub env_name: String,
    pub method: String,
    pub algorithm: String,
    pub max_episodes: u32,
    pub max_steps_per_episode: u32,
    pub num_envs: u32,
    pub num_process: u32,
    pub batch_size: u32,
    pub learning_rate: f64,
    pub seed: i64,
    pub save_id: String,
    pub save_path: String,
    pub log_interval: u32,
}

impl Default for MarsTrainingConfig {
    fn default() -> Self {
        Self {
            env_path: String::new(),
            env_name: "godot_env".to_string(),
            method: "nfsp".to_string(),
            algorithm: "PPO".to_string(),
            max_episodes: 10_000,
            max_steps_per_episode: 1_000,
            num_envs: 1,
            num_process: 1,
            batch_size: 128,
            learning_rate: 1e-4,
            seed: 0,
            save_id: "run_0".to_string(),
            save_path: "logs/mars".to_string(),
            log_interval: 20,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RllibStopMode {
    None,
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
    pub sb3_policy_type: PolicyType,
    pub sb3_policy_layers: Vec<usize>,
    pub sb3_cnn_channels: Vec<usize>,
    pub sb3_lstm_hidden_size: usize,
    pub sb3_lstm_num_layers: usize,
    pub sb3_grn_hidden_size: usize,
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
    pub rllib_config_file: String,
    pub rllib_show_window: bool,
    pub rllib_algorithm: RllibAlgorithm,
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
    pub rllib_env_action_repeat: u32,
    pub rllib_env_speedup: u32,
    pub rllib_num_gpus: f64,
    pub rllib_max_seq_len: u32,
    pub rllib_fcnet_hiddens: Vec<usize>,
    pub rllib_policy_type: PolicyType,
    pub rllib_cnn_channels: Vec<usize>,
    pub rllib_lstm_cell_size: usize,
    pub rllib_lstm_num_layers: usize,
    pub rllib_lstm_include_prev_actions: bool,
    pub rllib_grn_hidden_size: usize,
    pub rllib_checkpoint_frequency: u32,
    pub rllib_resume_from: String,
    pub rllib_stop_mode: RllibStopMode,
    pub rllib_stop_time_seconds: u64,
    pub rllib_stop_timesteps_total: u64,
    pub rllib_stop_sustained_reward_enabled: bool,
    pub rllib_stop_sustained_reward_threshold: f64,
    pub rllib_stop_sustained_reward_window: u32,
    pub rllib_stop_file_enabled: bool,
    pub rllib_stop_file_path: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            mode: TrainingMode::SingleAgent,
            env_path: String::new(),
            timesteps: 1_000_000,
            experiment_name: "training".to_string(),
            sb3_policy_type: PolicyType::Mlp,
            sb3_policy_layers: vec![64, 64],
            sb3_cnn_channels: vec![32, 64, 64],
            sb3_lstm_hidden_size: 128,
            sb3_lstm_num_layers: 1,
            sb3_grn_hidden_size: 128,
            sb3_viz: true,
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
            rllib_config_file: default_rllib_config_file(),
            rllib_show_window: true,
            rllib_algorithm: RllibAlgorithm::Ppo,
            rllib_num_workers: 4,
            rllib_num_envs_per_worker: 1,
            rllib_train_batch_size: 1024,
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
            rllib_rollout_fragment_length: 256,
            rllib_env_action_repeat: 2,
            rllib_env_speedup: 30,
            rllib_num_gpus: 0.0,
            rllib_max_seq_len: 20,
            rllib_fcnet_hiddens: vec![64, 64],
            rllib_policy_type: PolicyType::Mlp,
            rllib_cnn_channels: vec![32, 64, 64],
            rllib_lstm_cell_size: 64,
            rllib_lstm_num_layers: 1,
            rllib_lstm_include_prev_actions: true,
            rllib_grn_hidden_size: 64,
            rllib_checkpoint_frequency: 20,
            rllib_resume_from: String::new(),
            rllib_stop_mode: RllibStopMode::TimeSeconds,
            rllib_stop_time_seconds: 1_000_000,
            rllib_stop_timesteps_total: 1_000_000,
            rllib_stop_sustained_reward_enabled: false,
            rllib_stop_sustained_reward_threshold: 0.0,
            rllib_stop_sustained_reward_window: 20,
            rllib_stop_file_enabled: true,
            rllib_stop_file_path: ".rlcontroller/STOP".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfigField {
    EnvPath,
    Timesteps,
    ExperimentName,
    Sb3PolicyType,
    Sb3Speedup,
    Sb3NParallel,
    Sb3Viz,
    Sb3PolicyLayers,
    Sb3CnnChannels,
    Sb3LstmHiddenSize,
    Sb3LstmNumLayers,
    Sb3GrnHiddenSize,
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
    RllibAlgorithm,
    RllibEnvActionRepeat,
    RllibEnvSpeedup,
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
    RllibNumGpus,
    RllibMaxSeqLen,
    RllibFcnetHiddens,
    RllibPolicyType,
    RllibCnnChannels,
    RllibLstmCellSize,
    RllibLstmNumLayers,
    RllibLstmIncludePrevActions,
    RllibGrnHiddenSize,
    RllibCheckpointFrequency,
    RllibResumeFrom,
    RllibStopMode,
    RllibStopTimeSeconds,
    RllibStopTimestepsTotal,
    RllibStopSustainedRewardEnabled,
    RllibStopSustainedRewardThreshold,
    RllibStopSustainedRewardWindow,
    RllibStopFileEnabled,
    RllibStopFilePath,

    // MARS experimental fields
    MarsEnvPath,
    MarsEnvName,
    MarsMethod,
    MarsAlgorithm,
    MarsMaxEpisodes,
    MarsMaxStepsPerEpisode,
    MarsNumEnvs,
    MarsNumProcess,
    MarsBatchSize,
    MarsLearningRate,
    MarsSeed,
    MarsSaveId,
    MarsSavePath,
    MarsLogInterval,
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
    RllibPrefix,
}

impl Default for ExportField {
    fn default() -> Self {
        Self::RllibPrefix
    }
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
            ExportField::RllibPrefix => "RLlib Prefix",
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
    pub rllib_prefix: String,
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
            rllib_prefix: String::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportState {
    pub mode: ExportMode,
    pub config: ExportConfig,
}

impl Default for ExportState {
    fn default() -> Self {
        Self {
            mode: ExportMode::default(),
            config: ExportConfig::default(),
        }
    }
}

impl ConfigField {
    pub fn label(self) -> &'static str {
        match self {
            ConfigField::EnvPath => "Environment Path",
            ConfigField::Timesteps => "Timesteps",
            ConfigField::ExperimentName => "Experiment Name",
            ConfigField::Sb3PolicyType => "SB3 Policy Type",
            ConfigField::Sb3Speedup => "SB3 Speedup",
            ConfigField::Sb3NParallel => "SB3 Parallel Envs",
            ConfigField::Sb3Viz => "SB3 Visualization",
            ConfigField::Sb3PolicyLayers => "SB3 Policy Layers",
            ConfigField::Sb3CnnChannels => "SB3 CNN Channels",
            ConfigField::Sb3LstmHiddenSize => "SB3 LSTM Hidden Size",
            ConfigField::Sb3LstmNumLayers => "SB3 LSTM Layers",
            ConfigField::Sb3GrnHiddenSize => "SB3 GRN Hidden Size",
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
            ConfigField::RllibAlgorithm => "RLlib Algorithm",
            ConfigField::RllibEnvActionRepeat => "RLlib Action Repeat",
            ConfigField::RllibEnvSpeedup => "RLlib Speedup",
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
            ConfigField::RllibNumGpus => "RLlib GPUs",
            ConfigField::RllibMaxSeqLen => "RLlib Max Seq Len",
            ConfigField::RllibFcnetHiddens => "RLlib FC Layers",
            ConfigField::RllibPolicyType => "RLlib Policy Type",
            ConfigField::RllibCnnChannels => "RLlib CNN Channels",
            ConfigField::RllibLstmCellSize => "RLlib LSTM Hidden Size",
            ConfigField::RllibLstmNumLayers => "RLlib LSTM Layers",
            ConfigField::RllibLstmIncludePrevActions => "RLlib LSTM Prev Actions",
            ConfigField::RllibGrnHiddenSize => "RLlib GRN Hidden Size",
            ConfigField::RllibCheckpointFrequency => "RLlib Checkpoint Frequency",
            ConfigField::RllibResumeFrom => "RLlib Resume Directory",
            ConfigField::RllibStopMode => "RLlib Stop Mode",
            ConfigField::RllibStopTimeSeconds => "RLlib Time Limit (s)",
            ConfigField::RllibStopTimestepsTotal => "RLlib Additional Timesteps",
            ConfigField::RllibStopSustainedRewardEnabled => "RLlib Stop on Sustained Reward",
            ConfigField::RllibStopSustainedRewardThreshold => "RLlib Reward Threshold",
            ConfigField::RllibStopSustainedRewardWindow => "RLlib Reward Window (iters)",
            ConfigField::RllibStopFileEnabled => "RLlib Stop File Enabled",
            ConfigField::RllibStopFilePath => "RLlib Stop File Path",
            ConfigField::MarsEnvPath => "MARS Godot Env Path",
            ConfigField::MarsEnvName => "MARS Env Name",
            ConfigField::MarsMethod => "MARS Method",
            ConfigField::MarsAlgorithm => "MARS Algorithm",
            ConfigField::MarsMaxEpisodes => "MARS Max Episodes",
            ConfigField::MarsMaxStepsPerEpisode => "MARS Steps per Episode",
            ConfigField::MarsNumEnvs => "MARS Parallel Envs",
            ConfigField::MarsNumProcess => "MARS Processes",
            ConfigField::MarsBatchSize => "MARS Batch Size",
            ConfigField::MarsLearningRate => "MARS Learning Rate",
            ConfigField::MarsSeed => "MARS Seed",
            ConfigField::MarsSaveId => "MARS Save ID",
            ConfigField::MarsSavePath => "MARS Save Path",
            ConfigField::MarsLogInterval => "MARS Log Interval",
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
            ConfigField::Sb3PolicyType => {
                "Select which SB3 policy backbone (MLP, CNN, LSTM, GRN) to build."
            }
            ConfigField::Sb3Speedup => "Simulation speed multiplier applied while SB3 is running.",
            ConfigField::Sb3NParallel => {
                "Number of parallel environments to launch for SB3 training."
            }
            ConfigField::Sb3Viz => "Enable to render the Godot window during SB3 training.",
            ConfigField::Sb3PolicyLayers => {
                "Comma-separated hidden layer sizes for the SB3 policy network."
            }
            ConfigField::Sb3CnnChannels => {
                "Comma-separated convolution channels for SB3 CNN feature extractors."
            }
            ConfigField::Sb3LstmHiddenSize => "Hidden size used by SB3 LSTM feature extractors.",
            ConfigField::Sb3LstmNumLayers => {
                "Number of stacked layers inside the SB3 LSTM extractor."
            }
            ConfigField::Sb3GrnHiddenSize => {
                "Hidden size applied by the SB3 GRN feature extractor."
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
            ConfigField::RllibAlgorithm => RLLIB_ALGO_DESCRIPTION,
            ConfigField::RllibEnvActionRepeat => {
                "Number of times each action is repeated by the RLlib Godot environment."
            }
            ConfigField::RllibEnvSpeedup => {
                "Simulation speed multiplier applied while RLlib is running."
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
            ConfigField::RllibNumGpus => "Number of GPUs (can be fractional) reserved by RLlib.",
            ConfigField::RllibMaxSeqLen => {
                "Unroll length for RLlib RNNs (padding/truncation per sequence)."
            }
            ConfigField::RllibFcnetHiddens => {
                "Comma-separated hidden layer sizes for RLlib's fully connected net."
            }
            ConfigField::RllibPolicyType => {
                "Select which RLlib policy backbone (MLP, CNN, LSTM, GRN) to build."
            }
            ConfigField::RllibCnnChannels => {
                "Comma-separated convolution channels for RLlib CNN models."
            }
            ConfigField::RllibLstmCellSize => "Hidden size of the RLlib LSTM backbone.",
            ConfigField::RllibLstmNumLayers => {
                "Number of stacked layers inside the RLlib LSTM backbone."
            }
            ConfigField::RllibLstmIncludePrevActions => {
                "When enabled, feed the previous action vector into the RLlib LSTM."
            }
            ConfigField::RllibGrnHiddenSize => "Hidden size applied by the RLlib GRN backbone.",
            ConfigField::RllibCheckpointFrequency => {
                "Number of iterations between RLlib checkpoints."
            }
            ConfigField::RllibResumeFrom => {
                "Optional checkpoint directory to resume RLlib training from."
            }
            ConfigField::RllibStopMode => {
                "Select whether RLlib stops by elapsed time, total timesteps, or runs until manually stopped."
            }
            ConfigField::RllibStopTimeSeconds => {
                "Target duration in seconds when using time-based stopping."
            }
            ConfigField::RllibStopTimestepsTotal => {
                "Additional environment timesteps to run for this training session (added on top of the resumed total)."
            }
            ConfigField::RllibStopSustainedRewardEnabled => {
                "Stop when episode_reward_mean stays above the configured threshold for a number of iterations."
            }
            ConfigField::RllibStopSustainedRewardThreshold => {
                "Reward threshold (episode_reward_mean) used for sustained-reward stopping."
            }
            ConfigField::RllibStopSustainedRewardWindow => {
                "Number of consecutive iterations that must meet the reward threshold before stopping."
            }
            ConfigField::RllibStopFileEnabled => {
                "When enabled, training stops once the stop file exists."
            }
            ConfigField::RllibStopFilePath => {
                "Path to a stop file; create this file to request training stop."
            }
            ConfigField::MarsEnvPath => {
                "Executable path to the exported Godot environment (required for MARS)."
            }
            ConfigField::MarsEnvName => "Name used for the environment within MARS runs.",
            ConfigField::MarsMethod => "Multi-agent training method used by MARS.",
            ConfigField::MarsAlgorithm => "Underlying algorithm class MARS instantiates.",
            ConfigField::MarsMaxEpisodes => "Maximum number of episodes to run.",
            ConfigField::MarsMaxStepsPerEpisode => "Maximum steps per episode before reset.",
            ConfigField::MarsNumEnvs => "Number of parallel environments to sample.",
            ConfigField::MarsNumProcess => "Processes to launch for sampling/updating.",
            ConfigField::MarsBatchSize => "Batch size for optimizer updates.",
            ConfigField::MarsLearningRate => "Learning rate for the chosen algorithm.",
            ConfigField::MarsSeed => "Random seed for reproducibility.",
            ConfigField::MarsSaveId => "Identifier appended to model/log directories.",
            ConfigField::MarsSavePath => "Root path where MARS writes logs/checkpoints.",
            ConfigField::MarsLogInterval => "How often (episodes) to log and emit metrics.",
        }
    }
}
