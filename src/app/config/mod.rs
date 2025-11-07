use serde::{Deserialize, Serialize};

pub const TRAINING_CONFIG_FILENAME: &str = "training_config.json";
pub const EXPORT_CONFIG_FILENAME: &str = "export_config.json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainingMode {
    SingleAgent,
    MultiAgent,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
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
