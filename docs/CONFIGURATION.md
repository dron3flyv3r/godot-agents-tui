# Configuration Guide

This document provides detailed information about all configuration files and options in the Godot RL Training Controller.

## Configuration File Formats

### Project Configuration (`project.json`)

**Location**: `projects/<project-name>/project.json`

**Format**:
```json
{
  "created": 1761917546,
  "name": "my_project"
}
```

**Fields**:
- `created` (integer): Unix timestamp of project creation
- `name` (string): Project display name

**Notes**:
- Automatically created when you create a new project
- Rarely needs manual editing

---

### Training Configuration (`training_config.json`)

**Location**: `projects/<project-name>/training_config.json`

**Format**: Large JSON object with all training parameters

#### Core Settings

```json
{
  "mode": "multi_agent",           // "single_agent" or "multi_agent"
  "env_path": "/path/to/game.x86_64",
  "timesteps": 1000000,
  "experiment_name": "training"
}
```

**Core Fields**:
- `mode` (string): Training framework to use
  - `"single_agent"` - Use Stable Baselines3
  - `"multi_agent"` - Use RLlib
- `env_path` (string): Path to your Godot game executable
- `timesteps` (integer): Number of environment steps to train
- `experiment_name` (string): Name for this training run

#### Stable Baselines3 Configuration

All fields prefixed with `sb3_`:

##### Network Architecture

```json
{
  "sb3_policy_layers": [64, 64],
  "sb3_activation": "relu"
}
```

- `sb3_policy_type` (string): Backbone selector for SB3 policies (`"mlp"`, `"cnn"`, `"lstm"`, `"grn"`). Default: `"mlp"`.
- `sb3_policy_layers` (array of integers): Hidden layer sizes
  - Default: `[64, 64]`
  - Example: `[128, 128, 64]` for 3-layer network
- `sb3_cnn_channels` (array of integers): Number of filters per convolution layer when using the CNN backbone. Default: `[32, 64, 64]`.
- `sb3_lstm_hidden_size` (integer): Hidden size for the LSTM backbone. Default: `256`.
- `sb3_lstm_num_layers` (integer): Number of stacked LSTM layers. Default: `1`.
- `sb3_grn_hidden_size` (integer): Hidden size for the GRN backbone. Default: `256`.
  
##### Training Hyperparameters

```json
{
  "sb3_learning_rate": 0.0003,
  "sb3_batch_size": 64,
  "sb3_n_steps": 2048,
  "sb3_gamma": 0.99,
  "sb3_gae_lambda": 0.95,
  "sb3_ent_coef": 0.01,
  "sb3_clip_range": 0.2,
  "sb3_vf_coef": 0.5,
  "sb3_max_grad_norm": 0.5
}
```

**Field Descriptions**:

- `sb3_learning_rate` (float): Learning rate for optimizer
  - Range: `0.00001` to `0.01`
  - Default: `0.0003`
  - Higher = faster learning but less stable

- `sb3_batch_size` (integer): Minibatch size for training
  - Range: `32` to `512`
  - Default: `64`
  - Must be divisor of `sb3_n_steps * sb3_n_parallel`

- `sb3_n_steps` (integer): Steps per rollout
  - Range: `128` to `8192`
  - Default: `2048`
  - Higher = more stable but slower updates

- `sb3_gamma` (float): Discount factor
  - Range: `0.9` to `0.9999`
  - Default: `0.99`
  - How much to value future rewards

- `sb3_gae_lambda` (float): GAE lambda for advantage estimation
  - Range: `0.8` to `1.0`
  - Default: `0.95`
  - Higher = more biased, less variance

- `sb3_ent_coef` (float): Entropy coefficient
  - Range: `0.0` to `0.1`
  - Default: `0.01`
  - Encourages exploration

- `sb3_clip_range` (float): PPO clipping parameter
  - Range: `0.1` to `0.3`
  - Default: `0.2`
  - Limits policy updates

- `sb3_vf_coef` (float): Value function coefficient
  - Range: `0.1` to `1.0`
  - Default: `0.5`
  - Weight of value loss vs policy loss

- `sb3_max_grad_norm` (float): Gradient clipping
  - Range: `0.1` to `10.0`
  - Default: `0.5`
  - Prevents exploding gradients

##### Environment Settings

```json
{
  "sb3_viz": true,
  "sb3_speedup": 30,
  "sb3_n_parallel": 1
}
```

- `sb3_viz` (boolean): Show game window during training
  - `true` - Visual feedback (slower)
  - `false` - Headless mode (faster)

- `sb3_speedup` (integer): Physics speed multiplier
  - Range: `1` to `100`
  - Default: `30`
  - Higher = faster training but less accurate

- `sb3_n_parallel` (integer): Number of parallel environments
  - Range: `1` to `64`
  - Default: `1`
  - More = faster but more memory

#### RLlib Configuration

All fields prefixed with `rllib_`:

##### Core Settings

```json
{
  "rllib_config_file": ".rlcontroller/rllib_config.yaml",
  "rllib_show_window": true
}
```

- `rllib_config_file` (string): Path to RLlib YAML config
  - Relative paths are resolved against the project directory; by default the controller keeps the file under `.rlcontroller/`
  - Default: `".rlcontroller/rllib_config.yaml"`

- `rllib_show_window` (boolean): Show game window
  - `true` - Visual feedback
  - `false` - Headless mode

##### Worker Configuration

```json
{
  "rllib_num_workers": 4,
  "rllib_num_envs_per_worker": 1,
  "rllib_rollout_fragment_length": 200
}
```

- `rllib_num_workers` (integer): Number of parallel workers
  - Range: `0` to `64`
  - Default: `4`
  - 0 = single-threaded

- `rllib_num_envs_per_worker` (integer): Environments per worker
  - Range: `1` to `16`
  - Default: `1`
  - Total environments = workers × envs_per_worker

- `rllib_rollout_fragment_length` (integer): Steps per rollout
  - Range: `50` to `1000`
  - Default: `200`

##### Training Hyperparameters

```json
{
  "rllib_train_batch_size": 4000,
  "rllib_sgd_minibatch_size": 128,
  "rllib_num_sgd_iter": 30,
  "rllib_lr": 0.0003,
  "rllib_gamma": 0.99,
  "rllib_lambda": 0.95,
  "rllib_clip_param": 0.2,
  "rllib_entropy_coeff": 0.01,
  "rllib_vf_loss_coeff": 0.5,
  "rllib_grad_clip": 0.5
}
```

**Field Descriptions**:

- `rllib_train_batch_size` (integer): Total samples per training iteration
  - Range: `1000` to `100000`
  - Default: `4000`

- `rllib_sgd_minibatch_size` (integer): SGD minibatch size
  - Range: `64` to `1024`
  - Default: `128`
  - Must divide `train_batch_size`

- `rllib_num_sgd_iter` (integer): SGD iterations per training
  - Range: `1` to `50`
  - Default: `30`

- `rllib_lr` (float): Learning rate
  - Range: `0.00001` to `0.01`
  - Default: `0.0003`

- `rllib_gamma` (float): Discount factor
  - Range: `0.9` to `0.9999`
  - Default: `0.99`

- `rllib_lambda` (float): GAE lambda
  - Range: `0.8` to `1.0`
  - Default: `0.95`

- `rllib_clip_param` (float): PPO clip parameter
  - Range: `0.1` to `0.3`
  - Default: `0.2`

- `rllib_entropy_coeff` (float): Entropy coefficient
  - Range: `0.0` to `0.1`
  - Default: `0.01`

- `rllib_vf_loss_coeff` (float): Value function coefficient
  - Range: `0.1` to `1.0`
  - Default: `0.5`

- `rllib_grad_clip` (float): Gradient clipping
  - Range: `0.1` to `10.0`
  - Default: `0.5`

##### Network Architecture

```json
{
  "rllib_framework": "torch",
  "rllib_activation": "relu",
  "rllib_policy_type": "mlp",
  "rllib_fcnet_hiddens": [64, 64]
}
```

- `rllib_framework` (string): Deep learning framework
  - `"torch"` - PyTorch (recommended)
  - `"tf2"` - TensorFlow 2

- `rllib_activation` (string): Activation function
  - `"relu"` - ReLU (default)
  - `"tanh"` - Tanh
  - `"elu"` - ELU

- `rllib_policy_type` (string): Selects the policy backbone (`"mlp"`, `"cnn"`, `"lstm"`, `"grn"`). Default: `"mlp"`.
- `rllib_fcnet_hiddens` (array of integers): Hidden layer sizes shared by the value/policy heads
  - Default: `[64, 64]`
  - Example: `[256, 256, 128]`
- `rllib_cnn_channels` (array of integers): Convolution filters per layer when using the CNN policy type. Default: `[32, 64, 64]`.
- `rllib_lstm_cell_size` (integer): LSTM hidden size for the LSTM policy type. Default: `256`.
- `rllib_lstm_num_layers` (integer): Number of stacked LSTM layers. Default: `1`.
- `rllib_grn_hidden_size` (integer): Hidden size used by the GRN backbone. Default: `256`.

##### Batch Mode

```json
{
  "rllib_batch_mode": "truncate_episodes"
}
```

- `rllib_batch_mode` (string): How to batch episodes
  - `"truncate_episodes"` - Cut episodes at rollout length
  - `"complete_episodes"` - Only complete episodes

##### Checkpointing

```json
{
  "rllib_checkpoint_frequency": 20,
  "rllib_resume_from": ""
}
```

- `rllib_checkpoint_frequency` (integer): Save every N iterations
  - Range: `1` to `1000`
  - Default: `20`
  - 0 = no checkpoints

- `rllib_resume_from` (string): Resume from checkpoint path
  - Empty = start fresh
  - Path to checkpoint directory = resume

##### Stopping Criteria

```json
{
  "rllib_stop_mode": "time_seconds",
  "rllib_stop_time_seconds": 1000000
}
```

- `rllib_stop_mode` (string): When to stop training
  - `"time_seconds"` - Stop after time limit
  - `"training_iteration"` - Stop after N iterations
  - `"timesteps_total"` - Stop after N timesteps

- `rllib_stop_time_seconds` (integer): Time limit in seconds
  - Only used if `stop_mode` is `"time_seconds"`

---

### RLlib YAML Configuration (`rllib_config.yaml`)

**Location**: `projects/<project-name>/.rlcontroller/rllib_config.yaml`

**Purpose**: Defines environment setup and multi-agent policies

**Example**:

```yaml
env: godot

# Environment configuration
env_config:
  env_path: null  # Will be overridden by training_config.json
  show_window: true
  seed: 0
  speedup: 1

# Multi-agent setup
multiagent:
  policies:
    left_policy:
      # Policy configuration
    right_policy:
      # Policy configuration
  
  policy_mapping_fn: |
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "left":
            return "left_policy"
        else:
            return "right_policy"
  
  policies_to_train:
    - left_policy
    - right_policy
```

**Sections**:

#### Environment Config

```yaml
env_config:
  env_path: "/path/to/game.x86_64"  # Godot binary
  show_window: true                  # Visual feedback
  seed: 0                            # Random seed
  speedup: 1                         # Physics speed
```

#### Multi-Agent Policies

```yaml
multiagent:
  policies:
    policy_name:
      # Leave empty for default config
      # Or specify custom obs/action spaces
    
    custom_policy:
      observation_space: ...
      action_space: ...
```

#### Policy Mapping

```yaml
multiagent:
  policy_mapping_fn: |
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Map agent IDs to policy names
        if "left" in agent_id:
            return "left_policy"
        return "right_policy"
```

**Options for policy_mapping_fn**:
- Return different policies based on `agent_id`
- Use `episode` for curriculum learning
- Dynamic policy selection

#### Policies to Train

```yaml
multiagent:
  policies_to_train:
    - left_policy
    - right_policy
```

- List of policies to update during training
- Other policies remain frozen
- Useful for self-play or curriculum learning

#### Custom Policy Models (RLlib)

The `model` section of `rllib_config.yaml` now plugs directly into the bundled Torch architectures found in `custom_models/rllib_models.py`. Set `custom_model` to switch bodies and describe overrides inside `custom_model_config`:

```yaml
model:
  vf_share_layers: false
  custom_model: tui_lstm        # tui_lstm, tui_grn, or tui_cnn
  custom_model_config:
    hidden_size: 64             # Body width
    num_layers: 1               # LSTM depth only
    fcnet_hiddens: [128, 128]   # Shared policy/value head
```

Supported models:

- `tui_lstm` – Adds an internal LSTM so each policy can learn temporal context. Tunable keys: `hidden_size`, `num_layers`, `fcnet_hiddens`.
- `tui_grn` – Lightweight gating residual network that mixes candidate/context features. Tunable keys: `hidden_size`, `fcnet_hiddens`.
- `tui_cnn` – 1D convolutional stack for ordered observation vectors. Tunable keys: `channels` (per-layer widths), `fcnet_hiddens`. Treat this as experimental unless your env exposes spatial/temporal grids.

All models end with a configurable fully connected head whose layers are controlled by `fcnet_hiddens` (defaults to `[256, 256]`). If `custom_model_config` is omitted, the helper use defaults baked into the Python module. You can also register your own model names by extending `custom_models/rllib_models.py` and calling `register_rllib_models()`.

---

### Export Configuration (`export_config.json`)

**Location**: `projects/<project-name>/export_config.json`

**Purpose**: Settings for ONNX model export

**Format**:

```json
{
  "mode": "rllib",
  "config": {
    "sb3_model_path": "",
    "sb3_output_path": "",
    "sb3_algo": "",
    "sb3_opset": 13,
    "sb3_ir_version": 9,
    "sb3_use_obs_array": false,
    "sb3_skip_verify": false,
    "rllib_checkpoint_path": "/path/to/checkpoint",
    "rllib_checkpoint_number": null,
    "rllib_output_dir": "/output/path",
    "rllib_policy_id": "",
    "rllib_opset": 13,
    "rllib_ir_version": 9,
    "rllib_multiagent": true
  }
}
```

#### Mode Selection

- `mode` (string): Export framework
  - `"sb3"` - Stable Baselines3 export
  - `"rllib"` - RLlib export

#### SB3 Export Settings

- `sb3_model_path` (string): Path to `.zip` model file
- `sb3_output_path` (string): Where to save `.onnx` file
- `sb3_algo` (string): Algorithm name
  - Auto-detected if empty
  - Options: `"ppo"`, `"a2c"`, `"dqn"`, `"sac"`, `"td3"`, `"ddpg"`
- `sb3_opset` (integer): ONNX opset version
  - Default: `13`
  - Range: `9` to `17`
- `sb3_ir_version` (integer): ONNX IR version
  - Default: `9`
- `sb3_use_obs_array` (boolean): Use array input format
  - Default: `false`
- `sb3_skip_verify` (boolean): Skip ONNX verification
  - Default: `false`

#### RLlib Export Settings

- `rllib_checkpoint_path` (string): Path to checkpoint directory
  - Can be training run directory (auto-finds latest)
  - Or specific checkpoint path
- `rllib_checkpoint_number` (integer or null): Specific checkpoint
  - `null` - Use latest checkpoint
  - Integer - Use checkpoint N
- `rllib_output_dir` (string): Directory for exported models
  - Creates one `.onnx` file per policy
- `rllib_policy_id` (string): Export specific policy
  - Empty - Export all policies
  - Policy name - Export only that policy
- `rllib_opset` (integer): ONNX opset version
  - Default: `13`
- `rllib_ir_version` (integer): ONNX IR version
  - Default: `9`
- `rllib_multiagent` (boolean): Multi-agent mode
  - Default: `true`

---

### Project Index (`projects/index.json`)

**Location**: `projects/index.json`

**Purpose**: Registry of all projects

**Format**:
```json
[
  {
    "name": "project_name",
    "path": "/absolute/path/to/project",
    "last_used": 1762350919
  }
]
```

**Fields**:
- `name` (string): Project identifier
- `path` (string): Absolute path to project directory
- `last_used` (integer): Unix timestamp of last access

**Notes**:
- Automatically managed by controller
- Sorted by last_used (most recent first)
- Safe to manually edit for corrections

---

## Configuration Best Practices

### 1. Start with Defaults

Don't change too many parameters at once. Train with defaults first, then adjust based on results.

### 2. Hyperparameter Tuning Order

1. **Learning Rate**: Most important parameter
   - Too high: Training unstable
   - Too low: Training too slow

2. **Network Architecture**: Adjust layer sizes
   - Larger networks: More capacity, slower training
   - Smaller networks: Faster, may underfit

3. **Batch Size / Steps**: Trade off stability vs speed
   - Larger batches: More stable, slower iteration
   - Smaller batches: Faster iteration, noisier

4. **Exploration**: Entropy coefficient
   - Higher: More exploration
   - Lower: More exploitation

### 3. Common Configuration Patterns

#### Fast Iteration (Debugging)
```json
{
  "timesteps": 100000,
  "sb3_speedup": 50,
  "sb3_n_parallel": 4,
  "sb3_n_steps": 512,
  "sb3_viz": false
}
```

#### Stable Training (Production)
```json
{
  "timesteps": 10000000,
  "sb3_speedup": 10,
  "sb3_n_parallel": 1,
  "sb3_n_steps": 2048,
  "sb3_viz": false
}
```

#### Visual Debugging
```json
{
  "timesteps": 100000,
  "sb3_speedup": 1,
  "sb3_n_parallel": 1,
  "sb3_viz": true
}
```

### 4. Configuration Validation

The controller validates:
- File paths exist
- Numeric ranges are reasonable
- Required fields are present

But you should also:
- Test configurations on small runs first
- Keep notes on what works
- Version control your configs

### 5. Sharing Configurations

To share a configuration:
1. Copy the entire project directory
2. Update `env_path` for new machine
3. Clear `logs/` if not needed
4. Update `export_config.json` paths

---

## Environment Variables

The controller respects these environment variables:

### `CONTROLLER_METRICS`

**Purpose**: Enable structured metric emission

**Set by**: Controller automatically sets this when launching training scripts

**Value**: `"controller"` or similar identifier

**Usage**: Training scripts check this to decide whether to emit `@METRIC` lines

### `PYTHONUNBUFFERED`

**Purpose**: Disable Python stdout buffering

**Set by**: Controller (implicitly through process spawning)

**Effect**: Ensures real-time output streaming

---

## Troubleshooting Configuration Issues

### Configuration Not Loading

**Symptoms**: Settings reset on restart

**Solutions**:
1. Check file permissions
2. Verify JSON syntax (use validator)
3. Check controller logs for parse errors

### Training Uses Wrong Settings

**Symptoms**: Training ignores your configuration

**Solutions**:
1. Verify you saved (`s` key in Train tab)
2. Check correct project is selected
3. Ensure config file in correct location
4. Verify training script reads the config

### Export Fails with Path Errors

**Symptoms**: Can't find checkpoint or model

**Solutions**:
1. Use absolute paths
2. Use file browser (`b` key)
3. Check checkpoint directory structure
4. Verify checkpoint number exists

### RLlib Config Not Applied

**Symptoms**: RLlib uses different settings

**Solutions**:
1. Verify `.rlcontroller/rllib_config.yaml` exists in the project
2. Check `rllib_config_file` path in `training_config.json`
3. Ensure YAML syntax is valid
4. Review RLlib logs for errors

---

## Configuration Schema Reference

For JSON schema validation, refer to the Rust structs in `src/project.rs`:

- `ProjectInfo`
- `TrainingConfig`
- `ExportConfig`

These define the canonical configuration format.

---

**Last Updated**: November 5, 2025
