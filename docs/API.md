# Python Scripts API Reference

This document provides detailed information about the Python training and export scripts used by the Godot RL Training Controller.

## Overview

The controller uses Python scripts to handle the actual training and export operations. These scripts are based on [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) with modifications for enhanced metric emission and controller integration.

---

## Training Scripts

### `stable_baselines3_training_script.py`

Single-agent reinforcement learning using Stable Baselines3.

#### Command Line Interface

```bash
python stable_baselines3_training_script.py [OPTIONS]
```

#### Arguments

##### Environment Configuration

**`--env_path PATH`**
- **Type**: String
- **Default**: None (trains in-editor)
- **Required**: Yes (for training with exported Godot binary)
- **Description**: Path to your Godot game executable
- **Example**: `--env_path ./BallChase.x86_64`

**`--speedup MULTIPLIER`**
- **Type**: Integer
- **Default**: 1
- **Range**: 1-100
- **Description**: Physics speed multiplier for faster training
- **Example**: `--speedup 30`

**`--n_parallel COUNT`**
- **Type**: Integer
- **Default**: 1
- **Range**: 1-64
- **Description**: Number of parallel environment instances
- **Example**: `--n_parallel 4`

**`--viz`**
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Show visualization window during training
- **Example**: `--viz`

##### Training Configuration

**`--timesteps N`**
- **Type**: Integer
- **Default**: 1,000,000
- **Description**: Total number of environment steps to train
- **Example**: `--timesteps 5000000`

**`--experiment_dir PATH`**
- **Type**: String
- **Default**: `logs/sb3`
- **Description**: Directory for logs and checkpoints
- **Example**: `--experiment_dir ./my_logs`

**`--experiment_name NAME`**
- **Type**: String
- **Default**: `experiment`
- **Description**: Name for this training run (used in logs)
- **Example**: `--experiment_name pong_v1`

**`--seed SEED`**
- **Type**: Integer
- **Default**: 0
- **Description**: Random seed for reproducibility
- **Example**: `--seed 42`

##### Model Management

**`--resume_model_path PATH`**
- **Type**: String
- **Default**: None
- **Description**: Resume training from saved model
- **Example**: `--resume_model_path ./models/checkpoint.zip`

**`--save_model_path PATH`**
- **Type**: String
- **Default**: None
- **Description**: Save trained model to this path
- **Example**: `--save_model_path ./final_model.zip`

**`--save_checkpoint_frequency N`**
- **Type**: Integer
- **Default**: None (no checkpoints)
- **Description**: Save checkpoints every N steps
- **Example**: `--save_checkpoint_frequency 100000`

**`--onnx_export_path PATH`**
- **Type**: String
- **Default**: None
- **Description**: Export to ONNX after training
- **Example**: `--onnx_export_path ./model.onnx`

##### Advanced Options

**`--inference`**
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Run inference instead of training
- **Requires**: `--resume_model_path`
- **Example**: `--inference`

**`--linear_lr_schedule`**
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Use linear learning rate decay
- **Example**: `--linear_lr_schedule`

#### Environment Variables

**`CONTROLLER_METRICS`**
- **Purpose**: Enable structured metric emission
- **Format**: Any truthy string (e.g., `"controller"`)
- **Effect**: Script emits `@METRIC` prefixed JSON lines
- **Set by**: Controller automatically

#### Output Format

##### Standard Output

Regular training logs:
```
Episode 100: reward=45.2
Total timesteps: 100000
```

##### Metric Lines

When `CONTROLLER_METRICS` is set:
```
@METRIC {"kind":"iteration","training_iteration":10,"timesteps_total":100000,...}
```

##### Standard Error

Error messages and warnings:
```
WARNING: Environment not responding
ERROR: Failed to connect to Godot
```

#### Example Usage

**Basic Training**:
```bash
python stable_baselines3_training_script.py \
  --env_path ./MyGame.x86_64 \
  --timesteps 1000000 \
  --experiment_name my_first_training
```

**Fast Training with Parallel Envs**:
```bash
python stable_baselines3_training_script.py \
  --env_path ./MyGame.x86_64 \
  --timesteps 5000000 \
  --speedup 30 \
  --n_parallel 8 \
  --experiment_name fast_train
```

**Training with Checkpoints**:
```bash
python stable_baselines3_training_script.py \
  --env_path ./MyGame.x86_64 \
  --timesteps 10000000 \
  --save_checkpoint_frequency 100000 \
  --experiment_dir ./checkpoints \
  --experiment_name long_train
```

**Resume Training**:
```bash
python stable_baselines3_training_script.py \
  --env_path ./MyGame.x86_64 \
  --resume_model_path ./checkpoints/rl_model_1000000_steps.zip \
  --timesteps 1000000 \
  --experiment_name continue_train
```

---

### `rllib_training_script.py`

Multi-agent reinforcement learning using Ray RLlib.

#### Command Line Interface

```bash
python rllib_training_script.py [OPTIONS]
```

#### Arguments

**`--config_file PATH`**
- **Type**: String
- **Default**: `rllib_config.yaml`
- **Description**: Path to RLlib YAML configuration file (controller projects pass `.rlcontroller/rllib_config.yaml`)
- **Example**: `--config_file ./my_config.yaml`

**`--experiment_dir PATH`**
- **Type**: String
- **Default**: `logs/rllib`
- **Description**: Directory for Ray results and checkpoints
- **Example**: `--experiment_dir ./rllib_results`

#### Environment Variables

**`CONTROLLER_METRICS`**
- **Purpose**: Enable structured metric emission
- **Format**: Any truthy string
- **Effect**: Emits `@METRIC` lines for each iteration
- **Set by**: Controller automatically

#### RLlib Configuration File

The script reads configuration from a YAML file (default: `rllib_config.yaml`; controller-managed projects keep it under `.rlcontroller/rllib_config.yaml`).

**Required Structure**:
```yaml
env: godot

env_config:
  env_path: "/path/to/game.x86_64"
  show_window: true
  seed: 0
  speedup: 1

multiagent:
  policies:
    policy_1:
      # Configuration
    policy_2:
      # Configuration
  
  policy_mapping_fn: |
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "policy_1"  # Map agent to policy
  
  policies_to_train:
    - policy_1
    - policy_2

# Algorithm configuration
train_batch_size: 4000
sgd_minibatch_size: 128
num_sgd_iter: 30
lr: 0.0003
gamma: 0.99
lambda: 0.95
# ... more settings
```

**Note**: The controller's `training_config.json` settings override many of these YAML settings at runtime.

#### Output Format

##### Standard Output

Training progress:
```
== Status ==
...
Current time: 2025-11-05 14:30:00
Training iteration: 10
...
```

##### Metric Lines

```
@METRIC {"kind":"iteration","training_iteration":10,"policies":{"policy_1":{...}},...}
```

##### Standard Error

Ray warnings and errors:
```
WARNING: Worker crashed, restarting...
```

#### Callbacks

The script includes `ControllerMetricsCallback` which:
- Captures iteration results
- Extracts metrics for all policies
- Formats as JSON with `@METRIC` prefix
- Emits to stdout

#### Example Usage

**Basic Multi-Agent Training**:
```bash
python rllib_training_script.py \
  --config_file ./pong_config.yaml \
  --experiment_dir ./results
```

**Custom Configuration**:
```bash
python rllib_training_script.py \
  --config_file ./custom_config.yaml \
  --experiment_dir ./my_experiments
```

---

## Export Scripts

### `convert_sb3_to_onnx.py`

Convert Stable Baselines3 models to ONNX format.

#### Command Line Interface

```bash
python convert_sb3_to_onnx.py MODEL_PATH [OPTIONS]
```

#### Arguments

**`MODEL_PATH`** (positional)
- **Type**: String
- **Required**: Yes
- **Description**: Path to `.zip` model file
- **Example**: `./model.zip`

**`-o, --output PATH`**
- **Type**: String
- **Required**: Yes (unless using defaults)
- **Description**: Output path for `.onnx` file
- **Example**: `-o ./exported_model.onnx`

**`--algo NAME`**
- **Type**: String
- **Default**: Auto-detect
- **Options**: `ppo`, `a2c`, `dqn`, `sac`, `td3`, `ddpg`
- **Description**: Algorithm type (auto-detected if omitted)
- **Example**: `--algo ppo`

**`--opset VERSION`**
- **Type**: Integer
- **Default**: 13
- **Range**: 9-17
- **Description**: ONNX opset version
- **Example**: `--opset 15`

**`--ir-version VERSION`**
- **Type**: Integer
- **Default**: 9
- **Description**: ONNX IR version
- **Example**: `--ir-version 9`

**`--no-verify`**
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Skip ONNX model verification
- **Example**: `--no-verify`

**`--use-obs-array`**
- **Type**: Flag (boolean)
- **Default**: False
- **Description**: Use array format for observations
- **Example**: `--use-obs-array`

#### Algorithm Auto-Detection

The script attempts to detect the algorithm from the model file:
1. Check for `algo` field in metadata
2. Analyze model structure (replay buffer, target networks, etc.)
3. Try loading with candidate algorithms

If auto-detection fails, manually specify with `--algo`.

#### Example Usage

**Basic Export**:
```bash
python convert_sb3_to_onnx.py model.zip -o model.onnx
```

**Specify Algorithm**:
```bash
python convert_sb3_to_onnx.py model.zip -o model.onnx --algo ppo
```

**Custom ONNX Version**:
```bash
python convert_sb3_to_onnx.py model.zip -o model.onnx --opset 15
```

---

### `convert_rllib_to_onnx.py`

Convert RLlib checkpoints to ONNX format.

#### Command Line Interface

```bash
python convert_rllib_to_onnx.py CHECKPOINT_PATH [OPTIONS]
```

#### Arguments

**`CHECKPOINT_PATH`** (positional)
- **Type**: String
- **Required**: Yes
- **Description**: Path to checkpoint directory or training run
- **Example**: `./logs/rllib/PPO_2025-11-05/PPO_godot_abc123/checkpoint_000010`

**`-o, --output-dir PATH`**
- **Type**: String
- **Default**: `./onnx_exports`
- **Description**: Output directory for ONNX files
- **Example**: `-o ./my_models`

**`--policy POLICY_ID`**
- **Type**: String
- **Default**: Export all policies
- **Description**: Export only specific policy
- **Example**: `--policy left_policy`

**`--checkpoint-number N`**
- **Type**: Integer
- **Default**: Latest checkpoint
- **Description**: Specific checkpoint number to export
- **Example**: `--checkpoint-number 10`

**`--opset VERSION`**
- **Type**: Integer
- **Default**: 13
- **Range**: 9-17
- **Description**: ONNX opset version
- **Example**: `--opset 15`

**`--ir-version VERSION`**
- **Type**: Integer
- **Default**: 9
- **Description**: ONNX IR version
- **Example**: `--ir-version 9`

#### Checkpoint Auto-Discovery

If you provide a training run directory instead of a specific checkpoint:
```
logs/rllib/PPO_2025-11-05/PPO_godot_abc123/
├── checkpoint_000001/
├── checkpoint_000005/
└── checkpoint_000010/  # Latest, will be used
```

The script automatically finds the latest checkpoint.

#### Multi-Agent Export

For multi-agent training, the script:
1. Identifies all policies in the checkpoint
2. Exports each policy as a separate `.onnx` file
3. Names files as `<policy_id>.onnx`

**Output Example**:
```
onnx_exports/
├── left_policy.onnx
└── right_policy.onnx
```

#### Example Usage

**Export Latest Checkpoint (All Policies)**:
```bash
python convert_rllib_to_onnx.py \
  ./logs/rllib/PPO_2025-11-05/PPO_godot_abc123/
```

**Export Specific Checkpoint**:
```bash
python convert_rllib_to_onnx.py \
  ./logs/rllib/PPO_2025-11-05/PPO_godot_abc123/ \
  --checkpoint-number 10
```

**Export Single Policy**:
```bash
python convert_rllib_to_onnx.py \
  ./logs/rllib/PPO_2025-11-05/PPO_godot_abc123/checkpoint_000010 \
  --policy left_policy \
  -o ./models
```

**Custom Output Directory**:
```bash
python convert_rllib_to_onnx.py \
  ./logs/rllib/PPO_2025-11-05/PPO_godot_abc123/ \
  -o /path/to/godot/project/models
```

---

## Utility Scripts

### `check_py_env.py`

Check Python environment for required packages.

#### Usage

```bash
python check_py_env.py
```

#### Output

JSON format with package availability:
```json
{
  "sb3_available": true,
  "rllib_available": true,
  "godot_rl_available": true,
  "torch_available": true,
  "onnx_available": true
}
```

#### Exit Codes

- `0` - All packages available
- `1` - Some packages missing

**Note**: The controller uses this script to check environment status (press `p` in Home tab).

---

### `demo.py`

Test script for output streaming.

#### Usage

```bash
python demo.py
```

#### Behavior

- Prints test messages to stdout
- Emits sample metrics
- Simulates training output
- Used to verify controller's output streaming

**Note**: Accessible in controller with `d` key in Train tab.

---

## Metric Format Specification

### Metric Line Format

```
@METRIC <JSON_OBJECT>
```

### JSON Schema

```json
{
  "kind": "iteration",
  "timestamp": "2025-11-05T14:30:00Z",
  "training_iteration": 10,
  "timesteps_total": 100000,
  "episodes_total": 500,
  "episodes_this_iter": 50,
  "episode_reward_mean": 45.2,
  "episode_reward_min": 10.0,
  "episode_reward_max": 98.5,
  "episode_len_mean": 200.0,
  "time_this_iter_s": 30.5,
  "time_total_s": 305.0,
  "env_steps_this_iter": 10000,
  "env_throughput": 328.0,
  "num_env_steps_sampled": 100000,
  "num_env_steps_trained": 100000,
  "num_agent_steps_sampled": 100000,
  "num_agent_steps_trained": 100000,
  "custom_metrics": {
    "goals_scored": 123.0,
    "shots_taken": 456.0
  },
  "policies": {
    "policy_1": {
      "reward_mean": 45.2,
      "reward_min": 10.0,
      "reward_max": 98.5,
      "episode_len_mean": 200.0,
      "completed_episodes": 50,
      "learner_stats": {
        "policy_loss": -0.0123,
        "vf_loss": 0.456,
        "entropy": 1.234
      },
      "custom_metrics": {
        "hits": 10.0
      }
    }
  },
  "checkpoints": 1
}
```

### Field Descriptions

**Top-Level Fields**:
- `kind`: Always `"iteration"` for iteration metrics
- `timestamp`: ISO 8601 timestamp
- `training_iteration`: Current iteration number
- `timesteps_total`: Total environment steps
- `episodes_total`: Total episodes completed
- `episode_reward_mean/min/max`: Episode reward statistics
- `episode_len_mean`: Average episode length

**Policy-Specific Fields**:
- `policies`: Map of policy_id to policy metrics
- Each policy contains:
  - Reward statistics
  - Learner statistics (loss, entropy, etc.)
  - Custom metrics from environment

**Custom Metrics**:
- Any additional metrics from your Godot environment
- Automatically captured and displayed

---

## Integration with Godot RL Agents

These scripts are designed to work with the [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) framework.

### Requirements

1. **Godot Project Setup**:
   - Install Godot RL Agents plugin
   - Configure observation and action spaces
   - Implement environment logic
   - Export as executable

2. **Environment Interface**:
   - Observations: Dict or Box space
   - Actions: Discrete or Continuous
   - Rewards: Float per step
   - Done signals: Episode termination

3. **Communication**:
   - Environment runs as separate process
   - Communicates via pipes
   - Supports multiple instances

### Custom Metrics from Godot

To emit custom metrics from your Godot environment:

```gdscript
# In your Godot script
func get_info() -> Dictionary:
    return {
        "goals_scored": goals,
        "shots_taken": shots,
        "distance_traveled": distance
    }
```

These metrics automatically appear in:
- Training logs
- Controller metrics tab
- Exported data

For more details on setting up your Godot environment, see the [Godot RL Agents documentation](https://github.com/edbeeching/godot_rl_agents).

---

## Troubleshooting

### Script Not Found

**Error**: `FileNotFoundError: training script not found`

**Solution**:
- Ensure scripts are in project directory or controller root
- Check file names match exactly
- Verify file permissions (executable not required)

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'stable_baselines3'`

**Solution**:
```bash
source .venv/bin/activate
pip install -r requrements.txt
```

### ONNX Export Fails

**Error**: `RuntimeError: ONNX export failed`

**Solutions**:
1. Update PyTorch: `pip install --upgrade torch`
2. Update ONNX: `pip install --upgrade onnx`
3. Try different opset: `--opset 11`
4. Check model compatibility

### RLlib Checkpoint Not Found

**Error**: `ValueError: No checkpoints found`

**Solutions**:
1. Verify checkpoint path is correct
2. Check training completed successfully
3. Look for `checkpoint_*` directories
4. Use absolute paths

---

## Script Modifications

The training scripts in this controller are modified versions of the original Godot RL Agents scripts. Key changes:

1. **Metric Emission**: Added `@METRIC` prefix for structured output
2. **Callback System**: Enhanced callbacks for real-time metrics
3. **Controller Integration**: Environment variable support
4. **Error Handling**: Improved error messages for controller parsing

For the original scripts, see [Godot RL Agents repository](https://github.com/edbeeching/godot_rl_agents).

---

**Last Updated**: November 5, 2025
