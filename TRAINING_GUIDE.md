# Training Guide

This guide explains how to use the training functionality in the controller.

## Overview

The controller supports two training modes:
1. **Single-Agent (Stable Baselines 3)** - For single-agent reinforcement learning
2. **Multi-Agent (RLlib)** - For multi-agent reinforcement learning scenarios

## Getting Started

### Prerequisites

1. **Python environment** with the required packages:
   - For Single-Agent: `stable-baselines3`, `godot-rl`
   - For Multi-Agent: `ray[rllib]`, `godot-rl`

2. **Training scripts** must be present:
   - `stable_baselines3_training_script.py` (for SB3)
   - `rllib_training_script.py` (for RLlib)

3. **Godot environment binary** - The executable for your training environment

### Quick Start

1. **Navigate to Train tab**: Press `2` or use arrow keys to go to the "Train" tab

2. **Select a project**: First, select a project from the Home tab (press `1` and select)

3. **Toggle training mode**: Press `m` to switch between Single-Agent and Multi-Agent modes

4. **Configure training**: 
   - Environment path (required)
   - Timesteps (default: 1,000,000)
   - Experiment name (default: "training")

5. **Start training**: Press `t` to start training

6. **Monitor output**: Watch the training output stream in real-time

7. **Cancel if needed**: Press `c` to send a keyboard interrupt (SIGINT) to the training process

## Training Modes

### Single-Agent (Stable Baselines 3)

Uses the `stable_baselines3_training_script.py` script with these parameters:
- `--env_path`: Path to the Godot environment binary
- `--experiment_dir`: Directory for logs (defaults to `logs/sb3`)
- `--experiment_name`: Name for this training run
- `--timesteps`: Number of environment steps to train
- `--speedup`: Physics speedup multiplier (default: 1)
- `--n_parallel`: Number of parallel environment instances (default: 1)
- `--viz`: Show visualization window (optional)

**Configuration options:**
- `sb3_viz`: Enable/disable visualization
- `sb3_speedup`: Speed multiplier for physics
- `sb3_n_parallel`: Number of parallel environments

### Multi-Agent (RLlib)

Uses the `rllib_training_script.py` script with these parameters:
- `--config_file`: Path to RLlib config YAML (default: `rllib_config.yaml`)
- `--experiment_dir`: Directory for logs (defaults to `logs/rllib`)

**Configuration options:**
- `rllib_config_file`: Path to the RLlib YAML configuration

## Keyboard Controls (Train Tab)

- `t` or `T` - Start training with current configuration
- `d` or `D` - Run demo training (uses demo.py for testing)
- `m` or `M` - Toggle between Single-Agent and Multi-Agent modes
- `c` or `C` - Cancel running training (sends SIGINT)
- `Left/Right` - Switch tabs
- `q` or `Esc` - Quit application

## Output Streaming

The training output streams in real-time, emulating a read-only terminal. You'll see:
- Command being executed
- Real-time output from the training script
- Metrics (if enabled with `CONTROLLER_METRICS` environment variable)
- Error messages (prefixed with `!`)
- Exit status when training completes

## Metrics Integration

Both training scripts support emitting structured metrics when the `CONTROLLER_METRICS` environment variable is set. The controller automatically enables this, and metrics will appear in the output prefixed with `@METRIC`.

## Tips

1. **Testing setup**: Use the demo training (`d` key) to test that output streaming works

2. **Script location**: Training scripts are searched in this order:
   - Project directory
   - Controller root directory
   - Current working directory

3. **Graceful cancellation**: Pressing `c` sends SIGINT (like Ctrl+C), allowing the script to clean up:
   - SB3: Saves model and exports ONNX if configured
   - RLlib: Attempts graceful Ray shutdown and locates latest checkpoint

4. **Environment path**: Make sure your Godot binary has execute permissions:
   ```bash
   chmod +x path/to/your/environment.x86_64
   ```

5. **Configuration files**: For RLlib training, ensure `rllib_config.yaml` is present in your project

## Troubleshooting

### "Environment path is required"
Configure the environment path in your training settings before starting.

### "No project selected"
Go to the Home tab (press `1`) and select/activate a project first.

### "Training script not found"
Ensure the training scripts are accessible from your project or controller directory.

### Output appears all at once
This shouldn't happen anymore - the `-u` flag forces Python unbuffered output. If it does, check your Python version and environment.

### Training won't cancel
The controller sends SIGINT, but the script needs to handle it. Both provided scripts handle this correctly.

## Example Workflow

```
1. Press `1` - Go to Home tab
2. Select project with arrow keys
3. Press Enter - Activate project
4. Press `2` - Go to Train tab
5. Press `m` - Toggle to desired mode (if needed)
6. [Configure settings - feature to be added]
7. Press `t` - Start training
8. Watch output stream in real-time
9. Press `c` if you need to stop early
```

## Future Enhancements

The following features are planned:
- Interactive configuration editor
- Saved training presets
- Training history and results viewer
- Checkpoint management
- Tensorboard integration
- Metrics visualization
