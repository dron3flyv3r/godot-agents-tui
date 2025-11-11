# Godot RL Training Controller

A powerful TUI (Terminal User Interface) application for managing reinforcement learning training workflows with Godot game environments. This controller is specifically designed to work with [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents) and streamlines the process of training, monitoring, and exporting RL agents using both Stable Baselines3 (single-agent) and RLlib (multi-agent) frameworks.

> **Note**: This controller uses the Godot RL Agents framework with some small script modifications for enhanced metrics and integration. For detailed information on setting up your Godot environment and creating RL-ready games, please refer to the [official Godot RL Agents documentation](https://github.com/edbeeching/godot_rl_agents).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)

## Features

- 🎮 **Project Management**: Organize multiple Godot RL projects with persistent configuration
- 🤖 **Dual Training Modes**: 
  - Single-agent training with Stable Baselines3
  - Multi-agent training with RLlib
- 📊 **Real-time Metrics**: Live visualization of training progress with charts and statistics
- 🔄 **ONNX Export**: Convert trained models to ONNX format for Godot integration
- 💻 **Interactive TUI**: Intuitive keyboard-driven interface built with Ratatui
- 🐍 **Python Integration**: Seamless integration with Python training scripts
- 📈 **Advanced Configuration**: Extensive hyperparameter tuning options for both frameworks

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Home Tab](#home-tab)
  - [Train Tab](#train-tab)
  - [Metrics Tab](#metrics-tab)
  - [Export Tab](#export-tab)
- [Configuration](#configuration)
- [Training Scripts](#training-scripts)
- [ONNX Export](#onnx-export)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Installation

### Prerequisites

1. **Godot RL Agents Setup**
   
   First, set up your Godot project with RL capabilities. Follow the [Godot RL Agents documentation](https://github.com/edbeeching/godot_rl_agents) to:
   - Install the Godot RL Agents plugin in your Godot project
   - Set up your environment with observation spaces and action spaces
   - Export your game as an executable for training
   
   For training details, see the [Godot RL Agents training guide](https://github.com/edbeeching/godot_rl_agents#training).

2. **Rust** (1.70 or higher)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Python** (3.8 or higher) with a virtual environment

4. **Required Python packages**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requrements.txt
   ```

   The requirements include:
   - `stable-baselines3` - For single-agent training
   - `ray[rllib]` - For multi-agent training
   - `godot-rl` - Godot RL integration
   - `torch` - PyTorch for neural networks
   - `onnx` - ONNX model export

### Building the Controller

```bash
# Debug build
cargo build

# Release build (recommended for performance)
cargo build --release

# Run directly
cargo run --release
```

The compiled binary will be in `target/release/controller-mk2`.

To bake a specific Python interpreter/virtualenv path into the binary, set `CONTROLLER_PYTHON_BIN` before building (and optionally `CONTROLLER_SCRIPTS_ROOT` so the controller can locate the bundled Python scripts even when run from other directories):

```bash
CONTROLLER_PYTHON_BIN="$PWD/.venv/bin/python" \
CONTROLLER_SCRIPTS_ROOT="$PWD" \
cargo build --release
```

At runtime you can still override these by exporting `CONTROLLER_PYTHON_BIN` or `CONTROLLER_SCRIPTS_ROOT`, but if they are unset the embedded values will be used automatically. The provided `scripts/install.sh` looks for `.venv/bin/python` in the repository and embeds it when present, and it always embeds the repository root as the script directory so the installed binary can find `check_py_env.py`, training scripts, etc., even when launched from elsewhere.

To use the controller from any project directory, either install it with Cargo:

```bash
cargo install --path .
```

which places the binary in `~/.cargo/bin`, or copy/symlink the release build into a directory that is already on your `PATH`, for example:

```bash
install -Dm755 target/release/controller-mk2 ~/.local/bin/controller-mk2
```

## Quick Start

1. **Activate Python environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Launch the controller**:
   ```bash
   ./target/release/controller-mk2
   ```

3. **Create a new project** (Home tab):
   - Press `n` to create a new project
   - Enter a project name and press Enter
   - Choose the project directory (defaults to a unique folder under `$PROJECTS_ROOT`); the controller will create/manage a `.rlcontroller/` config folder plus a `logs/` subfolder inside it and run all scripts relative to that project directory.

4. **Configure training** (Train tab):
   - Press `2` or navigate to the Train tab
   - Set the environment path to your Godot binary
   - Adjust training parameters as needed
   - Press `t` to start training

5. **Monitor progress** (Metrics tab):
- Press `3` to view real-time training metrics
- Watch reward curves and other statistics
- Press `c` in the Metrics tab to load saved run overlays and compare multiple runs on the same chart

6. **Export models** (Export tab):
   - Press `4` to access export options
   - Configure export settings
   - Press `e` to export trained models to ONNX

## Usage

### Navigation

- **Arrow Keys** or **Tab**: Navigate between UI elements
- **Number Keys (1-4)**: Jump directly to tabs
- **Enter**: Select/Confirm
- **Esc** or **q**: Quit (with confirmation)
- **?**: Show help overlay

### Home Tab

The Home tab is your project management hub.

**Key Bindings:**
- `n` - Create a new project
- `Enter` - Select/activate a project
- `p` - Re-check Python environment status

When you press `n`, the controller first prompts for a project name and then for the project directory. The default is a unique folder under `$PROJECTS_ROOT`, but you can point it at any writable location (e.g., your game repo). The controller manages a `.rlcontroller/` hidden folder for its JSON configs plus a `logs/` subfolder inside that directory and runs all scripts relative to it.

**Features:**
- View and manage all projects
- See active project details
- Monitor Python environment status (SB3 and RLlib availability)
- Every project remembers its name, the log directory you chose, and the parent directory that becomes the working directory whenever the project is active

### Train Tab

Configure and execute training runs.

**Key Bindings:**
- `t` - Start training
- `c` - Cancel running training (SIGINT)
- `m` - Toggle between Single-Agent (SB3) and Multi-Agent (RLlib) modes
- `s` - Save configuration
- `o` - Open advanced configuration
- `d` - Run demo (test output streaming)

**Configuration Fields:**
- **Environment Path**: Path to your Godot game binary (absolute or relative to the project directory, required)
- **Timesteps**: Number of training steps (default: 1,000,000)
- **Experiment Name**: Name for this training run

**Training Modes:**

1. **Single-Agent (Stable Baselines3)**:
   - Best for: Simple environments, single agent scenarios
   - Algorithm: PPO (Proximal Policy Optimization)
   - Output: Model saved to `logs/sb3/`

2. **Multi-Agent (RLlib)**:
   - Best for: Complex environments, multiple agents
   - Algorithm: PPO with multi-agent support
   - Output: Checkpoints saved to `logs/rllib/`
   - Configuration: Uses `rllib_config.yaml` in project directory

**Advanced Configuration:**
Press `o` to access 50+ hyperparameters for fine-tuning:
- Network architecture (hidden layers, activation functions)
- Learning rates and schedules
- Batch sizes and rollout settings
- Exploration parameters
- Checkpointing options

### Metrics Tab

Real-time visualization of training progress.

**Key Bindings:**
- `Up/Down` - Scroll through metric history
- `Left/Right` - Switch between metric categories
- `p` - Cycle through policies (multi-agent)
- `f` - Toggle focus mode (custom metrics, learner stats, etc.)
- `c` - Load a saved run overlay for comparison (pick from `.rlcontroller/runs/`)
- `C` - Clear all overlays

**Displayed Metrics:**
- Episode rewards (mean, min, max)
- Episode lengths
- Training iterations and timesteps
- Environment throughput
- Policy-specific statistics
- Custom metrics from your environment
- Learner statistics (loss, entropy, KL divergence)

**Charts:**
- Reward progression over time
- Episode length trends
- Up to 2000 samples retained in history
- Every time a training run finishes, the controller stores its metrics as a JSON file under `.rlcontroller/runs/`. Use `c` to pick any saved run and overlay it on the current chart, then press `C` to clear overlays when you're done comparing.

### Export Tab

Convert trained models to ONNX format for Godot integration.

**Key Bindings:**
- `e` - Execute export
- `m` - Toggle between SB3 and RLlib export modes
- `b` - Browse for checkpoint files
- `s` - Save export configuration

**Export Modes:**

1. **Stable Baselines3 Export**:
   - Select model `.zip` file
   - Specify output path
   - Choose algorithm (PPO, A2C, etc.)
   - Configure ONNX opset and IR version

2. **RLlib Export**:
   - Select checkpoint directory
   - Choose specific checkpoint number (or latest)
   - Export all policies or specific policy
   - Supports multi-agent configurations

**Output:**
- ONNX files compatible with Godot RL Agents
- Files can be loaded directly in Godot using the `ONNXModel` resource

## Configuration

### Project Configuration Files

Each project directory (located under `$PROJECTS_ROOT` by default) contains:

1. **`project.json`**: Basic project metadata
   ```json
   {
     "created": 1761917546,
     "name": "my_project"
   }
   ```

2. **`training_config.json`**: Training parameters
   - Environment paths
   - Hyperparameters for both SB3 and RLlib
   - Experiment settings

3. **`rllib_config.yaml`**: RLlib-specific configuration
   - Environment factory
   - Multi-agent policies
   - Algorithm configuration

4. **`export_config.json`**: ONNX export settings
   - Checkpoint paths
   - Output directories
   - Export options

### Global Configuration

- **`$PROJECTS_ROOT/index.json`**: List of all projects with last used timestamps
- Projects are automatically indexed when created or accessed

## Training Scripts

The controller uses Python scripts for actual training:

### `stable_baselines3_training_script.py`

Single-agent training with Stable Baselines3.

**Key Parameters:**
- `--env_path`: Godot environment binary
- `--timesteps`: Training duration
- `--experiment_dir`: Output directory (default: `logs/sb3`)
- `--experiment_name`: Run identifier
- `--speedup`: Physics speed multiplier
- `--n_parallel`: Parallel environments
- `--viz`: Show visualization window

**Example:**
```bash
python stable_baselines3_training_script.py \
  --env_path ./MyGame.x86_64 \
  --timesteps 1000000 \
  --experiment_name my_experiment \
  --speedup 10 \
  --n_parallel 4
```

### `rllib_training_script.py`

Multi-agent training with RLlib.

**Key Parameters:**
- `--config_file`: RLlib YAML configuration (default: `rllib_config.yaml`)
- `--experiment_dir`: Output directory (default: `logs/rllib`)

**Example:**
```bash
python rllib_training_script.py \
  --config_file ./my_config.yaml \
  --experiment_dir ./my_logs
```

### Metrics Emission

Both scripts support the `CONTROLLER_METRICS` environment variable:
- When set, scripts emit structured metrics prefixed with `@METRIC`
- The controller automatically enables this and parses metrics for visualization
- Metrics include rewards, episode lengths, training statistics, and custom metrics

## ONNX Export

### Using the Export Scripts

#### Stable Baselines3 Export

```bash
python convert_sb3_to_onnx.py model.zip \
  -o output_model.onnx \
  --algo ppo \
  --opset 13
```

**Options:**
- `-o, --output`: Output ONNX file path
- `--algo`: Algorithm (ppo, a2c, dqn, sac, td3, ddpg)
- `--opset`: ONNX opset version (default: 13)
- `--no-verify`: Skip ONNX verification
- `--use-obs-array`: Use observation array input

#### RLlib Export

```bash
python convert_rllib_to_onnx.py checkpoint_000008 \
  -o models/ \
  --policy left_policy \
  --opset 13
```

**Options:**
- `-o, --output-dir`: Output directory for ONNX files
- `--policy`: Export specific policy (default: all policies)
- `--checkpoint-number`: Specific checkpoint number
- `--opset`: ONNX opset version (default: 13)

**Auto-discovery:**
- If you provide a training directory, the script finds the latest checkpoint
- Supports exporting multiple policies from multi-agent training

### Using Exported Models in Godot

1. Copy `.onnx` files to your Godot project
2. Load with `ONNXModel` resource:
   ```gdscript
   var model = ONNXModel.new()
   model.load_model("res://models/policy.onnx")
   
   var obs = get_observation()
    var action = model.run_inference([obs])
    ```

## Project Storage

- By default the controller stores its index (not the logs themselves) under:
  1. The directory specified by `CONTROLLER_PROJECTS_ROOT` (if set)
  2. `$XDG_DATA_HOME/godot_rl_controller/projects`
  3. `$HOME/.local/share/godot_rl_controller/projects`
- A single `index.json` inside that directory tracks every project (name + log path), so you can launch the controller from any working directory and still see the same list.
- When you create a project you are prompted for the project directory. Press Enter to accept the default (a slugged folder under `$PROJECTS_ROOT`) or point at any writable location. The controller creates/uses a `.rlcontroller/` folder (for `project.json`, `training_config.json`, `export_config.json`, etc.) plus a `logs/` subdirectory within it for SB3/RLlib runs, and treats that project directory as the working directory for scripts and pickers.
- Because each project points at explicit paths, you can move or back up individual log folders without touching the controller binary—just recreate the project entry if you relocate the logs.
- Export `CONTROLLER_PROJECTS_ROOT=/path/to/dir` before launching if you want to move the entire project library somewhere else (e.g., another drive).

## Project Structure

```
$PROJECTS_ROOT/
└── index.json            # Project index (names + project directories)

<project-dir>/            # Directory you selected/created for the project
├── .rlcontroller/
│   ├── project.json
│   ├── training_config.json
│   ├── export_config.json
│   └── runs/
│       └── <timestamp>_<experiment>.json
├── logs/
│   ├── sb3/…             # SB3 runs
│   └── rllib/…           # RLlib runs
└── rllib_config.yaml

agents/                   # Controller workspace (this repository)
├── src/
│   ├── main.rs           # Application entry point
│   ├── app/…             # Core application logic
│   └── ui/…              # Rendering and widgets
├── logs/                 # Global logs
├── stable_baselines3_training_script.py
├── rllib_training_script.py
├── convert_sb3_to_onnx.py
├── convert_rllib_to_onnx.py
├── check_py_env.py       # Python environment checker
├── demo.py               # Demo script for testing
├── Cargo.toml            # Rust dependencies
└── requrements.txt       # Python dependencies
```

## Documentation

Comprehensive documentation is available to help you at every step:

### 🚀 Getting Started
- **[Quick Start Guide](docs/QUICK_START.md)** - Get up and running in minutes
- **[Training Guide](TRAINING_GUIDE.md)** - Detailed training workflows

### 📚 Reference
- **[Configuration Guide](docs/CONFIGURATION.md)** - All configuration options explained
- **[API Reference](docs/API.md)** - Python script interfaces and parameters
- **[Architecture Documentation](docs/ARCHITECTURE.md)** - Technical design and implementation

### 🔧 Help
- **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Solutions to common problems
- **[Documentation Index](docs/DOCS.md)** - Complete documentation overview

**New to the project?** Start with the [Quick Start Guide](docs/QUICK_START.md)!

## Troubleshooting

### Python Environment Issues

**Problem**: Python packages not found

**Solution**:
```bash
source .venv/bin/activate
pip install -r requrements.txt
```

Press `p` in the Home tab to check environment status.

### Training Doesn't Start

**Common causes:**
1. Environment path not set or incorrect
2. Python environment not activated
3. Training script not found

**Check:**
- Verify environment path points to executable
- Ensure scripts are in project or root directory
- Check terminal output for error messages

### No Metrics Displayed

**Problem**: Metrics tab is empty during training

**Solution**:
- Metrics are only available when training is running
- Check that `CONTROLLER_METRICS` is set (controller sets this automatically)
- Verify training script supports metric emission

### ONNX Export Fails

**Common issues:**
1. Checkpoint path incorrect
2. Model architecture mismatch
3. ONNX dependencies missing

**Solution**:
```bash
pip install onnx torch
```
Use file browser (`b` key) to select correct checkpoint.

### Performance Issues

**For faster training:**
1. Use release build: `cargo build --release`
2. Increase physics speedup (SB3)
3. Use multiple parallel environments
4. Enable headless mode (disable visualization)

## Contributing

Contributions are welcome! Areas for improvement:

- Additional RL frameworks (TorchRL, CleanRL)
- More export formats
- Enhanced visualization options
- Better error handling and validation
- Additional algorithms and presets

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Ratatui](https://github.com/ratatui-org/ratatui) for the TUI
- Integrates with [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents)
- Uses [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- Uses [Ray RLlib](https://docs.ray.io/en/latest/rllib/index.html)

## Support

For questions and issues:
- Check the documentation in `docs/`
- Review the training guide: `TRAINING_GUIDE.md`
- Examine example projects inside `$PROJECTS_ROOT`

---

**Happy Training! 🎮🤖**

> [!IMPORTANT] AI generated docs
> This documentation was generated using Claude Sonnet 4.5, I can't personally check the accuracy of all details. Please keep this in mind when using the information provided here. I will try to review and correct any mistakes over time. If you find any errors, please open an issue on GitHub.
