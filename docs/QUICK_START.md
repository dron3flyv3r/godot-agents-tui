# Quick Start Guide

Get up and running with the Godot RL Training Controller in minutes.

## Prerequisites

Before starting, ensure you have:

1. ✅ **Rust** installed (1.70 or higher)
2. ✅ **Python** 3.8+ with pip
3. ✅ **A Godot game** set up with [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents)
4. ✅ **Exported Godot binary** of your game

## Installation Steps

### 1. Clone or Download the Controller

```bash
cd ~/GameProjects
git clone <repository-url> agents
cd agents
```

### 2. Set Up Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# Or on Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requrements.txt
```

This installs:
- `stable-baselines3` for single-agent training
- `ray[rllib]` for multi-agent training
- `godot-rl` for Godot integration
- `torch`, `onnx` for model export

### 3. Build the Controller

```bash
# Debug build (for development)
cargo build

# OR release build (recommended for training)
cargo build --release
```

The binary will be at:
- Debug: `target/debug/controller-mk2`
- Release: `target/release/controller-mk2`

### 4. Verify Installation

```bash
# Check Python packages
python check_py_env.py

# Should output:
# {
#   "sb3_available": true,
#   "rllib_available": true,
#   ...
# }
```

## Your First Training Session

### Step 1: Launch the Controller

```bash
# Make sure Python env is activated
source .venv/bin/activate

# Run the controller
./target/release/controller-mk2
```

You'll see the TUI interface with tabs at the top.

### Step 2: Create a Project

1. **Press `1`** or navigate to the **Home** tab
2. **Press `n`** to create a new project
3. **Type a project name** (e.g., "my_first_agent")
4. **Press Enter** to confirm

Your project is now created in `projects/my_first_agent/`

### Step 3: Configure Training

1. **Press `2`** to go to the **Train** tab
2. **Navigate to "Environment Path"** (use arrow keys or Tab)
3. **Press Enter** to edit
4. **Type the path** to your Godot game binary
   - Example: `/home/user/MyGame/exports/MyGame.x86_64`
5. **Press Enter** to save
6. **Press `s`** to save configuration

Optional: Adjust other settings like:
- **Timesteps**: How long to train (default: 1,000,000)
- **Experiment Name**: Name for this run

### Step 4: Start Training

1. **Press `t`** to start training
2. **Watch the output** stream in real-time
3. Training begins! You'll see:
   - Command being executed
   - Training logs
   - Progress updates

### Step 5: Monitor Progress

1. **Press `3`** to go to the **Metrics** tab
2. **View real-time charts** of:
   - Episode rewards (mean, min, max)
   - Episode lengths
   - Training statistics
3. Charts update automatically as training progresses

### Step 6: Wait for Completion

Training will run until:
- Timesteps reached
- You press `c` to cancel
- An error occurs

When done, you'll see "Process exited with code 0" (success).

### Step 7: Export to ONNX

1. **Press `4`** to go to the **Export** tab
2. **Navigate through export options**:
   - For SB3: Set model path (from `logs/sb3/`)
   - For RLlib: Set checkpoint path (from `logs/rllib/`)
3. **Set output path** where you want the `.onnx` file
4. **Press `e`** to start export
5. **Wait for completion**

Your ONNX model is now ready to use in Godot!

## Using the Model in Godot

### Load the ONNX Model

put the model in your Godot project, e.g., `res://models/my_agent.onnx`
in your Godot AIController node, set the model path and choose ONNXInference

### Use in Your Game

```gdscript
extends CharacterBody2D

var ai_controller = $AIController2D # or AIController3D

func _physics_process(delta):
    # Get observation from your game state
    var obs = get_observation()

    Velocity = ai_controller.move
    move_and_slide()
```

## Common First-Time Issues

### 1. "Python packages not found"

**Solution**:
```bash
source .venv/bin/activate
pip install -r requrements.txt
```

Press `p` in Home tab to verify.

### 2. "Training doesn't start"

**Check**:
- ✅ Environment path is correct
- ✅ Binary is executable: `chmod +x /path/to/game.x86_64`
- ✅ Python environment activated
- ✅ Configuration saved (press `s`)

### 3. "Controller shows no metrics"

**Explanation**: Metrics only appear while training is running.

**Check**:
- Training is active
- Scroll down in output to see logs

### 4. "Can't find checkpoint for export"

**Solution**:
- Use the file browser: Press `b` in Export tab
- Navigate to `logs/sb3/` or `logs/rllib/`
- Select the checkpoint directory

### 5. "ONNX export fails"

**Solution**:
```bash
pip install --upgrade torch onnx
```

Try different opset version in export settings.

## Next Steps

### Learn More

- **Read [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)** for detailed training workflows
- **Explore [CONFIGURATION.md](CONFIGURATION.md)** for all configuration options
- **Check [API.md](API.md)** for script parameters
- **Review [ARCHITECTURE.md](ARCHITECTURE.md)** for technical details

### Optimize Training

1. **Increase speedup** for faster iteration:
   - Navigate to advanced config (press `o` in Train tab)
   - Set `sb3_speedup` to 20-50
   
2. **Use parallel environments**:
   - Set `sb3_n_parallel` to 4-8
   - Requires more CPU/RAM

3. **Adjust network architecture**:
   - Smaller networks train faster: `[32, 32]`
   - Larger networks more powerful: `[128, 128, 64]`

### Multi-Agent Training

1. **Switch to multi-agent mode**: Press `m` in Train tab
2. **Generate `.rlcontroller/rllib_config.yaml`** with `g` (or edit it manually)
3. **Define multiple policies** for different agents
4. **Train**: Press `t`

See [Godot RL Agents multi-agent examples](https://github.com/edbeeching/godot_rl_agents) for configuration templates.

### Visual Debugging

For seeing what your agent is doing:

1. Enable visualization: Set `sb3_viz` to true in advanced config
2. Reduce speedup: Set `sb3_speedup` to 1
3. Train: You'll see the game window

Good for:
- Verifying observations are correct
- Watching agent behavior
- Debugging reward function

## Keyboard Shortcuts Cheat Sheet

### Global
- `1-4` - Jump to tab
- `?` - Help overlay
- `q` or `Esc` - Quit (with confirmation)
- `↑↓←→` - Navigate

### Home Tab
- `n` - New project
- `d` - Delete project
- `Enter` - Select project
- `p` - Check Python environment

### Train Tab
- `t` - Start training
- `c` - Cancel training
- `m` - Toggle single/multi-agent
- `s` - Save configuration
- `o` - Advanced configuration
- `d` - Run demo

### Metrics Tab
- `↑↓` - Scroll history
- `←→` - Switch metric categories
- `p` - Cycle through policies
- `f` - Toggle focus mode

### Export Tab
- `e` - Execute export
- `m` - Toggle SB3/RLlib mode
- `b` - Browse files
- `s` - Save export config

## Tips for Success

### 1. Start Small
- Train for 100K steps first
- Verify everything works
- Then do long training runs

### 2. Monitor Rewards
- Rewards should generally increase
- If stuck, try different learning rate
- Check reward function in Godot

### 3. Save Checkpoints
- Enable checkpoint saving for long runs
- Set checkpoint frequency: 100,000-500,000 steps
- Can resume if training crashes

### 4. Keep Notes
- Document what works
- Track hyperparameters
- Version control your configs

### 5. Experiment
- Try different architectures
- Adjust learning rates
- Test various speedup values

## Getting Help

### Check Documentation
1. This Quick Start
2. [TRAINING_GUIDE.md](../TRAINING_GUIDE.md)
3. [README.md](../README.md)
4. [Godot RL Agents docs](https://github.com/edbeeching/godot_rl_agents)

### Debug Steps
1. Check Python environment (`p` key)
2. Verify file paths are correct
3. Look at training output for errors
4. Try demo script (`d` key)
5. Test with small timesteps first

### Common Solutions
- **Restart controller**: Sometimes helps with state issues
- **Recreate project**: If config corrupted
- **Check file permissions**: Ensure binaries are executable
- **Update dependencies**: `pip install --upgrade -r requrements.txt`

## What's Next?

You now know how to:
- ✅ Set up the controller
- ✅ Create projects
- ✅ Configure training
- ✅ Start training
- ✅ Monitor progress
- ✅ Export models
- ✅ Use models in Godot

**Now**: Create amazing AI for your games! 🎮🤖

For advanced topics, see the other documentation files.

---

**Happy Training!**

Last Updated: November 5, 2025
