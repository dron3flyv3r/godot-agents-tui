# Troubleshooting Guide

Comprehensive solutions to common issues with the Godot RL Training Controller.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Python Environment Problems](#python-environment-problems)
- [Training Issues](#training-issues)
- [Metric Display Problems](#metric-display-problems)
- [Export Issues](#export-issues)
- [Performance Problems](#performance-problems)
- [Configuration Issues](#configuration-issues)
- [UI/Display Issues](#uidisplay-issues)
- [Godot Integration Issues](#godot-integration-issues)
- [Advanced Debugging](#advanced-debugging)

---

## Installation Issues

### Rust Compilation Fails

**Symptoms**:
```
error: failed to compile controller-mk2
```

**Solutions**:

1. **Update Rust**:
   ```bash
   rustup update stable
   ```

2. **Check Rust version**:
   ```bash
   rustc --version  # Should be 1.70+
   ```

3. **Clean and rebuild**:
   ```bash
   cargo clean
   cargo build --release
   ```

4. **Check dependencies**:
   - On Linux: `sudo apt install build-essential`
   - On Mac: Install Xcode Command Line Tools

### Python Package Installation Fails

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement
```

**Solutions**:

1. **Update pip**:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```

2. **Check Python version**:
   ```bash
   python --version  # Should be 3.8+
   ```

3. **Install packages individually**:
   ```bash
   pip install stable-baselines3
   pip install "ray[rllib]"
   pip install godot-rl
   pip install torch onnx
   ```

4. **Platform-specific issues**:
   
   **Mac M1/M2**:
   ```bash
   # Use conda for better ARM support
   conda create -n rl python=3.9
   conda activate rl
   pip install -r requrements.txt
   ```
   
   **Windows**:
   ```bash
   # May need Visual C++ Build Tools
   # Download from Microsoft
   ```

### Binary Won't Run

**Symptoms**:
```
Permission denied
```

**Solutions**:

1. **Make executable**:
   ```bash
   chmod +x target/release/controller-mk2
   ```

2. **Check architecture**:
   ```bash
   file target/release/controller-mk2
   # Should match your system architecture
   ```

---

## Python Environment Problems

### "Python packages not found"

**Symptoms**: In Home tab, packages show ✗ (red X)

**Solutions**:

1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```
   
   Verify activation:
   ```bash
   which python  # Should point to .venv/bin/python
   ```

2. **Reinstall packages**:
   ```bash
   pip install -r requrements.txt
   ```

3. **Check installation manually**:
   ```bash
   python check_py_env.py
   ```

4. **Run controller from activated environment**:
   ```bash
   source .venv/bin/activate
   ./target/release/controller-mk2
   ```

### Ray/RLlib Won't Install

**Symptoms**:
```
ERROR: Cannot install ray[rllib]
```

**Solutions**:

1. **Check Python version** (Ray needs 3.8-3.10):
   ```bash
   python --version
   ```

2. **Install without extras first**:
   ```bash
   pip install ray
   pip install ray[rllib]
   ```

3. **Use conda** (recommended for Ray):
   ```bash
   conda install -c conda-forge ray-rllib
   ```

4. **Platform-specific**:
   - **Windows**: May need WSL2 for full Ray support
   - **Mac M1/M2**: Use conda or wait for native wheels

### Stable Baselines3 Import Error

**Symptoms**:
```
ModuleNotFoundError: No module named 'stable_baselines3'
```

**Solutions**:

1. **Install directly**:
   ```bash
   pip install stable-baselines3[extra]
   ```

2. **Check for conflicts**:
   ```bash
   pip list | grep -i baselines
   ```

3. **Reinstall clean**:
   ```bash
   pip uninstall stable-baselines3
   pip install stable-baselines3
   ```

---

## Training Issues

### Training Doesn't Start

**Symptoms**: Press `t`, nothing happens or immediate error

**Checklist**:

1. ✅ **Environment path set and correct**
   - Navigate to Train tab
   - Verify path to Godot binary
   - Check file exists: `ls -l /path/to/game.x86_64`

2. ✅ **Binary is executable**:
   ```bash
   chmod +x /path/to/game.x86_64
   ```

3. ✅ **Python environment activated**:
   ```bash
   source .venv/bin/activate
   ```

4. ✅ **Training script exists**:
   ```bash
   ls stable_baselines3_training_script.py
   ls rllib_training_script.py
   ```

5. ✅ **Configuration saved**: Press `s` in Train tab

**Debug**:
- Check training output in Train tab
- Look for error messages (prefixed with `!`)
- Try running script manually:
  ```bash
  python stable_baselines3_training_script.py --env_path /path/to/game.x86_64 --timesteps 1000
  ```

### Training Starts But Crashes Immediately

**Symptoms**:
```
Process exited with code 1
```

**Common Causes**:

1. **Godot binary not compatible**:
   - Wrong architecture (x86_64 vs ARM)
   - Missing dependencies
   - Run binary directly to check:
     ```bash
     /path/to/game.x86_64
     ```

2. **Environment observation/action space issues**:
   - Check Godot RL Agents configuration
   - Verify observation space matches expected
   - Check action space is defined

3. **Port already in use**:
   - Godot RL uses specific ports
   - Kill existing processes:
     ```bash
     pkill -f "game.x86_64"
     ```

4. **Missing libraries** (Linux):
   ```bash
   ldd /path/to/game.x86_64  # Check dependencies
   sudo apt install libgl1 libasound2  # Common requirements
   ```

### Training Hangs/Freezes

**Symptoms**: No output, no progress

**Solutions**:

1. **Cancel and restart**: Press `c` to send SIGINT

2. **Check Godot process**:
   ```bash
   ps aux | grep godot
   ```
   Kill if hung:
   ```bash
   pkill -9 -f "game.x86_64"
   ```

3. **Reduce speedup**: High speedup can cause instability
   - Set `sb3_speedup` to 1-10
   - Test incrementally

4. **Check resource usage**:
   ```bash
   htop  # or top
   ```
   - Out of memory? Reduce `sb3_n_parallel`
   - CPU maxed? Reduce workers

5. **Enable visualization**: Set `sb3_viz` to true to see what's happening

### Training Very Slow

**Symptoms**: Takes forever to complete steps

**Solutions**:

1. **Increase speedup**:
   - SB3: Set `sb3_speedup` to 20-50
   - Verify game can handle it

2. **Use parallel environments**:
   - SB3: Set `sb3_n_parallel` to 4-8
   - More CPU cores = better

3. **Disable visualization**:
   - Set `sb3_viz` to false
   - Set `rllib_show_window` to false

4. **Optimize Godot game**:
   - Disable unnecessary rendering
   - Simplify physics
   - Reduce visual effects

5. **Use headless Godot export**:
   - Export without rendering
   - Faster for training

6. **Hardware**:
   - More CPU cores help with parallel environments
   - GPU not needed for PPO (CPU-based)

### Rewards Not Improving

**Symptoms**: Rewards stay flat or decrease

**Solutions**:

1. **Check reward function** in Godot:
   - Is it providing meaningful signals?
   - Are rewards too sparse?
   - Try denser rewards

2. **Adjust learning rate**:
   - Too high: Training unstable
   - Too low: No learning
   - Try: 0.0001, 0.0003, 0.001

3. **Increase training time**:
   - Some tasks need millions of steps
   - Try 5-10M timesteps

4. **Adjust exploration**:
   - Increase `entropy_coeff` for more exploration
   - Default: 0.01, try 0.05

5. **Network capacity**:
   - Larger networks: `[128, 128, 64]`
   - More layers for complex tasks

6. **Check observation space**:
   - Are observations normalized?
   - Do they contain useful information?
   - Try adding/removing features

---

## Metric Display Problems

### No Metrics Showing

**Symptoms**: Metrics tab is empty during training

**Solutions**:

1. **Training must be running**: Metrics only appear during active training

2. **Check metric emission**:
   - `CONTROLLER_METRICS` should be set (controller does this automatically)
   - Look for `@METRIC` lines in training output

3. **Scroll down**: Metrics may be below visible area

4. **Wait for first iteration**:
   - SB3: Metrics every N steps (configurable)
   - RLlib: Metrics every iteration (can take minutes)

### Metrics Look Wrong

**Symptoms**: Numbers don't make sense

**Solutions**:

1. **Check reward scale**:
   - Rewards too large/small?
   - Normalize in Godot

2. **Policy-specific metrics**:
   - Press `p` to cycle through policies
   - Each policy tracked separately

3. **Custom metrics**:
   - Press `f` to toggle focus mode
   - View custom metrics from Godot

### Charts Not Updating

**Symptoms**: Charts frozen

**Solutions**:

1. **Check training is active**: Look at training output

2. **Metric buffer full**:
   - Controller keeps last 2000 samples
   - Older samples drop off

3. **Restart controller**: Sometimes helps with display issues

---

## Export Issues

### Can't Find Checkpoint

**Symptoms**: "Checkpoint not found" error

**Solutions**:

1. **Use file browser**: Press `b` in Export tab

2. **Check training completed**:
   - Look in `logs/sb3/` or `logs/rllib/`
   - Verify checkpoint directories exist

3. **For SB3**:
   - Look for `.zip` files
   - Check `logs/sb3/<experiment_name>/`

4. **For RLlib**:
   - Look for `checkpoint_*` directories
   - Format: `logs/rllib/PPO_<timestamp>/PPO_<trial>/checkpoint_N/`

5. **Use absolute paths**: Relative paths can cause issues

### ONNX Export Fails

**Symptoms**:
```
RuntimeError: ONNX export failed
```

**Solutions**:

1. **Update dependencies**:
   ```bash
   pip install --upgrade torch onnx
   ```

2. **Try different opset**:
   - Start with opset 11
   - Try 13, 15 if that fails

3. **Check model compatibility**:
   - Some architectures don't export well
   - Try simpler network

4. **Disable verification**:
   - Set `skip_verify` to true
   - Only if other options fail

5. **Check PyTorch version**:
   ```bash
   pip show torch
   ```
   - Need 1.10+

### Exported Model Doesn't Work in Godot

**Symptoms**: Model loads but gives errors or wrong outputs

**Solutions**:

1. **Check ONNX version compatibility**:
   - Godot RL Agents supports specific versions
   - Try opset 13 (most compatible)

2. **Verify input shape**:
   - Observation space must match
   - Check dimensions in Godot

3. **Test model outside Godot**:
   ```python
   import onnx
   model = onnx.load("model.onnx")
   onnx.checker.check_model(model)
   ```

4. **Re-export with different settings**:
   - Try `use_obs_array` option
   - Adjust IR version

5. **Check action space**:
   - Discrete vs continuous
   - Number of actions matches

---

## Performance Problems

### Controller Sluggish

**Symptoms**: UI slow to respond

**Solutions**:

1. **Use release build**:
   ```bash
   cargo build --release
   ```

2. **Clear old logs**:
   - Delete old training runs
   - Clean `logs/` directory

3. **Reduce metric history**:
   - Controller keeps 2000 samples
   - Restart controller to clear

4. **Terminal size**:
   - Larger terminals require more rendering
   - Resize to reasonable size

### High Memory Usage

**Symptoms**: System running out of RAM

**Solutions**:

1. **Reduce parallel environments**:
   - Lower `sb3_n_parallel` to 1-2
   - Lower `rllib_num_workers`

2. **Smaller batch sizes**:
   - Reduce `train_batch_size`
   - Reduce `n_steps`

3. **Clear logs periodically**:
   ```bash
   rm -rf logs/old_experiments/
   ```

4. **Restart controller**: Clears metric buffers

### Training Consumes All CPU

**Symptoms**: System becomes unresponsive

**Solutions**:

1. **Limit workers**:
   - RLlib: Set `num_workers` to `(CPU cores - 2)`
   - Leave cores for system

2. **Reduce parallel environments**:
   - SB3: Lower `n_parallel`

3. **Lower speedup**:
   - Reduces CPU load per environment

4. **Use `nice`**:
   ```bash
   nice -n 10 ./target/release/controller-mk2
   ```

---

## Configuration Issues

### Configuration Not Saving

**Symptoms**: Changes lost on restart

**Solutions**:

1. **Press `s` to save**: Changes aren't auto-saved

2. **Check file permissions**:
   ```bash
   ls -la projects/my_project/
   chmod 644 projects/my_project/*.json
   ```

3. **Verify JSON syntax**:
   - Invalid JSON won't save
   - Check for trailing commas

4. **Look for error messages**: Status bar shows save errors

### Settings Not Applied

**Symptoms**: Training ignores configuration

**Solutions**:

1. **Save configuration**: Press `s`

2. **Correct project selected**: Check Home tab

3. **Configuration file in right place**:
   - Should be in `projects/<project>/training_config.json`

4. **Script reads config correctly**:
   - Check training output for loaded settings

### Advanced Config Won't Open

**Symptoms**: Pressing `o` does nothing

**Solutions**:

1. **Project must be selected**: Select project in Home tab

2. **Configuration must exist**: Try saving first (`s` key)

3. **Check for config errors**: Invalid JSON prevents loading

---

## UI/Display Issues

### Display Garbled/Broken

**Symptoms**: Characters misaligned, colors wrong

**Solutions**:

1. **Update terminal emulator**: Some terminals have better TUI support

2. **Check terminal size**:
   ```bash
   echo $COLUMNS $LINES
   ```
   - Need at least 80x24

3. **Resize terminal**: Sometimes fixes layout issues

4. **Try different terminal**:
   - Recommended: Alacritty, iTerm2, Windows Terminal
   - Avoid: Basic terminals, some SSH clients

### Colors Not Showing

**Symptoms**: Everything same color

**Solutions**:

1. **Enable color support**:
   ```bash
   export TERM=xterm-256color
   ```

2. **Check terminal settings**: Enable 256 colors

3. **Some terminals don't support**: Controller works without colors

### Help Overlay Won't Close

**Symptoms**: Stuck on help screen

**Solutions**:

1. **Press Esc or ?**: Should close help

2. **Restart controller**: Rare bug, restart fixes

---

## Godot Integration Issues

### Godot Can't Connect to Training

**Symptoms**: "Connection failed" in Godot logs

**Solutions**:

1. **Check port availability**:
   ```bash
   netstat -tuln | grep 11008  # Default Godot RL port
   ```

2. **Kill existing processes**:
   ```bash
   pkill -f "game.x86_64"
   ```

3. **Firewall issues**: Allow port 11008

4. **Check Godot RL Agents setup**:
   - Verify plugin installed
   - Check sync node configuration

### Observations Not Working

**Symptoms**: Training starts but agent doesn't learn

**Solutions**:

1. **Check observation space in Godot**:
   ```gdscript
   func get_obs():
       return [pos.x, pos.y, velocity.x, velocity.y]
   ```

2. **Verify shape matches**:
   - Controller expects specific shape
   - Print observations to debug

3. **Normalize observations**:
   - Large values hurt training
   - Scale to [-1, 1] or [0, 1]

4. **Check for NaN/Inf**:
   - Can crash training
   - Add validation in Godot

### Actions Not Applied

**Symptoms**: Agent doesn't move/act

**Solutions**:

1. **Check action space**:
   - Discrete: Integer index
   - Continuous: Float array

2. **Verify action mapping in Godot**:
   ```gdscript
   func set_action(action):
       if action[0] == 0:
           move_left()
       elif action[0] == 1:
           move_right()
   ```

3. **Print actions**: Debug what's being sent

### Reward Function Not Working

**Symptoms**: Rewards always zero or not changing

**Solutions**:

1. **Check reward calculation**:
   ```gdscript
   func get_reward():
       var reward = 0.0
       if hit_target:
           reward += 10.0
       if hit_wall:
           reward -= 5.0
       return reward
   ```

2. **Make rewards meaningful**:
   - Not too sparse
   - Not too dense
   - Properly scaled

3. **Test reward function**:
   - Print rewards
   - Verify they make sense

---

## Advanced Debugging

### Enable Verbose Logging

**For training scripts**:
```bash
# Run manually with verbose output
python stable_baselines3_training_script.py \
  --env_path ./game.x86_64 \
  --timesteps 1000 2>&1 | tee training.log
```

### Check Process Status

```bash
# Find training processes
ps aux | grep python

# Monitor resource usage
top -p $(pgrep -f training_script)

# Check open files
lsof -p <pid>
```

### Debug Python Scripts

Add debug prints:
```python
# In training script
import sys
print(f"DEBUG: Starting with env_path={args.env_path}", file=sys.stderr)
```

### Validate Configuration Files

```bash
# Check JSON syntax
python -m json.tool projects/my_project/training_config.json

# Check YAML syntax
python -c "import yaml; yaml.safe_load(open('.rlcontroller/rllib_config.yaml'))"
```

### Test ONNX Models

```python
import onnx
import numpy as np

# Load model
model = onnx.load("model.onnx")

# Check model
onnx.checker.check_model(model)

# Print input/output info
print("Inputs:", [i.name for i in model.graph.input])
print("Outputs:", [o.name for o in model.graph.output])
```

### Network Debugging

```bash
# Check if Godot RL port is in use
netstat -tuln | grep 11008

# Test port connectivity
telnet localhost 11008

# Monitor network traffic
sudo tcpdump -i lo port 11008
```

### File System Checks

```bash
# Verify project structure
tree projects/my_project/

# Check permissions
ls -laR projects/my_project/

# Disk space
df -h .

# Inode usage
df -i .
```

---

## Still Having Issues?

### Gather Information

1. **Controller version**: Check `Cargo.toml`
2. **Python version**: `python --version`
3. **Package versions**: `pip list`
4. **OS**: `uname -a` (Linux/Mac) or `ver` (Windows)
5. **Error messages**: Copy exact error text
6. **Configuration**: Relevant JSON/YAML files

### Check Documentation

- [README.md](../README.md) - Overview
- [TRAINING_GUIDE.md](../TRAINING_GUIDE.md) - Training workflows
- [CONFIGURATION.md](CONFIGURATION.md) - Config details
- [API.md](API.md) - Script parameters
- [Godot RL Agents docs](https://github.com/edbeeching/godot_rl_agents)

### Clean Start

Sometimes a fresh start helps:

```bash
# Backup project
cp -r projects/my_project projects/my_project.backup

# Clean Python environment
deactivate
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requrements.txt

# Clean Rust build
cargo clean
cargo build --release

# Start fresh
./target/release/controller-mk2
```

---

**Last Updated**: November 5, 2025
