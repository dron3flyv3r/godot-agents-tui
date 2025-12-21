# Training Guide

This guide focuses on running training jobs from the TUI.

## Tabs and modes
- **Home**: select or create a project; the active project decides where configs, logs, and runs are saved.
- **Train**: configure SB3 or RLlib runs and launch them.
- **Metrics**: inspect live samples from the active run and overlay saved runs.
- **Projects**: browse stored runs and archives when you are not training.

## Preparing a project
1. Create a project from the Home tab (`n`) and activate it (`Enter`).
2. Confirm the `.rlcontroller` folder now exists in the project directory with `training_config.json` and `rllib_config.yaml`.
3. Ensure your Python environment is active and has the packages from `requrements.txt`.

## Training configuration
Key fields in the Train tab map directly to `training_config.json`:
- **Training Mode**: `single_agent` (SB3) or `multi_agent` (RLlib).
- **Environment Path**: exported Godot binary for SB3; optional for RLlib if you connect to the editor.
- **Timesteps / Experiment Name**: run length and label for SB3 runs.
- **SB3 basics**: policy type (MLP/CNN/LSTM/GRN), speedup, parallel envs, PPO hyperparameters.
- **RLlib basics**: algorithm (PPO/DQN/SAC/APPO/IMPALA), worker counts, rollout length, checkpoints, and stop criteria.
- **Advanced RLlib**: policy backbone selection, learning rates, batch sizes, stopping by time/timesteps, and checkpoint frequency.

When you edit a field, the TUI writes changes back to `training_config.json`. RLlib-specific values are also mirrored into `.rlcontroller/rllib_config.yaml` before each run.

## Starting and stopping runs
- Press `t` in the Train tab to launch the appropriate Python script (`stable_baselines3_training_script.py` or `rllib_training_script.py`).
- The controller sets `CONTROLLER_METRICS` so both scripts emit structured `@METRIC` lines that the UI charts.
- Press `c` to send SIGINT for a graceful stop; RLlib stop files and sustained reward thresholds are honored if configured.

## Watching progress
- Output from the Python process streams into the Train tab so you can see checkpoints and warnings immediately.
- The Metrics tab charts reward and episode-length samples as they arrive. Switching focus allows you to view different sample buckets or compare against saved runs.
- Completed runs are serialized into `.rlcontroller/runs/<timestamp>_<name>.json` along with optional paged metric logs for large jobs.

## Resuming work
- SB3 runs can resume via `resume_model_path` (set in `training_config.json`).
- RLlib runs can resume from a checkpoint directory using the **Resume From** field; the controller keeps track of checkpoint frequency and offsets to label new checkpoints correctly.
- Use the Projects tab to reopen saved runs, inspect logs, or export archives that include configs, runs, and optional model files.

## Tips
- Start with short runs (e.g., 100k timesteps) to validate observation/action wiring before launching longer jobs.
- Keep the Godot window hidden while training unless you are debugging visuals; both SB3 and RLlib have switches for this.
- Use the simulator tab before training to confirm your environment produces sensible observations and rewards.
