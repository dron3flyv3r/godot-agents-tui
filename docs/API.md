# API Reference

This project exposes a small CLI plus several Python entry points. All commands support `--help` for full details.

## controller-mk2 (Rust binary)
Global flags:
- `--exp`: launch the experimental MARS-only UI variant.
- `--log <PATH>`: write debug logs to a file.

Subcommands:
- `train [--config_file rllib_config.yaml] [--restore PATH] [--resume DIR] [--experiment_dir logs/rllib] [--generate] [--debug_panel]`
  - Runs `rllib_training_script.py` with the provided YAML config. `--generate` writes a default `rllib_config.yaml` and exits.
- `simulator [--env_path PATH] [--mode auto|single|multi] [--show_window] [--headless] [--seed INT] [--step_delay SEC] [--restart_delay SEC] [--max_episodes N] [--max_steps N] [--no_auto_restart] [--log_tracebacks]`
  - Wraps `simulator.py` and prints prettified events/actions.
- `export [--config_file rllib_config.yaml] [--checkpoint PATH] [--output_dir onnx_export]`
  - Runs `convert_rllib_to_onnx.py` with the given config and checkpoint.

If no subcommand is provided, the interactive TUI launches.

## Python scripts

### stable_baselines3_training_script.py
Trains a single-agent PPO policy with Stable Baselines3.
Key arguments:
- `--env_path PATH` (optional for editor training)
- `--experiment_dir`, `--experiment_name`
- `--timesteps`, `--seed`
- `--viz`, `--speedup`, `--n_parallel`
- `--policy-type` (`mlp|cnn|lstm|grn`) and shape arguments (`--policy-hidden-layers`, `--cnn-channels`, `--lstm-hidden-size`, `--lstm-num-layers`, `--grn-hidden-size`)
- Checkpointing/export: `--resume_model_path`, `--save_model_path`, `--save_checkpoint_frequency`, `--onnx_export_path`
- Learning rate schedule: `--linear_lr_schedule`
The controller sets `CONTROLLER_METRICS` so this script emits `@METRIC` lines for the Metrics tab.

### rllib_training_script.py
Runs RLlib with a YAML config (generated from `training_config.json`).
Key arguments:
- `--config_file rllib_config.yaml`
- `--restore` or `--resume` to continue from a checkpoint/trial directory
- `--experiment_dir` for logs and checkpoints
- `--debug_panel` to print a one-step rollout before training
The script registers custom models from `custom_models/` and supports both single- and multi-agent Godot environments.

### simulator.py
Exercises a Godot environment with random actions and streams structured output.
Key arguments:
- `--env-path PATH` (omit to connect to the editor for multi-agent)
- `--mode single|multi|auto`
- `--show-window` / `--headless`
- `--seed`, `--step-delay`, `--restart-delay`
- `--max-episodes`, `--max-steps`
- `--no-auto-restart`, `--log-tracebacks`
Outputs `@SIM_EVENT` and `@SIM_ACTION` lines consumed by the Simulator tab.

### interface.py
Loads a trained agent into the Godot editor and streams actions/events.
Key arguments:
- `--agent-type sb3|rllib`, `--agent-path PATH`
- `--mode single|multi`
- Environment options: `--env-path`, `--show-window`/`--headless`, `--action-repeat`, `--speedup`, `--seed`, `--step-delay`, `--restart-delay`, `--no-auto-restart`
- RLlib options: `--checkpoint-number`, `--policy-id`, `--multiagent`
- SB3 options: `--algo` to force the checkpoint algorithm
Outputs `@INTERFACE_EVENT` and `@INTERFACE_ACTION` lines consumed by the Interface tab.

### Export scripts
- `convert_sb3_to_onnx.py <MODEL_ZIP> -o <OUTPUT> [--algo ppo] [--opset 13] [--ir-version 9] [--use-obs-array] [--no-verify]`
- `convert_rllib_to_onnx.py <CHECKPOINT_DIR> -o <OUTPUT_DIR> [--policy ID] [--checkpoint-number N] [--opset 13] [--ir-version 9] [--multiagent/--no-multiagent] [--prefix NAME]`
Both scripts rely on Torch/ONNX and the Godot RL wrappers; the Export tab fills these arguments for you.
