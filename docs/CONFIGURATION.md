# Configuration

The controller keeps project-specific settings in `.rlcontroller/` inside each project directory. Files are JSON unless noted otherwise.

## Project files
- `project.json` – project name and creation metadata.
- `training_config.json` – SB3/RLlib settings edited from the Train tab.
- `rllib_config.yaml` – generated RLlib trainer config derived from `training_config.json` before each multi-agent run.
- `export_config.json` – last-used SB3/RLlib export options from the Export tab.
- `runs/` – saved run manifests and paged metric logs.
- `sessions.json` / archives – session metadata and optional exported bundles when using the Projects tab.

## Training settings
`training_config.json` mirrors the fields shown in the Train tab. Important keys:

### Shared
- `mode`: `"single_agent"` (SB3) or `"multi_agent"` (RLlib).
- `env_path`: exported Godot binary; RLlib can leave this empty for in-editor multi-agent runs.
- `timesteps`: SB3 training length.
- `experiment_name`: label for SB3 runs and checkpoints.

### SB3 highlights
- `sb3_policy_type`: `mlp`, `cnn`, `lstm`, or `grn` (mapped to custom models in `custom_models`).
- `sb3_speedup`, `sb3_n_parallel`, `sb3_viz`: environment speed, parallel executables, and window toggle.
- PPO hyperparameters: `sb3_learning_rate`, `sb3_batch_size`, `sb3_n_steps`, `sb3_gamma`, `sb3_gae_lambda`, `sb3_ent_coef`, `sb3_clip_range`, `sb3_vf_coef`, `sb3_max_grad_norm`.
- Network shapes: `sb3_policy_layers`, `sb3_cnn_channels`, `sb3_lstm_hidden_size`, `sb3_lstm_num_layers`, `sb3_grn_hidden_size`.

### RLlib highlights
- `rllib_algorithm`: `ppo`, `dqn`, `sac`, `appo`, or `impala`.
- Worker layout: `rllib_num_workers`, `rllib_num_envs_per_worker`, `rllib_rollout_fragment_length`.
- Optimization: `rllib_train_batch_size`, `rllib_sgd_minibatch_size`, `rllib_num_sgd_iter`, `rllib_lr`, `rllib_gamma`, `rllib_lambda`, `rllib_clip_param`, `rllib_entropy_coeff`, `rllib_vf_loss_coeff`, `rllib_grad_clip`.
- Model choice: `rllib_policy_type` plus `rllib_fcnet_hiddens`, `rllib_cnn_channels`, `rllib_lstm_cell_size`, `rllib_lstm_num_layers`, `rllib_lstm_include_prev_actions`, `rllib_grn_hidden_size`.
- Runtime: `rllib_framework`, `rllib_activation`, `rllib_batch_mode`, `rllib_num_gpus`, `rllib_max_seq_len`, `rllib_env_action_repeat`, `rllib_env_speedup`, `rllib_show_window`.
- Checkpoints and stopping: `rllib_checkpoint_frequency`, `rllib_resume_from`, `rllib_stop_mode`, `rllib_stop_time_seconds`, `rllib_stop_timesteps_total`, `rllib_stop_sustained_reward_*`, `rllib_stop_file_*`.

### Experimental MARS fields
When launched with `--exp`, additional `mars_*` keys are available for the MARS workflow (environment path/name, method, algorithm, batch sizes, etc.). These live alongside the standard training config.

## Export settings
`export_config.json` stores the last export choices from the Export tab:
- SB3: `sb3_model_path`, `sb3_output_path`, `sb3_algo`, `sb3_opset`, `sb3_ir_version`, `sb3_use_obs_array`, `sb3_skip_verify`.
- RLlib: `rllib_checkpoint_path`, `rllib_checkpoint_number`, `rllib_output_dir`, `rllib_policy_id`, `rllib_opset`, `rllib_ir_version`, `rllib_multiagent`, `rllib_prefix`.

## Metrics and UI settings
- `metrics_settings.json` (in the project config directory) records chart palettes, legend positions, and axis preferences chosen from the Metrics tab.
- Session and archive data are stored in `sessions.json` and optional exported archives managed via the Projects tab.

## Environment variables
- `CONTROLLER_PROJECTS_ROOT`: override the default location for the global project index.
- `CONTROLLER_PYTHON_BIN`: force a specific Python executable.
- `CONTROLLER_SCRIPTS_ROOT`: where the Python scripts are located; defaults to the repository root.
- `CONTROLLER_METRICS`: set automatically by the TUI so Python scripts emit `@METRIC` lines.
