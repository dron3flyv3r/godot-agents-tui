# Architecture

The controller pairs a Rust TUI with a small set of Python scripts. The Rust side manages state, spawns processes, and renders the UI; the Python side performs training, simulation, and exports.

## Rust components
- **`src/main.rs`**: application entry point. Sets up the crossterm terminal, routes key events, and switches between tabs.
- **`src/app/`**: core state and process management. Tracks projects, training/export processes, metrics streams, file browsers, and session archives.
- **`src/domain/projects.rs`**: project index stored under `CONTROLLER_PROJECTS_ROOT`, plus helpers to create or import projects.
- **`src/ui/`**: Ratatui widgets and layouts for the tabs (Home, Train, Metrics, Simulator, Interface, Export, Projects, Settings). The UI reflects the state held in `App` and responds to key events from `main.rs`.
- **`src/cli.rs`**: optional CLI subcommands (`train`, `simulator`, `export`) for running the Python scripts without the TUI while keeping the same metric formatting.

## Python scripts
- **`stable_baselines3_training_script.py`**: SB3 PPO training with optional checkpoints, ONNX export, and custom policy backbones.
- **`rllib_training_script.py`**: RLlib training that consumes `rllib_config.yaml`, supports multi-agent environments, and emits metrics.
- **`simulator.py`**: drives random actions against a Godot environment to validate wiring; emits `@SIM_EVENT` and `@SIM_ACTION` records.
- **`interface.py`**: runs trained SB3 or RLlib policies directly in the Godot editor and streams `@INTERFACE_EVENT` and `@INTERFACE_ACTION` records.
- **`convert_sb3_to_onnx.py` / `convert_rllib_to_onnx.py`**: convert checkpoints to ONNX for Godot RL Agents.

## Data flow
- The TUI spawns Python processes with `-u` for unbuffered output and reads stdout/stderr lines.
- Structured lines prefixed with `@METRIC`, `@SIM_EVENT`, `@SIM_ACTION`, `@INTERFACE_EVENT`, or `@INTERFACE_ACTION` are parsed into in-memory structures and displayed in the Metrics, Simulator, or Interface tabs.
- Run summaries and logs are saved to `.rlcontroller/runs/` after each training session, including paged metric logs for large jobs.

## Storage layout
```
<project>/
├── .rlcontroller/
│   ├── project.json
│   ├── training_config.json
│   ├── export_config.json
│   ├── rllib_config.yaml
│   ├── metrics_settings.json
│   ├── runs/
│   └── sessions.json (plus archives)
└── logs/
```
`ProjectManager` also maintains a global `index.json` under `CONTROLLER_PROJECTS_ROOT` so the TUI can surface recent projects regardless of the current working directory.

## Extensibility notes
- Custom SB3 or RLlib models can be registered in `custom_models/`; the Train tab exposes policy type and shape fields.
- If you add new structured output, use a unique prefix so the parsers in `src/app/state.rs` can dispatch it without ambiguity.
- Most UI colors, legends, and chart options are stored in `metrics_settings.json` and can be extended without changing saved runs.
