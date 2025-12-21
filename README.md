# Godot Agents TUI

A Rust terminal UI for managing reinforcement-learning projects built with [Godot RL Agents](https://github.com/edbeeching/godot_rl_agents). The app wraps the provided Python training, export, simulator, and interface scripts so you can create projects, launch SB3 or RLlib training, monitor metrics, and export ONNX models without leaving the terminal.

## Requirements
- Rust toolchain (stable)
- Python 3.9+ with `pip`
- Godot build or exported environment for your game
- Python dependencies from `requrements.txt` (includes SB3, RLlib, Godot RL, Torch, ONNX)

## Quick start
1. Create a virtual environment and install Python dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requrements.txt
   ```
2. Build and run the TUI:
   ```bash
   cargo run --release
   ```
3. In the Home tab press `n` to make a project, set the environment path in the Train tab, and press `t` to start SB3 or RLlib training.
4. Use the Metrics tab to watch charts, the Simulator tab to validate observations/actions, the Interface tab to run trained policies in-editor, and the Export tab to write ONNX checkpoints.

Binary builds look for the Python interpreter and scripts relative to the repository by default. Set `CONTROLLER_PYTHON_BIN` or `CONTROLLER_SCRIPTS_ROOT` if you need to point at a different environment.

## Project layout
Each project stores its state under a `.rlcontroller/` folder next to your game files:
- `project.json` and `training_config.json` track metadata and training/export settings
- `rllib_config.yaml` contains the generated RLlib trainer config
- `runs/` keeps saved metrics and logs for completed sessions

## Documentation
See the docs directory for focused guides:
- [docs/QUICK_START.md](docs/QUICK_START.md) for setup and the first run
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for day-to-day training workflows
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for configuration reference
- [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues
