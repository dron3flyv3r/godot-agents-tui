# Quick Start

Follow these steps to get the controller running with a fresh project.

## 1) Install prerequisites
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requrements.txt
cargo build --release
```

## 2) Launch the TUI
```bash
./target/release/controller-mk2
```

Navigation cheatsheet:
- `1-8` switch tabs (Home, Train, Metrics, Simulator, Interface, Export, Projects, Settings)
- `?` opens the built-in help overlay
- `q` or `Esc` asks to quit

## 3) Create or select a project
- Go to the **Home** tab and press `n` to create a project.
- Pick a directory (the controller creates a `.rlcontroller/` folder and `logs/` inside it).
- Activate a project with `Enter`.

## 4) Configure training
In the **Train** tab:
- Set **Environment Path** to your exported Godot binary (or leave empty for in-editor RLlib multi-agent tests).
- Choose **Training Mode** (SB3 single-agent or RLlib multi-agent).
- Adjust key values like **Timesteps**, **Experiment Name**, and basic SB3/RLlib hyperparameters.
- Advanced fields and file pickers show up inline; save changes before starting a run.

## 5) Start a run
- Press `t` to launch training.
- Press `c` to send SIGINT if you need to stop.
- Output and metrics stream directly into the tab while the Python script runs.

## 6) Inspect metrics
- Open the **Metrics** tab to view live samples from the current run.
- Saved runs from `.rlcontroller/runs/` can be overlaid for comparison.
- Metrics are paged to avoid memory bloat; scroll to review history.

## 7) Validate with the simulator
- The **Simulator** tab runs `simulator.py` to fire random actions at your environment.
- Toggle single vs. multi-agent mode, headless vs. windowed, and auto-restart behavior from the tab controls.
- Action and event streams appear with prefixes `@SIM_ACTION` and `@SIM_EVENT`.

## 8) Export models
- In the **Export** tab pick SB3 or RLlib mode.
- Use the file browser to select checkpoints, set output paths, and choose opset/IR options.
- Press `e` to run the corresponding `convert_*_to_onnx.py` script.

## 9) Run agents in-editor
- The **Interface** tab wraps `interface.py` so you can load SB3 `.zip` files or RLlib checkpoints directly into a running Godot editor session.
- Configure agent type, checkpoint details, and window/headless options, then start streaming actions and events back into the UI.

## 10) Keep going
- Use the **Projects** tab to archive/import sessions, view saved runs, and manage storage.
- Adjust colors, animation, and chart preferences in **Settings**.
