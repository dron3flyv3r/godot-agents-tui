# Troubleshooting

## Python package errors
**Symptom:** `ModuleNotFoundError` for SB3, RLlib, or Godot RL.

**Fix:** Activate your virtualenv and install dependencies:
```bash
source .venv/bin/activate
pip install -r requrements.txt
```

## Training fails to start
**Symptom:** Train tab shows an immediate exit or path error.

**Fix:**
- Confirm the Environment Path points to an executable Godot export (or leave empty for RLlib editor testing).
- Ensure the active project directory exists and is writable.
- For RLlib, confirm `.rlcontroller/rllib_config.yaml` was written; regenerate by editing/saving the Train tab.

## No metrics are displayed
**Symptom:** Charts stay empty during training.

**Fix:**
- Metrics only arrive while a run is active. Check the Train tab for script output or errors.
- The controller sets `CONTROLLER_METRICS` automatically; avoid overriding it when launching scripts manually.
- For long runs, wait a few iterations—samples are batched before the UI plots them.

## Resume or checkpoint issues
**Symptom:** RLlib cannot find the provided checkpoint or resumes with the wrong index.

**Fix:**
- Use absolute paths or paths relative to the project root in `rllib_resume_from`.
- Keep `rllib_checkpoint_frequency` consistent between runs so checkpoint numbering aligns.
- If a checkpoint folder was moved, update the path in `training_config.json` and retry.

## Export failures
**Symptom:** ONNX conversion scripts exit with missing ops or Torch errors.

**Fix:**
- Upgrade Torch/ONNX inside the virtualenv: `pip install --upgrade torch onnx`.
- Try a lower opset/IR version from the Export tab.
- Verify the checkpoint path points at a valid SB3 `.zip` or RLlib `checkpoint_*/` directory.

## Simulator or Interface cannot connect
**Symptom:** Tabs show repeated reconnect attempts.

**Fix:**
- Make sure the Godot editor or export is running and listening on the expected port.
- Toggle headless/window options to match how the environment was built.
- For multi-agent, ensure the environment uses the PettingZoo wrapper provided by Godot RL Agents.

## Training will not stop
**Symptom:** Pressing `c` appears to have no effect.

**Fix:**
- The controller sends SIGINT first, then SIGKILL after a timeout. Give RLlib time to flush checkpoints.
- Enable the RLlib stop file option and create the configured file path to request shutdown from outside the TUI.
