import argparse
import json
import os
import pathlib
import signal
import sys
from datetime import datetime, timezone
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

from godot_rl.core.utils import can_import
from godot_rl.wrappers.onnx.stable_baselines_export import export_model_as_onnx
from godot_rl.wrappers.stable_baselines_wrapper import StableBaselinesGodotEnv

# To download the env source and binary:
# 1.  gdrl.env_from_hub -r edbeeching/godot_rl_BallChase
# 2.  chmod +x examples/godot_rl_BallChase/bin/BallChase.x86_64
if can_import("ray"):
    print("WARNING, stable baselines and ray[rllib] are not compatible")

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument(
    "--env_path",
    default=None,
    type=str,
    help="The Godot binary to use, do not include for in editor training",
)
parser.add_argument(
    "--experiment_dir",
    default="logs/sb3",
    type=str,
    help="The name of the experiment directory, in which the tensorboard logs and checkpoints (if enabled) are "
    "getting stored.",
)
parser.add_argument(
    "--experiment_name",
    default="experiment",
    type=str,
    help="The name of the experiment, which will be displayed in tensorboard and "
    "for checkpoint directory and name (if enabled).",
)
parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
parser.add_argument(
    "--resume_model_path",
    default=None,
    type=str,
    help="The path to a model file previously saved using --save_model_path or a checkpoint saved using "
    "--save_checkpoints_frequency. Use this to resume training or infer from a saved model.",
)
parser.add_argument(
    "--save_model_path",
    default=None,
    type=str,
    help="The path to use for saving the trained sb3 model after training is complete. Saved model can be used later "
    "to resume training. Extension will be set to .zip",
)
parser.add_argument(
    "--save_checkpoint_frequency",
    default=None,
    type=int,
    help=(
        "If set, will save checkpoints every 'frequency' environment steps. "
        "Requires a unique --experiment_name or --experiment_dir for each run. "
        "Does not need --save_model_path to be set. "
    ),
)
parser.add_argument(
    "--onnx_export_path",
    default=None,
    type=str,
    help="If included, will export onnx file after training to the path specified.",
)
parser.add_argument(
    "--timesteps",
    default=1_000_000,
    type=int,
    help="The number of environment steps to train for, default is 1_000_000. If resuming from a saved model, "
    "it will continue training for this amount of steps from the saved state without counting previously trained "
    "steps",
)
parser.add_argument(
    "--inference",
    default=False,
    action="store_true",
    help="Instead of training, it will run inference on a loaded model for --timesteps steps. "
    "Requires --resume_model_path to be set.",
)
parser.add_argument(
    "--linear_lr_schedule",
    default=False,
    action="store_true",
    help="Use a linear LR schedule for training. If set, learning rate will decrease until it reaches 0 at "
    "--timesteps"
    "value. Note: On resuming training, the schedule will reset. If disabled, constant LR will be used.",
)
parser.add_argument(
    "--viz",
    action="store_true",
    help="If set, the simulation will be displayed in a window during training. Otherwise "
    "training will run without rendering the simulation. This setting does not apply to in-editor training.",
    default=False,
)
parser.add_argument("--speedup", default=1, type=int, help="Whether to speed up the physics in the env")
parser.add_argument(
    "--n_parallel",
    default=1,
    type=int,
    help="How many instances of the environment executable to " "launch - requires --env_path to be set if > 1.",
)
parser.add_argument(
    "--policy-hidden-layers",
    default="64,64",
    type=str,
    help="Comma-separated list of hidden layer sizes applied to both policy and value networks.",
)
args, extras = parser.parse_known_args()

METRIC_PREFIX = "@METRIC "
METRIC_SOURCE = os.environ.get("CONTROLLER_METRICS")


class ControllerMetricsCallback(BaseCallback):
    """Emit episode metrics in a controller-friendly format."""

    def __init__(self, source: str) -> None:
        super().__init__()
        self._source = source or "sb3"

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        timestamp = datetime.now(timezone.utc).isoformat()
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            payload = {
                "source": self._source,
                "kind": "episode",
                "timestamp": timestamp,
                "num_timesteps": int(self.num_timesteps),
                "episode_reward": float(episode.get("r", 0.0)),
                "episode_length": int(episode.get("l", 0)),
            }
            print(f"{METRIC_PREFIX}{json.dumps(payload)}", flush=True)
        return True


def handle_onnx_export():
    # Enforce the extension of onnx and zip when saving model to avoid potential conflicts in case of same name
    # and extension used for both
    if args.onnx_export_path is not None:
        path_onnx = pathlib.Path(args.onnx_export_path).with_suffix(".onnx")
        print("Exporting onnx to: " + os.path.abspath(path_onnx))
        export_model_as_onnx(model, str(path_onnx))


def handle_model_save():
    if args.save_model_path is not None:
        zip_save_path = pathlib.Path(args.save_model_path).with_suffix(".zip")
        print("Saving model to: " + os.path.abspath(zip_save_path))
        model.save(zip_save_path)


def close_env():
    try:
        print("closing env")
        env.close()
    except Exception as e:
        print("Exception while closing env: ", e)


_cleanup_done = False


def cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    handle_onnx_export()
    handle_model_save()
    close_env()
    _cleanup_done = True


def _handle_signal(signum, _frame):
    try:
        printable = signal.Signals(signum).name
    except ValueError:
        printable = str(signum)
    print(f"Received signal {printable}. Cleaning up before exit...")
    cleanup()
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


path_checkpoint = os.path.join(args.experiment_dir, args.experiment_name + "_checkpoints")
abs_path_checkpoint = os.path.abspath(path_checkpoint)

# Prevent overwriting existing checkpoints when starting a new experiment if checkpoint saving is enabled
if args.save_checkpoint_frequency is not None and os.path.isdir(path_checkpoint):
    raise RuntimeError(
        abs_path_checkpoint + " folder already exists. "
        "Use a different --experiment_dir, or --experiment_name,"
        "or if previous checkpoints are not needed anymore, "
        "remove the folder containing the checkpoints. "
    )

if args.inference and args.resume_model_path is None:
    raise parser.error("Using --inference requires --resume_model_path to be set.")

if args.env_path is None and args.viz:
    print("Info: Using --viz without --env_path set has no effect, in-editor training will always render.")

env = StableBaselinesGodotEnv(
    env_path=args.env_path, show_window=args.viz, seed=args.seed, n_parallel=args.n_parallel, speedup=args.speedup
)
env = VecMonitor(env)


# LR schedule code snippet from:
# https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#learning-rate-schedule
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def parse_hidden_layers(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("must contain at least one integer.")
    layers: list[int] = []
    for part in parts:
        try:
            size = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid integer '{part}'") from exc
        if size <= 0:
            raise ValueError(f"values must be positive (got {size}).")
        layers.append(size)
    return layers


try:
    hidden_layers = parse_hidden_layers(args.policy_hidden_layers)
except ValueError as exc:
    parser.error(f"Invalid --policy-hidden-layers: {exc}")


if args.resume_model_path is None:
    learning_rate = 0.0003 if not args.linear_lr_schedule else linear_schedule(0.0003)
    policy_kwargs = dict(
        net_arch=[dict(pi=hidden_layers, vf=hidden_layers)]
    )
    print(f"Using policy/value hidden layers: {hidden_layers}")
    model: PPO = PPO(
        "MultiInputPolicy",
        env,
        ent_coef=0.0001,
        verbose=2,
        n_steps=32,
        tensorboard_log=args.experiment_dir,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
    )
else:
    path_zip = pathlib.Path(args.resume_model_path)
    print("Loading model: " + os.path.abspath(path_zip))
    if args.policy_hidden_layers != "64,64":
        print("Info: --policy-hidden-layers is ignored when loading a saved model.")
    model = PPO.load(path_zip, env=env, tensorboard_log=args.experiment_dir)

if args.inference:
    obs = env.reset()
    for i in range(args.timesteps):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
else:
    learn_arguments = dict(total_timesteps=args.timesteps, tb_log_name=args.experiment_name)
    callbacks: list[BaseCallback] = []
    if args.save_checkpoint_frequency:
        print("Checkpoint saving enabled. Checkpoints will be saved to: " + abs_path_checkpoint)
        checkpoint_callback = CheckpointCallback(
            save_freq=(args.save_checkpoint_frequency // env.num_envs),
            save_path=path_checkpoint,
            name_prefix=args.experiment_name,
        )
        callbacks.append(checkpoint_callback)

    if METRIC_SOURCE:
        callbacks.append(ControllerMetricsCallback(METRIC_SOURCE))

    if callbacks:
        if len(callbacks) == 1:
            learn_arguments["callback"] = callbacks[0]
        else:
            learn_arguments["callback"] = CallbackList(callbacks)
    try:
        model.learn(**learn_arguments)
    except (KeyboardInterrupt, ConnectionError, ConnectionResetError):
        print(
            """Training interrupted by user or a ConnectionError. Will save if --save_model_path was
            used and/or export if --onnx_export_path was used."""
        )
    finally:
        cleanup()
