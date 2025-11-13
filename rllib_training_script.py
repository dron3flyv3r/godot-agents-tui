# Rllib Example for single and multi-agent training for GodotRL with onnx export,
# needs rllib_config.yaml in the same folder or --config_file argument specified to work.

import argparse
import glob
import os
import pathlib
import signal
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Optional

import json

import ray
import yaml
from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
from ray import train, tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from custom_models import register_rllib_models

METRIC_PREFIX = "@METRIC "
METRIC_SOURCE = os.environ.get("CONTROLLER_METRICS")

try:
    from ray.train.callbacks import Callback as TrainCallbackBase
except ImportError:  # pragma: no cover - fallback for older ray versions
    try:
        from ray.train.callbacks.callback import Callback as TrainCallbackBase
    except ImportError:  # pragma: no cover - minimal stub
        class TrainCallbackBase:  # type: ignore[override]
            pass


class ControllerMetricsCallback(TrainCallbackBase, tune.Callback):
    """Emit iteration metrics for consumption by the controller."""

    def __init__(self, source: str, checkpoint_freq: int = -1) -> None:
        super().__init__()
        self._source = source or "rllib"
        self._checkpoint_freq = checkpoint_freq

    def on_trial_result(self, iteration, trials, trial, result, **info):  # type: ignore[override]  # noqa: D401
        self._emit_result(result)

    def handle_result(self, result: dict, **info):  # type: ignore[override]
        self._emit_result(result)

    def _emit_result(self, result: dict[str, Any]) -> None:
        if not self._source:
            return

        def _to_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            if isinstance(val, Number):
                return float(val)
            if isinstance(val, str):
                try:
                    return float(val)
                except ValueError:
                    return None
            if hasattr(val, "item"):
                try:
                    return float(val.item())  # type: ignore[call-arg]
                except Exception:
                    return None
            return None

        def _to_int(val: Any) -> Optional[int]:
            if val is None:
                return None
            if isinstance(val, Number):
                return int(val)
            if isinstance(val, str):
                try:
                    return int(float(val))
                except ValueError:
                    return None
            if hasattr(val, "item"):
                try:
                    return int(val.item())  # type: ignore[call-arg]
                except Exception:
                    return None
            return None

        def _numeric_map(data: Any) -> dict[str, float]:
            if not isinstance(data, dict):
                return {}
            out: dict[str, float] = {}
            for key, value in data.items():
                if not isinstance(key, str):
                    key = str(key)
                num = _to_float(value)
                if num is not None:
                    out[key] = num
            return out

        def _flatten_numeric(prefix: str, data: Any, out: dict[str, float]) -> None:
            if isinstance(data, dict):
                for sub_key, sub_val in data.items():
                    key = f"{prefix}.{sub_key}" if prefix else str(sub_key)
                    _flatten_numeric(key, sub_val, out)
            else:
                num = _to_float(data)
                if num is not None:
                    out[str(prefix)] = num

        env_runners = result.get("env_runners") or {}
        sampler_results = result.get("sampler_results") or {}
        perf = result.get("perf") or {}
        info = result.get("info") or {}
        learner_raw = info.get("learner") or {}
        
        # Get custom metrics from multiple sources
        custom_metrics = result.get("custom_metrics") or {}
        env_runners_custom = env_runners.get("custom_metrics") or {}
        # Merge env_runners custom metrics into custom_metrics
        custom_metrics = {**custom_metrics, **env_runners_custom}

        policy_reward_mean = (
            result.get("policy_reward_mean")
            or env_runners.get("policy_reward_mean")
            or sampler_results.get("policy_reward_mean")
            or {}
        )
        policy_reward_min = (
            result.get("policy_reward_min")
            or env_runners.get("policy_reward_min")
            or sampler_results.get("policy_reward_min")
            or {}
        )
        policy_reward_max = (
            result.get("policy_reward_max")
            or env_runners.get("policy_reward_max")
            or sampler_results.get("policy_reward_max")
            or {}
        )
        policy_episode_len_mean = (
            env_runners.get("policy_episode_len_mean")
            or sampler_results.get("policy_episode_len_mean")
            or {}
        )
        policy_completed_episodes = (
            env_runners.get("policy_completed_episodes")
            or sampler_results.get("policy_completed_episodes")
            or {}
        )
        
        # Get custom metrics per policy if available
        policy_custom_metrics = env_runners.get("policy_custom_metrics") or {}

        env_steps_this_iter = (
            _to_int(env_runners.get("num_env_steps_sampled_this_iter"))
            or _to_int(sampler_results.get("num_env_steps_sampled"))
            or _to_int(result.get("num_env_steps_sampled_this_iter"))
        )
        time_this_iter_s = _to_float(result.get("time_this_iter_s"))
        env_throughput = None
        if env_steps_this_iter is not None and time_this_iter_s:
            if time_this_iter_s > 0:
                env_throughput = env_steps_this_iter / time_this_iter_s

        policies: dict[str, dict[str, Any]] = {}
        policy_names = set()
        policy_names.update(policy_reward_mean.keys())
        policy_names.update(policy_reward_min.keys())
        policy_names.update(policy_reward_max.keys())
        policy_names.update(policy_episode_len_mean.keys())
        policy_names.update(policy_completed_episodes.keys())
        policy_names.update(learner_raw.keys())
        policy_names.update(policy_custom_metrics.keys())

        for policy_id in sorted(str(name) for name in policy_names):
            entry: dict[str, Any] = {}
            entry["reward_mean"] = _to_float(policy_reward_mean.get(policy_id))
            entry["reward_min"] = _to_float(policy_reward_min.get(policy_id))
            entry["reward_max"] = _to_float(policy_reward_max.get(policy_id))
            entry["episode_len_mean"] = _to_float(
                policy_episode_len_mean.get(policy_id)
            )
            entry["completed_episodes"] = _to_int(
                policy_completed_episodes.get(policy_id)
            )

            learner_entry: dict[str, float] = {}
            raw = learner_raw.get(policy_id) or learner_raw.get(str(policy_id))
            if isinstance(raw, dict):
                # Capture top-level numeric stats.
                top = _numeric_map(raw)
                learner_entry.update(top)
                # Include nested learner_stats or stats dictionaries.
                for nested_key in ("learner_stats", "stats"):
                    nested = raw.get(nested_key)
                    if isinstance(nested, dict):
                        nested_stats = _numeric_map(nested)
                        learner_entry.update(nested_stats)
                    elif nested is not None:
                        _flatten_numeric(
                            nested_key,
                            nested,
                            learner_entry,
                        )
            entry["learner"] = learner_entry
            
            # Add custom metrics for this policy if available
            policy_custom = policy_custom_metrics.get(policy_id) or {}
            if policy_custom:
                entry["custom_metrics"] = _numeric_map(policy_custom)

            policies[policy_id] = entry

        payload: dict[str, Any] = {
            "source": self._source,
            "kind": "iteration",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_iteration": _to_int(result.get("training_iteration")),
            "timesteps_total": _to_int(result.get("timesteps_total")),
            "time_this_iter_s": time_this_iter_s,
            "time_total_s": _to_float(result.get("time_total_s")),
            "episodes_total": _to_int(result.get("episodes_total")),
            "episodes_this_iter": _to_int(result.get("episodes_this_iter")),
            "episode_reward_mean": _to_float(result.get("episode_reward_mean")),
            "episode_reward_min": _to_float(result.get("episode_reward_min")),
            "episode_reward_max": _to_float(result.get("episode_reward_max")),
            "episode_len_mean": _to_float(result.get("episode_len_mean")),
            "num_env_steps_sampled": _to_int(result.get("num_env_steps_sampled")),
            "num_env_steps_trained": _to_int(result.get("num_env_steps_trained")),
            "num_agent_steps_sampled": _to_int(result.get("num_agent_steps_sampled")),
            "num_agent_steps_trained": _to_int(result.get("num_agent_steps_trained")),
            "env_steps_this_iter": env_steps_this_iter,
            "env_throughput": _to_float(
                perf.get("mean_throughput") 
                or result.get("num_env_steps_sampled_throughput_per_sec")
                or env_throughput
            ),
            "num_workers": _to_int(result.get("num_workers") or result.get("num_healthy_workers")),
            "custom_metrics": {
                key: value
                for key, value in _numeric_map(custom_metrics).items()
            },
            "policies": policies,
        }

        def _sanitize(obj: Any) -> Any:
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            if isinstance(obj, dict):
                sanitized: dict[str, Any] = {}
                for key, value in obj.items():
                    sanitized[str(key)] = _sanitize(value)
                return sanitized
            if isinstance(obj, (list, tuple)):
                return [_sanitize(item) for item in obj]
            numeric = _to_float(obj)
            if numeric is not None:
                return numeric
            return str(obj)

        print(f"{METRIC_PREFIX}{json.dumps(_sanitize(payload))}", flush=True)


_SIGNAL_RECEIVED = False


def _handle_interrupt_signal(signum, _frame) -> None:
    global _SIGNAL_RECEIVED
    if _SIGNAL_RECEIVED:
        return
    _SIGNAL_RECEIVED = True
    try:
        name = signal.Signals(signum).name
    except ValueError:
        name = str(signum)
    print(
        f"\nReceived signal {name}. Attempting graceful RLlib shutdown...",
        flush=True,
    )
    raise KeyboardInterrupt()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_file",
        default="rllib_config.yaml",
        type=str,
        help="The yaml config file",
    )
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="the location of a checkpoint to restore from",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="directory containing tuner.pkl for resuming a previous run",
    )
    parser.add_argument(
        "--experiment_dir",
        default="logs/rllib",
        type=str,
        help="The name of the the experiment directory, used to store logs.",
    )
    args, extras = parser.parse_known_args()

    resume_target = args.resume or args.restore
    if args.resume and args.restore:
        print(
            "Both --resume and --restore provided; preferring --resume directory.",
            flush=True,
        )
    if resume_target:
        resume_dir = os.path.abspath(resume_target)
        tuner_state = os.path.join(resume_dir, "tuner.pkl")
        if not os.path.isfile(tuner_state):
            parser.error(
                f"Resume directory must contain tuner.pkl, but none found at {tuner_state}."
            )
        args.restore = resume_dir
    else:
        args.restore = None

    signal.signal(signal.SIGINT, _handle_interrupt_signal)
    signal.signal(signal.SIGTERM, _handle_interrupt_signal)
    register_rllib_models()

    # Get config from file
    with open(args.config_file) as f:
        exp = yaml.safe_load(f)

    is_multiagent = exp["env_is_multiagent"]

    # Register env
    env_name = "godot"
    env_wrapper = None

    def env_creator(env_config):
        index = (
            env_config.worker_index * exp["config"]["num_envs_per_worker"]
            + env_config.vector_index
        )
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        if is_multiagent:
            return ParallelPettingZooEnv(
                GDRLPettingZooEnv(
                    config=env_config, port=port, seed=seed, show_window=True
                )
            )
        else:
            return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)

    policy_names = None
    num_envs = None
    tmp_env = None

    if (
        is_multiagent
    ):  # Make temp env to get info needed for multi-agent training config
        print("Starting a temporary multi-agent env to get the policy names")
        tmp_env = GDRLPettingZooEnv(
            config=exp["config"]["env_config"], show_window=False
        )
        policy_names = tmp_env.agent_policy_names
        print(
            "Policy names for each Agent (AIController) set in the Godot Environment",
            policy_names,
        )
    else:  # Make temp env to get info needed for setting num_workers training config
        print(
            "Starting a temporary env to get the number of envs and auto-set the num_envs_per_worker config value"
        )
        tmp_env = GodotEnv(
            env_path=exp["config"]["env_config"]["env_path"], show_window=False
        )
        num_envs = tmp_env.num_envs

    tmp_env.close()

    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
        return policy_names[agent_id]

    ray_temp_dir = os.path.join(os.path.expanduser("~"), ".ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    ray.init(_temp_dir=ray_temp_dir)

    if is_multiagent:
        exp["config"]["multiagent"] = {
            "policies": {policy_name: PolicySpec() for policy_name in policy_names},
            "policy_mapping_fn": policy_mapping_fn,
        }
    else:
        exp["config"]["num_envs_per_worker"] = num_envs

    tuner = None
    run_callbacks = []
    if METRIC_SOURCE:
        run_callbacks.append(ControllerMetricsCallback(METRIC_SOURCE))

    if not args.restore:
        run_config_kwargs = dict(
            storage_path=os.path.abspath(args.experiment_dir),
            stop=exp["stop"],
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=exp["checkpoint_frequency"]
            ),
        )
        if run_callbacks:
            run_config_kwargs["callbacks"] = run_callbacks

        tuner = tune.Tuner(
            trainable=exp["algorithm"],
            param_space=exp["config"],
            run_config=train.RunConfig(**run_config_kwargs),
        )
    else:
        tuner = tune.Tuner.restore(
            trainable=exp["algorithm"],
            path=args.restore,
            resume_unfinished=True,
        )
        if run_callbacks:
            try:  # pragma: no cover - relies on Ray internals
                tuner._local_tuner._run_config.callbacks = run_callbacks
            except AttributeError:  # pragma: no cover
                try:
                    tuner.get_internal_tuner()._run_config.callbacks = run_callbacks
                except Exception:
                    print("Warning: unable to attach controller metrics callback after restore.")

    result = None
    checkpoint = None
    latest_checkpoint_path: Optional[pathlib.Path] = None
    export_after_training = False

    try:
        try:
            result = tuner.fit()
            checkpoint = result.get_best_result().checkpoint
        except KeyboardInterrupt:
            export_after_training = False
            print(
                "\n\nTraining interrupted by user. Searching for the latest checkpoint..."
            )
            try:
                experiment_path = os.path.abspath(args.experiment_dir)
                trial_dirs = glob.glob(os.path.join(experiment_path, "*/"))
                if trial_dirs:
                    trial_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    latest_trial = pathlib.Path(trial_dirs[0])
                    checkpoint_dirs = glob.glob(
                        str(latest_trial / "checkpoint_*")
                    )
                    if checkpoint_dirs:
                        checkpoint_dirs.sort(
                            key=lambda x: os.path.getmtime(x), reverse=True
                        )
                        latest_checkpoint_path = pathlib.Path(checkpoint_dirs[0])
                        print(f"Latest checkpoint located at: {latest_checkpoint_path}")
                    else:
                        print("No checkpoints found in the most recent trial directory.")
                else:
                    print(
                        "No trial directories found. Check that checkpoints were being saved."
                    )
            except Exception as exc:
                print(f"Could not resolve latest checkpoint after interruption: {exc}")
    finally:
        ray.shutdown()

    if latest_checkpoint_path:
        print(
            "Resume training by selecting this checkpoint path in the controller or by"
            f" passing --restore {latest_checkpoint_path}"
        )

    if not export_after_training:
        print("Skip ONNX export")
    else:
        if checkpoint is None and result is not None:
            try:
                checkpoint = result.get_best_result().checkpoint
            except Exception:
                checkpoint = None

        if checkpoint and result:
            result_path = result.get_best_result().path
            ppo = Algorithm.from_checkpoint(checkpoint)
            if is_multiagent and policy_names:
                for policy_name in set(policy_names):
                    ppo.get_policy(policy_name).export_model(
                        f"{result_path}/onnx_export/{policy_name}_onnx", onnx=11
                    )
                    print(
                        f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/{policy_name}_onnx').resolve()}"
                    )
            else:
                ppo.get_policy().export_model(
                    f"{result_path}/onnx_export/single_agent_policy_onnx", onnx=11
                )
                print(
                    f"Saving onnx policy to {pathlib.Path(f'{result_path}/onnx_export/single_agent_policy_onnx').resolve()}"
                )
