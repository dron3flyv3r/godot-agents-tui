# Rllib Example for single and multi-agent training for GodotRL with onnx export,
# needs rllib_config.yaml in the same folder or --config_file argument specified to work.

import argparse
import hashlib
import logging
import os
import pathlib
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from numbers import Number
from typing import Any, Optional, Type

import json
import numpy as np
from gymnasium import spaces

import ray
import yaml
from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.replay_buffers.multi_agent_prioritized_replay_buffer import (
    MultiAgentPrioritizedReplayBuffer,
)
from ray.tune import Checkpoint
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls
from custom_models import register_rllib_models

METRIC_PREFIX = "@METRIC "
METRIC_SOURCE = os.environ.get("CONTROLLER_METRICS")


def _short_repr(obj: Any, limit: int = 400) -> str:
    """Return a shortened repr to avoid huge console dumps."""
    text = repr(obj)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _cast_obs_value(space: spaces.Space, value: Any) -> Any:
    """Cast raw observations to the dtype/structure expected by the space."""
    if isinstance(space, spaces.Box):
        arr = np.asarray(value, dtype=space.dtype)
        # Clip to valid bounds when they are finite to satisfy RLlib shape checks.
        low, high = space.low, space.high
        if np.any(np.isfinite(low)) or np.any(np.isfinite(high)):
            arr = np.clip(arr, low, high)
        return arr
    if isinstance(space, (spaces.MultiBinary, spaces.MultiDiscrete)):
        return np.asarray(value, dtype=space.dtype)
    if isinstance(space, spaces.Discrete):
        return int(value)
    if isinstance(space, spaces.Tuple):
        return tuple(
            _cast_obs_value(sub_space, sub_val)
            for sub_space, sub_val in zip(space.spaces, value)
        )
    if isinstance(space, spaces.Dict) and isinstance(value, dict):
        return {
            k: _cast_obs_value(space.spaces[k], value[k])
            for k in value
            if k in space.spaces
        }
    # Fallback: try to convert sequences to numpy arrays.
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def _space_is_continuous(space: spaces.Space) -> bool:
    """Return True if the action space consists purely of Box components."""
    if isinstance(space, spaces.Box):
        return True
    if isinstance(space, spaces.Tuple):
        return all(_space_is_continuous(sub_space) for sub_space in space.spaces)
    if isinstance(space, spaces.Dict):
        return all(_space_is_continuous(sub_space) for sub_space in space.spaces.values())
    return False


def _uses_episode_replay_buffer(replay_cfg: Optional[dict]) -> bool:
    """Return True if the replay buffer config stores full episodes."""
    if not replay_cfg:
        return True
    buffer_type = replay_cfg.get("type")
    if buffer_type is None:
        return True
    if isinstance(buffer_type, str):
        return "Episode" in buffer_type
    try:
        return isinstance(buffer_type, type) and issubclass(buffer_type, EpisodeReplayBuffer)
    except TypeError:
        return False


def _resolve_env_path(raw_path: Optional[str]) -> Optional[str]:
    """Resolve env_path to an existing file, adding platform suffix or searching folders."""
    if raw_path is None or raw_path == "debug":
        return raw_path

    path = os.path.abspath(os.path.expanduser(str(raw_path)))
    candidates: list[str] = [path]

    suffix_map = {
        "linux": ".x86_64",
        "linux2": ".x86_64",
        "darwin": ".app",
        "win32": ".exe",
    }
    suffix = suffix_map.get(sys.platform, "")

    # If no suffix was provided, try appending the platform suffix.
    if suffix and pathlib.Path(path).suffix != suffix:
        candidates.append(path + suffix)

    # If the path is a directory, look for likely binaries inside.
    if os.path.isdir(path):
        dir_path = pathlib.Path(path)
        candidates.append(str(dir_path / f"{dir_path.name}{suffix}"))
        candidates.extend(str(p) for p in dir_path.glob(f"*{suffix}"))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            if candidate != raw_path:
                print(
                    f"Resolved env_path from '{raw_path}' to '{candidate}'.",
                    flush=True,
                )
            return candidate

    raise FileNotFoundError(f"env_path '{raw_path}' not found. Tried: {candidates}")

def _config_sha1(config: dict[str, Any]) -> str:
    """Return a short, deterministic hash for the config for manifest/debugging."""
    try:
        blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        blob = repr(config).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _checkpoint_number_from_path(path: Optional[str]) -> Optional[int]:
    """Extract checkpoint index from a path like .../checkpoint_000123."""
    if not path:
        return None
    try:
        stem = pathlib.Path(path).name
        if stem.startswith("checkpoint_"):
            return int(stem.split("_")[-1])
    except Exception:
        return None
    return None


class NumpyObsGDRLPettingZooEnv(GDRLPettingZooEnv):
    """PettingZoo env that coerces list observations to numpy arrays for RLlib."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rl_action_spaces = {
            agent: self._normalize_action_space(space)
            for agent, space in self.action_spaces.items()
        }

    @staticmethod
    def _normalize_action_space(space: spaces.Space) -> spaces.Space:
        """Unwrap singleton Tuple/Dict spaces to expose the inner space to RLlib."""
        if isinstance(space, spaces.Tuple) and len(space.spaces) == 1:
            return space.spaces[0]
        if isinstance(space, spaces.Dict) and len(space.spaces) == 1:
            return list(space.spaces.values())[0]
        return space

    def action_space(self, agent): # type: ignore
        return self._rl_action_spaces.get(agent, super().action_space(agent))

    def _zero_action(self, space: spaces.Space) -> Any:
        """Create a zero-valued action matching the provided action space."""
        if isinstance(space, spaces.Box):
            return np.zeros(space.shape, dtype=space.dtype)
        if isinstance(space, spaces.MultiBinary):
            return np.zeros(space.shape, dtype=space.dtype)
        if isinstance(space, spaces.MultiDiscrete):
            return np.zeros_like(space.nvec, dtype=space.dtype)
        if isinstance(space, spaces.Discrete):
            return 0
        if isinstance(space, spaces.Tuple):
            return tuple(self._zero_action(s) for s in space.spaces)
        if isinstance(space, spaces.Dict):
            return {k: self._zero_action(s) for k, s in space.spaces.items()}
        try:
            sample = space.sample()
            return np.zeros_like(np.asarray(sample))
        except Exception:
            return 0

    def _format_action(self, space: spaces.Space, action: Any) -> Any:
        """Convert incoming action dict values to the expected dtype/shape."""
        if action is None:
            # RLlib may send None when an agent just finished; warn only once to avoid spam.
            if not getattr(self, "_none_action_warned", False):
                logging.warning("Received None action; substituting zero action.")
                self._none_action_warned = True
            return self._zero_action(space)
        try:
            if isinstance(space, spaces.Box):
                arr = np.asarray(action, dtype=space.dtype)
                if arr.shape != space.shape and arr.size == np.prod(space.shape):
                    arr = arr.reshape(space.shape)
                if np.any(np.isfinite(space.low)) or np.any(np.isfinite(space.high)):
                    arr = np.clip(arr, space.low, space.high)
                return arr
            if isinstance(space, spaces.MultiBinary):
                arr = np.asarray(action, dtype=space.dtype)
                if arr.shape != space.shape and arr.size == np.prod(space.shape):
                    arr = arr.reshape(space.shape)
                return arr
            if isinstance(space, spaces.MultiDiscrete):
                arr = np.asarray(action, dtype=space.dtype).flatten()
                if arr.shape != space.shape:
                    arr = arr.reshape(space.shape)
                return arr
            if isinstance(space, spaces.Discrete):
                return int(action)
            if isinstance(space, spaces.Tuple) and isinstance(action, (list, tuple)):
                return tuple(
                    self._format_action(sub_space, sub_action)
                    for sub_space, sub_action in zip(space.spaces, action)
                )
            if isinstance(space, spaces.Dict) and isinstance(action, dict):
                return {
                    k: self._format_action(space.spaces[k], action.get(k))
                    for k in space.spaces
                }
            if isinstance(space, spaces.Tuple) and not isinstance(action, (list, tuple)):
                if len(space.spaces) == 1:
                    # Wrap scalar actions back into a tuple for the Godot env.
                    return (
                        self._format_action(space.spaces[0], action),
                    )
                logging.warning(
                    "Expected tuple action for space %s but received scalar %s; "
                    "returning zero action.",
                    space,
                    _short_repr(action),
                )
                return self._zero_action(space)
        except Exception:
            logging.warning(
                "Failed to format action %s for space %s; substituting zero action.",
                _short_repr(action),
                space,
            )
            return self._zero_action(space)
        try:
            return np.asarray(action)
        except Exception:
            logging.warning(
                "Failed to convert action %s to array; substituting zero action.",
                _short_repr(action),
            )
            return self._zero_action(space)

    def _convert_obs(self, obs: dict[int, Any]) -> dict[int, Any]:
        converted: dict[int, Any] = {}
        for agent, agent_obs in obs.items():
            space = self.observation_space(agent)
            if isinstance(space, spaces.Dict) and isinstance(agent_obs, dict):
                converted[agent] = {
                    k: _cast_obs_value(space.spaces[k], v)
                    for k, v in agent_obs.items()
                    if k in space.spaces
                }
            else:
                converted[agent] = _cast_obs_value(space, agent_obs)
        return converted

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        return self._convert_obs(obs), infos

    def step(self, actions):
        godot_actions = []
        for agent_idx, agent in enumerate(self.agents):
            space = self.action_spaces[agent_idx]
            godot_actions.append(self._format_action(space, actions.get(agent)))

        godot_obs, godot_rewards, godot_dones, godot_truncations, godot_infos = (
            self.godot_env.step(godot_actions, order_ij=True)
        )
        active_agents = list(actions.keys())
        obs = {agent: godot_obs[agent] for agent in active_agents}
        rewards = {agent: godot_rewards[agent] for agent in active_agents}
        terminations = {agent: godot_dones[agent] for agent in active_agents}
        truncations = {agent: False for agent in active_agents}
        # RLlib expects a special "__all__" flag indicating whether the entire episode ended.
        # Without it, episode accounting stays at zero and rewards never show up.
        done_all = all(bool(flag) for flag in terminations.values())
        terminations["__all__"] = done_all
        truncations["__all__"] = False
        infos = {agent: godot_infos[agent] for agent in active_agents}

        return self._convert_obs(obs), rewards, terminations, truncations, infos


class TrackingParallelPettingZooEnv(ParallelPettingZooEnv):
    """Parallel PettingZoo wrapper that keeps RLlib's agent list in sync."""

    def __init__(self, env):
        super().__init__(env)
        self.agents = list(self._agent_ids)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.agents = list(obs.keys())
        return obs, info

    def step(self, action_dict):
        obs, rews, terms, truncs, infos = super().step(action_dict)
        self.agents = [agent_id for agent_id in obs.keys()]
        return obs, rews, terms, truncs, infos


class NumpyObsRayVectorGodotEnv(RayVectorGodotEnv):
    """Single-agent env wrapper that casts obs entries to numpy arrays."""

    def _convert_single_obs(self, obs: Any) -> Any:
        space = self.observation_space
        if isinstance(space, spaces.Dict) and isinstance(obs, dict):
            return {
                k: _cast_obs_value(space.spaces[k], v)
                for k, v in obs.items()
                if k in space.spaces
            }
        return _cast_obs_value(space, obs)

    def vector_reset(
        self, *, seeds: Optional[list[int]] = None, options: Optional[list[dict]] = None
    ):
        obs = super().vector_reset(seeds=seeds, options=options)
        converted_obs = [self._convert_single_obs(o) for o in obs]
        return converted_obs

    def vector_step(self, actions):
        # Ray/RL env vector_step may return either 4-tuple (obs, reward, done, info)
        # or 5-tuple (obs, reward, term, trunc, info); normalize to 4-tuple.
        res = super().vector_step(actions)
        if len(res) == 4:
            obs, reward, dones, info = res
        else:
            obs, reward, term, trunc, info = res # type: ignore
            # Combine termination and truncation into a single dones list
            if isinstance(term, (list, tuple)) and isinstance(trunc, (list, tuple)):
                dones = [bool(t or tr) for t, tr in zip(term, trunc)]
            else:
                # fallback scalar or unexpected types
                try:
                    dones = [bool(term or trunc) for _ in reward]
                except Exception:
                    dones = [bool(term or trunc)]
        converted_obs = [self._convert_single_obs(o) for o in obs]
        return converted_obs, reward, dones, info


class ControllerMetricsCallback(tune.Callback):
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
                return float(val) # type: ignore
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
                return int(val) # type: ignore
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
        
        # with open("/home/kasper/GameProjects/agents/debug_result.json", "w") as f:
        #     json.dump(_sanitize(payload), f, indent=2)

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
    parser.add_argument(
        "--debug_panel",
        action="store_true",
        help="Print a one-step debug panel with sampled observations/actions before training.",
    )
    args, extras = parser.parse_known_args()

    resume_target = args.resume or args.restore
    if args.resume and args.restore:
        print(
            "Both --resume and --restore provided; preferring --resume directory.",
            flush=True,
        )
    args.restore = os.path.abspath(resume_target) if resume_target else None

    signal.signal(signal.SIGINT, _handle_interrupt_signal)
    signal.signal(signal.SIGTERM, _handle_interrupt_signal)
    register_rllib_models()

    # Get config from file
    with open(args.config_file) as f:
        exp = yaml.safe_load(f)

    config_dict: dict[str, Any] = exp.get("config") or {}

    # RLlib 2.8+ renamed num_envs_per_worker -> num_envs_per_env_runner.
    legacy_num_envs = config_dict.pop("num_envs_per_worker", None)
    if legacy_num_envs is not None and "num_envs_per_env_runner" not in config_dict:
        config_dict["num_envs_per_env_runner"] = legacy_num_envs
        print(
            "num_envs_per_worker is deprecated; using num_envs_per_env_runner instead.",
            flush=True,
        )

    # Validate and normalize env_path early so Ray workers don't fail later.
    env_cfg = config_dict.get("env_config") or {}
    if "env_path" in env_cfg:
        try:
            env_cfg["env_path"] = _resolve_env_path(env_cfg["env_path"])
        except FileNotFoundError as exc:
            parser.error(str(exc))
        config_dict["env_config"] = env_cfg

    def _force_classic_api_stack(reason: str) -> bool:
        """Disable RLModule/env-runner stack if currently enabled."""
        changed = False
        if config_dict.get("enable_rl_module_and_learner", True):
            config_dict["enable_rl_module_and_learner"] = False
            changed = True
        if config_dict.get("enable_env_runner_and_connector_v2", True):
            config_dict["enable_env_runner_and_connector_v2"] = False
            changed = True
        if changed:
            print(reason, flush=True)
        return changed

    model_cfg = config_dict.get("model") or {}
    custom_model = model_cfg.get("custom_model")

    if custom_model == "tui_lstm":
        converted = dict(model_cfg)
        custom_cfg = model_cfg.get("custom_model_config") or {}
        config_dict["_disable_action_flattening"] = True
        converted["use_lstm"] = True
        if "lstm_cell_size" not in converted and custom_cfg.get("hidden_size") is not None:
            converted["lstm_cell_size"] = int(custom_cfg["hidden_size"])
        include_prev = custom_cfg.get("include_prev_actions")
        if "lstm_use_prev_action" not in converted:
            converted["lstm_use_prev_action"] = True if include_prev is None else bool(include_prev)
        converted.setdefault("lstm_use_prev_reward", False)
        if "fcnet_hiddens" not in converted and custom_cfg.get("fcnet_hiddens") is not None:
            converted["fcnet_hiddens"] = custom_cfg["fcnet_hiddens"]
        converted.pop("custom_model", None)
        converted.pop("custom_model_config", None)
        config_dict["model"] = converted
        model_cfg = converted
        custom_model = model_cfg.get("custom_model")
        print(
            "Converted tui_lstm custom model config to RLlib built-in LSTM settings.",
            flush=True,
        )

    algorithm_name = str(exp.get("algorithm") or "").upper()
    sac_like_algorithms = {"SAC"}
    if custom_model and algorithm_name in sac_like_algorithms:
        print(
            f"Algorithm {algorithm_name} manages its own policy/Q models; "
            f"ignoring custom_model '{custom_model}'.",
            flush=True,
        )
        model_cfg = dict(model_cfg)
        model_cfg.pop("custom_model", None)
        model_cfg.pop("custom_model_config", None)
        config_dict["model"] = model_cfg
        custom_model = None
    if algorithm_name in sac_like_algorithms and config_dict.get("lr") is not None:
        print("SAC uses actor/critic specific learning rates; setting global lr to None.", flush=True)
        config_dict["lr"] = None

    if custom_model:
        _force_classic_api_stack(
            "Detected custom RLlib model "
            f"'{custom_model}'. Disabling the RLModule/Learner and "
            "EnvRunner v2 stack so ModelV2 custom models remain compatible."
        )

    _force_classic_api_stack(
        "Using RLlib classic API stack for Godot multi-agent compatibility."
    )

    use_old_stack = not config_dict.get("enable_rl_module_and_learner", True)
    if (
        algorithm_name in sac_like_algorithms
        and use_old_stack
        and _uses_episode_replay_buffer(config_dict.get("replay_buffer_config"))
    ):
        defaults = {
            "type": MultiAgentPrioritizedReplayBuffer,
            "capacity": int(1e6),
            "storage_unit": "timesteps",
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
        }
        replay_config = {**defaults, **(config_dict.get("replay_buffer_config") or {})}
        replay_config["type"] = MultiAgentPrioritizedReplayBuffer
        config_dict["replay_buffer_config"] = replay_config
        exp["config"] = config_dict
        print(
            "Configured SAC to use MultiAgentPrioritizedReplayBuffer for old API "
            "compatibility.",
            flush=True,
        )
    exp["config"] = config_dict

    is_multiagent = exp["env_is_multiagent"]

    # Register env
    env_name = "godot"
    env_wrapper = None

    def env_creator(env_config):
        index = (
            env_config.worker_index * exp["config"].get("num_envs_per_env_runner", 1)
            + env_config.vector_index
        )
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        if is_multiagent:
            return TrackingParallelPettingZooEnv(
                NumpyObsGDRLPettingZooEnv(
                    config=env_config, port=port, seed=seed, show_window=True
                )
            )
        else:
            return NumpyObsRayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)

    policy_names = None
    num_envs = None
    tmp_env = None

    if (
        is_multiagent
    ):  # Make temp env to get info needed for multi-agent training config
        print("Starting a temporary multi-agent env to get the policy names")
        tmp_env = NumpyObsGDRLPettingZooEnv(
            config=exp["config"]["env_config"], show_window=False
        )
        if algorithm_name in sac_like_algorithms:
            invalid_agents = [
                agent
                for agent in tmp_env.agents
                if not _space_is_continuous(tmp_env.action_space(agent))
            ]
            if invalid_agents:
                tmp_env.close()
                raise ValueError(
                    "SAC requires continuous Box actions, but the following agents "
                    f"use discrete or structured spaces: {invalid_agents}. "
                    "Choose an algorithm that supports discrete actions (e.g. PPO/DQN) "
                    "or adjust the Godot action layout."
                )
        policy_names = tmp_env.agent_policy_names
        print(
            "Policy names for each Agent (AIController) set in the Godot Environment",
            policy_names,
        )
        if args.debug_panel:
            try:
                print("\n=== RLlib Debug Panel (multi-agent) ===")
                print(f"env_path: {exp['config']['env_config'].get('env_path')}")
                obs0, info0 = tmp_env.reset()
                print(f"agents: {tmp_env.agents}")
                print(f"policy_names: {policy_names}")
                print("action_spaces per agent:")
                for idx, agent in enumerate(tmp_env.agents):
                    print(f"  {agent} (policy {policy_names[idx]}): {tmp_env.action_spaces[idx]}")
                print(f"reset obs: {_short_repr(obs0)}")
                actions = {}
                for idx, agent in enumerate(tmp_env.agents):
                    space = tmp_env.action_spaces[idx]
                    raw_action = space.sample()
                    actions[agent] = tmp_env._format_action(space, raw_action)
                print(f"sampled actions (per agent): {_short_repr(actions)}")
                step_obs, step_rew, step_term, step_trunc, step_info = tmp_env.step(actions)
                print(f"step obs: {_short_repr(step_obs)}")
                print(f"step rewards: {_short_repr(step_rew)}")
                print(f"step terminations: {_short_repr(step_term)}")
                print(f"step truncations: {_short_repr(step_trunc)}")
                print(f"step infos: {_short_repr(step_info)}\n")
            except Exception as exc:  # pragma: no cover - debug helper
                print(f"Debug panel failed: {exc}")
    else:  # Make temp env to get info needed for setting num_workers training config
        print(
            "Starting a temporary env to get the number of envs and auto-set the num_envs_per_env_runner config value"
        )
        tmp_env = GodotEnv(
            env_path=exp["config"]["env_config"]["env_path"], show_window=False
        )
        if algorithm_name in sac_like_algorithms and not _space_is_continuous(tmp_env.action_space):
            tmp_env.close()
            raise ValueError(
                "SAC requires continuous Box actions, but the configured Godot env "
                "uses discrete controls. Choose an algorithm with discrete support "
                "or adjust the Godot action layout."
            )
        num_envs = tmp_env.num_envs
        if args.debug_panel:
            try:
                print("\n=== RLlib Debug Panel (single-agent) ===")
                print(f"env_path: {exp['config']['env_config'].get('env_path')}")
                print(f"action_space: {tmp_env.action_space}")
                obs0, info0 = tmp_env.reset()
                print(f"reset obs: {_short_repr(obs0)}")
                action = tmp_env.action_space.sample()
                print(f"sampled action: {_short_repr(action)}")
                step_obs, step_rew, step_done, step_trunc, step_info = tmp_env.step(action)
                print(f"step obs: {_short_repr(step_obs)}")
                print(f"step rewards: {_short_repr(step_rew)}")
                print(f"step dones: {_short_repr(step_done)}")
                print(f"step truncations: {_short_repr(step_trunc)}")
                print(f"step infos: {_short_repr(step_info)}\n")
            except Exception as exc:  # pragma: no cover - debug helper
                print(f"Debug panel failed: {exc}")

    tmp_env.close()

    def policy_mapping_fn(agent_id: int, episode, worker, **kwargs) -> str:
        if policy_names is None:
            raise ValueError("policy_names is not set for multi-agent policy mapping.")
        return policy_names[agent_id]

    ray_temp_dir = os.path.join(os.path.expanduser("~"), ".ray_tmp")
    os.makedirs(ray_temp_dir, exist_ok=True)
    ray.init(_temp_dir=ray_temp_dir)

    if is_multiagent:
        if policy_names is None:
            raise ValueError("policy_names is not set for multi-agent configuration.")
        exp["config"]["multiagent"] = {
            "policies": {policy_name: PolicySpec() for policy_name in policy_names},
            "policy_mapping_fn": policy_mapping_fn,
        }
    else:
        exp["config"]["num_envs_per_env_runner"] = num_envs

    run_callbacks: list[tune.Callback] = []
    if METRIC_SOURCE:
        run_callbacks.append(ControllerMetricsCallback(METRIC_SOURCE))

    checkpoint_frequency = int(exp.get("checkpoint_frequency") or 0)
    stop_config = exp.get("stop") or {}
    algorithm_name = str(exp.get("algorithm") or "UNKNOWN")

    resume_checkpoint = args.restore

    def _locate_checkpoint_from_tune_dir(path: pathlib.Path) -> Optional[str]:
        marker = path / "tuner.pkl"
        legacy = path / "tune.pkl"
        if marker.is_file() or legacy.is_file():
            checkpoint_dirs = list(path.rglob("checkpoint_*"))
            checkpoint_dirs = [p for p in checkpoint_dirs if p.is_dir()]
            if checkpoint_dirs:
                checkpoint_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                latest = checkpoint_dirs[0]
                print(
                    f"Detected Ray Tune run directory; resuming from latest checkpoint: {latest}"
                )
                return str(latest)
            print("Tune run directory found but no checkpoints were located.")
        return None

    if resume_checkpoint:
        resume_path = pathlib.Path(resume_checkpoint)
        if not resume_path.exists():
            parser.error(f"Resume checkpoint not found: {resume_path}")
        if resume_path.is_dir() and not (resume_path / "algorithm_state.pkl").is_file():
            found = _locate_checkpoint_from_tune_dir(resume_path)
            if found:
                resume_checkpoint = found
                resume_path = pathlib.Path(resume_checkpoint)
        algo_state = resume_path / "algorithm_state.pkl"
        rllib_meta = resume_path / "rllib_checkpoint.json"
        if not resume_path.is_dir():
            parser.error(f"Resume path must be a checkpoint directory: {resume_path}")
        if not algo_state.is_file() and not rllib_meta.is_file():
            parser.error(
                f"Resume checkpoint missing algorithm_state.pkl (or rllib_checkpoint.json): {resume_path}"
            )

    resume_env_path = None
    if resume_checkpoint:
        manifest_path = pathlib.Path(resume_checkpoint).parent / "run_manifest.json"
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r") as f:
                    resume_env_path = (json.load(f) or {}).get("env_path")
            except Exception:
                resume_env_path = None

    env_cfg = exp["config"].get("env_config") or {}
    env_path = env_cfg.get("env_path")
    if resume_env_path and env_path and env_path != resume_env_path:
        print(
            f"Warning: env_path changed from resume manifest ({resume_env_path}) to {env_path}. Continuing.",
            flush=True,
        )

    env_label = pathlib.Path(env_path or "env").stem or "env"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    token = uuid.uuid4().hex[:5]
    experiment_id = f"{algorithm_name}_{timestamp}"
    trial_id = f"{token}_manual"
    trial_dir_name = f"{algorithm_name}_{env_label}_{token}_{timestamp}"
    run_dir = pathlib.Path(args.experiment_dir).expanduser().resolve() / experiment_id
    trial_dir = run_dir / trial_dir_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    experiment_tag = trial_dir_name
    start_checkpoint_index = (_checkpoint_number_from_path(resume_checkpoint) or -1) + 1

    class _ManualTrial:
        """Lightweight trial-like object for Tune-style callbacks."""

        def __init__(self, trial_id: str, tag: str, local_path: str, config: dict[str, Any]):
            self.trial_id = trial_id
            self.experiment_tag = tag
            self.local_path = local_path
            self.logdir = local_path
            self.config = config

        def __str__(self) -> str:
            return self.experiment_tag

    trial = _ManualTrial(trial_id, experiment_tag, str(trial_dir), exp["config"])
    trials = [trial]

    algo_cls: Type[Algorithm] = get_trainable_cls(algorithm_name)

    def logger_creator(config):
        logdir = str(trial_dir)
        os.makedirs(logdir, exist_ok=True)
        return UnifiedLogger(config, logdir, loggers=None)

    manifest = {
        "algorithm": algorithm_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "experiment_tag": experiment_tag,
        "trial_id": trial_id,
        "trial_dir": str(trial_dir),
        "env_path": env_path,
        "checkpoint_frequency": checkpoint_frequency,
        "stop": stop_config,
        "resume_from": resume_checkpoint,
        "config_sha1": _config_sha1(exp["config"]),
        "checkpoint_index_offset": start_checkpoint_index,
    }
    try:
        with open(trial_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
    except Exception as exc:
        print(f"Warning: failed to write run_manifest.json: {exc}")

    algo = None
    latest_checkpoint_path: Optional[pathlib.Path] = None
    next_checkpoint_index = start_checkpoint_index

    def _save_checkpoint(algo_obj: Algorithm, index: int) -> pathlib.Path:
        path = trial_dir / f"checkpoint_{index:06d}"
        path.mkdir(parents=True, exist_ok=True)
        algo_obj.save_checkpoint(str(path))
        return path

    def _should_stop(stop_cfg: dict[str, Any], result: dict[str, Any], start_time: float) -> bool:
        for key, target in stop_cfg.items():
            if target is None:
                continue
            current = None
            if key == "time_total_s":
                current = result.get("time_total_s")
                if current is None:
                    current = time.perf_counter() - start_time
            else:
                current = result.get(key)
            try:
                if current is not None and float(current) >= float(target):
                    return True
            except Exception:
                continue
        return False

    start_perf = time.perf_counter()

    def _callback_setup():
        for cb in run_callbacks:
            try:
                cb.setup(stop=stop_config, num_samples=1, total_num_samples=1) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback setup failed: {exc}")

    def _callback_on_trial_start(iteration: int):
        for cb in run_callbacks:
            try:
                cb.on_trial_start(iteration=iteration, trials=trials, trial=trial) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_trial_start failed: {exc}")

    def _callback_on_trial_restore(iteration: int):
        for cb in run_callbacks:
            try:
                cb.on_trial_restore(iteration=iteration, trials=trials, trial=trial) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_trial_restore failed: {exc}")

    def _callback_on_step_begin(iteration: int):
        for cb in run_callbacks:
            try:
                cb.on_step_begin(iteration=iteration, trials=trials) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_step_begin failed: {exc}")

    def _callback_on_step_end(iteration: int):
        for cb in run_callbacks:
            try:
                cb.on_step_end(iteration=iteration, trials=trials) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_step_end failed: {exc}")

    def _callback_on_trial_result(iteration: int, result: dict[str, Any]):
        for cb in run_callbacks:
            try:
                cb.on_trial_result(iteration=iteration, trials=trials, trial=trial, result=result) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_trial_result failed: {exc}")

    def _callback_on_checkpoint(iteration: int, checkpoint_path: pathlib.Path):
        if not run_callbacks:
            return
        checkpoint_obj = Checkpoint.from_directory(str(checkpoint_path))
        for cb in run_callbacks:
            try:
                cb.on_checkpoint(
                    iteration=iteration, trials=trials, trial=trial, checkpoint=checkpoint_obj # type: ignore
                )
                cb.on_trial_save(iteration=iteration, trials=trials, trial=trial) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_checkpoint/on_trial_save failed: {exc}")

    def _callback_on_trial_complete(iteration: int):
        for cb in run_callbacks:
            try:
                cb.on_trial_complete(iteration=iteration, trials=trials, trial=trial) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_trial_complete failed: {exc}")
        for cb in run_callbacks:
            try:
                cb.on_experiment_end(trials=trials) # type: ignore
            except Exception as exc:
                logging.warning(f"Callback on_experiment_end failed: {exc}")

    _callback_setup()
    try:
        algo = algo_cls(config=exp["config"], logger_creator=logger_creator) # type: ignore
        if resume_checkpoint:
            print(f"Restoring from checkpoint: {resume_checkpoint}", flush=True)
            algo.restore(resume_checkpoint)
            _callback_on_trial_restore(iteration=0)

        _callback_on_trial_start(iteration=0)

        loop_iter = 0
        while True:
            _callback_on_step_begin(iteration=loop_iter)
            iter_start = time.perf_counter()
            result = algo.train()
            now = time.perf_counter()
            result.setdefault("date", datetime.now(timezone.utc).isoformat())
            result.setdefault("time_total_s", now - start_perf)
            result.setdefault("time_this_iter_s", now - iter_start)
            result["experiment_id"] = experiment_id
            result["experiment_tag"] = experiment_tag
            result["trial_id"] = trial_id

            _callback_on_trial_result(iteration=loop_iter, result=result)

            training_iteration = int(result.get("training_iteration") or 0)
            if checkpoint_frequency > 0 and training_iteration > 0:
                if training_iteration % checkpoint_frequency == 0:
                    latest_checkpoint_path = _save_checkpoint(algo, next_checkpoint_index)
                    next_checkpoint_index += 1
                    print(f"Saved checkpoint: {latest_checkpoint_path}", flush=True)
                    _callback_on_checkpoint(iteration=loop_iter, checkpoint_path=latest_checkpoint_path)

            if _should_stop(stop_config, result, start_perf):
                print("Stopping criteria reached. Ending training.", flush=True)
                _callback_on_step_end(iteration=loop_iter)
                break

            _callback_on_step_end(iteration=loop_iter)
            loop_iter += 1

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Attempting to save a checkpoint...", flush=True)
        if algo:
            try:
                latest_checkpoint_path = _save_checkpoint(algo, next_checkpoint_index)
                print(f"Checkpoint saved after interrupt: {latest_checkpoint_path}", flush=True)
            except Exception as exc:
                print(f"Could not save checkpoint after interrupt: {exc}", flush=True)
            if latest_checkpoint_path:
                _callback_on_checkpoint(iteration=0, checkpoint_path=latest_checkpoint_path)
    finally:
        _callback_on_trial_complete(iteration=loop_iter if 'loop_iter' in locals() else 0) # type: ignore
        if algo:
            try:
                algo.stop()
            except Exception:
                pass
        ray.shutdown()

    if latest_checkpoint_path:
        print(
            "Resume training by selecting this checkpoint path in the controller or by"
            f" passing --resume {latest_checkpoint_path}",
            flush=True,
        )
