#!/usr/bin/env python3
"""
Workspace-only interface runner for RLlib checkpoints.

Loads a raw RLlib checkpoint (or parent directory with checkpoint numbers),
restores the algorithm in inference-only mode (no worker env spin-up), and
streams interface events/actions for the Rust TUI while stepping a Godot
environment from this process.
"""

from __future__ import annotations

import argparse
import json
import pickle
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, override

try:
    from godot_rl.core.godot_env import GodotEnv
    from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
except ImportError as exc:  # pragma: no cover - dependency check
    print(
        "This script requires the 'godot-rl' package. "
        "Install it in your workspace Python environment.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

EVENT_PREFIX = "@INTERFACE_EVENT "
ACTION_PREFIX = "@INTERFACE_ACTION "

try:
    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
    from ray import tune
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except Exception:
    AlgorithmConfig = None  # type: ignore
    tune = None  # type: ignore
    PolicySpec = None  # type: ignore
    MultiAgentEnv = None  # type: ignore


def _json_default(value: Any) -> Any:
    """Convert numpy/scalar containers into JSON-friendly types."""
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - numpy optional
        np = None  # type: ignore

    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()

    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set)):
        return [_json_default(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_default(val) for key, val in value.items()}
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(value, "__dict__"):
        return {k: _json_default(v) for k, v in value.__dict__.items()}
    return value


def _bool_from_any(value: Any) -> bool:
    """Best-effort conversion to bool, works for numpy arrays and lists."""
    if isinstance(value, dict):
        return all(_bool_from_any(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return all(_bool_from_any(item) for item in value)
    if hasattr(value, "all"):
        try:
            return bool(value.all())
        except Exception:  # pragma: no cover - best effort
            pass
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except Exception:  # pragma: no cover - best effort
            pass
    return bool(value)


@dataclass
class InterfaceSettings:
    checkpoint_path: Path
    checkpoint_number: Optional[int]
    policy_id: Optional[str]
    mode: str  # "single" or "multi"
    seed: int
    step_delay: float
    restart_delay: float
    auto_restart: bool
    log_tracebacks: bool
    action_repeat: int
    speedup: int
    tuner_path: Optional[Path]


class RLlibInterfaceRunner:
    def __init__(self, settings: InterfaceSettings) -> None:
        self.settings = settings
        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._emit_event(
            "settings",
            agent_type="rllib-raw",
            agent_path=str(settings.checkpoint_path),
            mode=settings.mode,
            action_repeat=settings.action_repeat,
            speedup=settings.speedup,
            policy=settings.policy_id,
        )

        self.env = None
        self.algo = None
        self.policies: List[str] = []

    def run(self) -> None:
        attempt = 0
        while not self._stop_requested:
            attempt += 1
            try:
                if self.env is None:
                    self._emit_event("status", message="Creating Godot environment...")
                    self.env = self._create_env()
                if self.algo is None:
                    self._emit_event("status", message="Restoring RLlib checkpoint...")
                    self.algo = self._load_algorithm()
                    self._emit_event(
                        "status",
                        message=f"RLlib policies ready: {', '.join(self.policies) or 'default'}",
                    )

                self._emit_event("connected", attempt=attempt, mode=self.settings.mode)
                if self.settings.mode == "multi":
                    assert isinstance(self.env, GDRLPettingZooEnv)
                    self._run_multi(self.env)
                else:
                    assert isinstance(self.env, GodotEnv)
                    self._run_single(self.env)
            except KeyboardInterrupt:
                self._emit_event("status", message="Interrupted by user, shutting down.")
                self._stop_requested = True
            except TimeoutError:
                self._emit_event(
                    "status",
                    message="Connection timed out. Is the Godot editor running?",
                )
                self._cleanup_env()
            except Exception as exc:  # pragma: no cover - runtime only
                if self.settings.log_tracebacks:
                    traceback.print_exc()
                self._emit_error(exc)
                self._cleanup_env()

            if self._stop_requested or not self.settings.auto_restart:
                break

            self._emit_event(
                "status",
                message=(
                    f"Restarting interface in {self.settings.restart_delay:.1f}s "
                    "after exit or error."
                ),
            )
            self._sleep(self.settings.restart_delay)

        self._cleanup_env()

    def _cleanup_env(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
            self.env = None

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        self._emit_event("status", message=f"Received signal {name}, stopping...")
        self._stop_requested = True

    def _emit_error(self, exc: Exception) -> None:
        payload: dict[str, Any] = {"message": str(exc)}
        if self.settings.log_tracebacks:
            payload["traceback"] = traceback.format_exc()
        self._emit_event("error", **payload)

    def _emit_event(self, kind: str, **data: Any) -> None:
        payload = {"timestamp": time.time(), "kind": kind}
        payload.update(data)
        print(
            f"{EVENT_PREFIX}{json.dumps(payload, default=_json_default)}",
            flush=True,
        )

    def _emit_action(
        self, episode: int, step: int, agents: Iterable[dict[str, Any]]
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "episode": episode,
            "step": step,
            "mode": self.settings.mode,
            "agents": list(agents),
        }
        print(
            f"{ACTION_PREFIX}{json.dumps(payload, default=_json_default)}",
            flush=True,
        )

    def _sleep(self, duration: float) -> None:
        end = time.time() + max(0.0, duration)
        while not self._stop_requested and time.time() < end:
            time.sleep(min(0.1, end - time.time()))

    def _create_env(self):
        env_config = {
            "action_repeat": self.settings.action_repeat,
            "speedup": self.settings.speedup,
            "env_path": None,
            "show_window": False,
        }
        if self.settings.mode == "multi":
            env = GDRLPettingZooEnv(
                config=env_config,
                show_window=False,
                seed=self.settings.seed,
            )
            env.godot_env.port = GodotEnv.DEFAULT_PORT
            return env
        env_kwargs = dict(env_config)
        env_kwargs.pop("show_window", None)
        return GodotEnv(
            **env_kwargs,
            show_window=False,
        )

    def _locate_checkpoint(self) -> Path:
        base = self.settings.checkpoint_path
        if not base.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {base}")
        if base.is_file():
            return base
        if base.name.startswith("checkpoint_") and base.is_dir():
            return base

        number = self.settings.checkpoint_number
        if number is not None:
            padded = f"checkpoint_{number:06d}"
            simple = f"checkpoint_{number}"
            for candidate in (padded, simple):
                candidate_path = base / candidate
                if candidate_path.exists():
                    return candidate_path
            raise FileNotFoundError(f"Checkpoint {number} not found inside {base}")

        checkpoints = sorted(
            [p for p in base.glob("checkpoint_*") if p.is_dir()],
            key=lambda p: p.name,
        )
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint_* directories under {base}")
        return checkpoints[-1]

    def _load_algorithm(self):
        import ray

        checkpoint_dir = self._locate_checkpoint()
        ray.init(ignore_reinit_error=True, include_dashboard=False, local_mode=True)

        if AlgorithmConfig is None:
            raise RuntimeError("ray[rllib] is required to load RLlib checkpoints.")

        # Load raw algorithm_state.pkl to reuse the stored config/algo class without spinning envs.
        algo_state_path = checkpoint_dir / "algorithm_state.pkl"
        with algo_state_path.open("rb") as f:
            algo_state = pickle.load(f)

        algo_class = algo_state["algorithm_class"]
        config_dict = dict(algo_state["config"])

        # Force inference-only, no env runners/workers, and override env config to in-editor API.
        config_dict.update(
            {
                "num_env_runners": 0,
                "num_rollout_workers": 0,
                "num_gpus": 0,
                "num_gpus_per_env_runner": 0,
                "num_envs_per_env_runner": 0,
                "evaluation_num_env_runners": 0,
                "evaluation_interval": None,
                "in_evaluation": True,
                "create_env_on_local_worker": False,
                "env_runner_cls": None,
                "env": "godot-inline",
                "env_config": {
                    "env_path": None,
                    "action_repeat": self.settings.action_repeat,
                    "show_window": False,
                    "speedup": self.settings.speedup,
                },
            }
        )

        # Fill in observation/action spaces for multi-agent policies so RLlib does not
        # try to infer them from a (stub) env during restore.
        if (
            self.settings.mode == "multi"
            and isinstance(self.env, GDRLPettingZooEnv)
            and PolicySpec is not None
        ):
            obs_spaces = getattr(self.env, "observation_spaces", {}) or {}
            act_spaces = getattr(self.env, "action_spaces", {}) or {}
            policy_names = getattr(self.env, "agent_policy_names", []) or []
            policies_cfg = config_dict.get("policies") or {}
            # Map each policy name to the corresponding agent's spaces.
            for agent_idx, policy_name in enumerate(policy_names):
                obs_space = obs_spaces.get(agent_idx)
                act_space = act_spaces.get(agent_idx)
                if obs_space is None or act_space is None:
                    continue
                policies_cfg[policy_name] = (
                    None,  # keep stored policy class (None = default)
                    obs_space,
                    act_space,
                    {},
                )
            if policies_cfg:
                config_dict["policies"] = policies_cfg

        if tune is None:
            raise RuntimeError("ray[tune] is required to register the inline env.")

        def _dummy_env(_config):
            # Provide a minimal MultiAgentEnv so RLlib validation passes without
            # spinning a real environment.
            if (
                self.settings.mode == "multi"
                and isinstance(self.env, GDRLPettingZooEnv)
                and MultiAgentEnv is not None
            ):
                obs_spaces = getattr(self.env, "observation_spaces", {}) or {}
                act_spaces = getattr(self.env, "action_spaces", {}) or {}
                agent_ids = list(obs_spaces.keys())

                class _StaticMultiEnv(MultiAgentEnv):
                    def __init__(self):
                        super().__init__()
                        self._obs_spaces = obs_spaces
                        self._act_spaces = act_spaces
                        self._agents = list(agent_ids)
                    
                    def observation_space(self, agent_id):
                        return self._obs_spaces[agent_id]

                    def action_space(self, agent_id):
                        return self._act_spaces[agent_id]

                    def reset(self, *, seed=None, options=None):
                        return (
                            {aid: self._obs_spaces[aid].sample() for aid in self._agents},
                            {aid: {} for aid in self._agents},
                        )

                    def step(self, action_dict):
                        obs = {aid: self._obs_spaces[aid].sample() for aid in action_dict}
                        rewards = {aid: 0.0 for aid in action_dict}
                        terminations = {aid: True for aid in action_dict}
                        truncations = {aid: False for aid in action_dict}
                        infos = {aid: {} for aid in action_dict}
                        return obs, rewards, terminations, truncations, infos

                return _StaticMultiEnv()

            class _NoOpEnv:
                def __init__(self):
                    self.action_space = None
                    self.observation_space = None

                def reset(self, *_, **__):
                    raise RuntimeError(
                        "Inline env should never be stepped. This is a stub for RLlib restore."
                    )

            return _NoOpEnv()

        tune.register_env("godot-inline", lambda cfg: _dummy_env(cfg))

        cfg = AlgorithmConfig.from_dict(config_dict)
        algo = algo_class(config=cfg)
        algo.restore(str(checkpoint_dir))
        algo.stop_workers()  # extra safety to avoid stray envs

        try:
            self.policies = list(algo.workers.local_worker().policy_map.keys())
        except Exception:
            self.policies = ["default_policy"]
        return algo

    def _policy_for_agent(self, agent_id: str) -> str:
        if not self.policies:
            return "default_policy"
        if self.settings.policy_id and self.settings.policy_id in self.policies:
            return self.settings.policy_id
        if len(self.policies) == 1:
            return self.policies[0]
        if agent_id in self.policies:
            return agent_id
        return self.policies[0]

    def _compute_action(self, policy_id: str, observation):
        if self.algo is None:
            raise RuntimeError("RLlib algorithm not loaded")
        policy = self.algo.get_policy(policy_id)
        action = policy.compute_single_action(observation, explore=False)[0]
        return action

    def _run_single(self, env: GodotEnv) -> None:
        episode = 0
        while not self._stop_requested:
            episode += 1
            self._emit_event("episode_start", mode="single", episode=episode)
            try:
                obs_tuple = env.reset(seed=self.settings.seed + episode)
                observations = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
            except TypeError:
                observations = env.reset()

            step = 0
            done = False
            while not done and not self._stop_requested:
                step += 1
                actions = []
                policies = []
                for idx, obs in enumerate(observations):
                    policy_id = self._policy_for_agent(f"agent_{idx}")
                    act = self._compute_action(policy_id, obs)
                    actions.append(act)
                    policies.append(policy_id)

                obs, rewards, done_flags, _, infos = env.step(list(zip(*actions)))
                done = _bool_from_any(done_flags)

                agents = []
                for idx in range(len(rewards)):
                    agents.append(
                        {
                            "agent": f"agent_{idx}",
                            "policy": policies[idx],
                            "observation": observations[idx],
                            "action": actions[idx],
                            "reward": rewards[idx],
                            "done": done_flags[idx],
                            "info": infos[idx],
                        }
                    )
                self._emit_action(episode, step, agents)
                observations = obs
                if self.settings.step_delay > 0:
                    self._sleep(self.settings.step_delay)
            self._emit_event("episode_end", episode=episode, steps=step)

    def _run_multi(self, env: GDRLPettingZooEnv) -> None:
        agent_policy_names: Dict[Any, Any] = getattr(env, "agent_policy_names", {}) or {}
        episode = 0
        while not self._stop_requested:
            episode += 1
            self._emit_event("episode_start", mode="multi", episode=episode)
            observations, _ = self._safe_reset_pz(env, self.settings.seed + episode)
            step = 0
            while env.agents and not self._stop_requested:
                step += 1
                actions: Dict[str, Any] = {}
                policy_ids: Dict[str, str] = {}
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    policy_id = self._policy_for_agent(agent_id)
                    actions[agent_id] = self._compute_action(policy_id, obs)
                    policy_ids[agent_id] = policy_id

                step_result = env.step(actions)
                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = step_result

                rows: List[dict[str, Any]] = []
                for agent_id, action in actions.items():
                    rows.append(
                        {
                            "agent": agent_id,
                            "policy": policy_ids.get(agent_id, agent_policy_names.get(agent_id, "policy")),
                            "observation": observations.get(agent_id),
                            "action": action,
                            "reward": rewards.get(agent_id),
                            "terminated": terminations.get(agent_id, False),
                            "truncated": truncations.get(agent_id, False),
                            "info": infos.get(agent_id, {}),
                        }
                    )
                self._emit_action(episode, step, rows)
                if self.settings.step_delay > 0:
                    self._sleep(self.settings.step_delay)
                if not env.agents:
                    break
            self._emit_event("episode_end", episode=episode, steps=step)

    def _safe_reset_pz(
        self, env: GDRLPettingZooEnv, episode_seed: Optional[int]
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        try:
            if episode_seed is None:
                result = env.reset()
            else:
                result = env.reset(seed=episode_seed)
        except TypeError:
            result = env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, {}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a raw RLlib agent against Godot.")
    parser.add_argument("checkpoint_path", type=Path, help="Checkpoint directory or parent path.")
    parser.add_argument(
        "--checkpoint-number",
        type=int,
        help="Select a specific checkpoint number inside the directory.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Policy ID to force (defaults to first available or agent_id match).",
    )
    parser.add_argument(
        "--tuner",
        type=Path,
        help="Optional tuner.pkl path if your checkpoint relies on a Tune Tuner bundle.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="multi",
        help="Environment mode.",
    )
    parser.add_argument("--action-repeat", type=int, default=1, help="action_repeat override.")
    parser.add_argument("--speedup", type=int, default=1, help="speedup override.")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to the env.")
    parser.add_argument("--step-delay", type=float, default=0.0, help="Pause between steps.")
    parser.add_argument("--restart-delay", type=float, default=2.0, help="Delay before reconnect.")
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Exit immediately if the env exits or errors instead of reconnecting.",
    )
    parser.add_argument(
        "--log-tracebacks",
        action="store_true",
        help="Include Python tracebacks in error events.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = InterfaceSettings(
        checkpoint_path=args.checkpoint_path,
        checkpoint_number=args.checkpoint_number,
        policy_id=args.policy,
        mode=args.mode,
        seed=args.seed,
        step_delay=max(args.step_delay, 0.0),
        restart_delay=max(args.restart_delay, 0.0),
        auto_restart=not args.no_auto_restart,
        log_tracebacks=args.log_tracebacks,
        action_repeat=max(args.action_repeat, 1),
        speedup=max(args.speedup, 1),
        tuner_path=args.tuner,
    )
    runner = RLlibInterfaceRunner(settings)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
