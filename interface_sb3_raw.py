#!/usr/bin/env python3
"""
Workspace-only interface runner for Stable Baselines3 checkpoints.

This script avoids the old interface launcher and focuses purely on
loading a raw SB3 .zip model from your workspace, connecting to the
in-editor Godot environment, and streaming interface events/actions
for the Rust TUI.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

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
    model_path: Path
    mode: str  # "single" or "multi"
    seed: int
    step_delay: float
    restart_delay: float
    auto_restart: bool
    log_tracebacks: bool
    action_repeat: int
    speedup: int
    algo: Optional[str]


class SB3InterfaceRunner:
    def __init__(self, settings: InterfaceSettings) -> None:
        self.settings = settings
        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._emit_event(
            "settings",
            agent_type="sb3-raw",
            agent_path=str(settings.model_path),
            mode=settings.mode,
            action_repeat=settings.action_repeat,
            speedup=settings.speedup,
        )
        self.env = None
        self.model = None

    def run(self) -> None:
        attempt = 0
        while not self._stop_requested:
            attempt += 1
            try:
                if self.env is None:
                    self._emit_event("status", message="Creating Godot environment...")
                    self.env = self._create_env()
                if self.model is None:
                    self._emit_event("status", message="Loading SB3 model...")
                    self.model = self._load_model(self.settings.model_path, self.settings.algo)
                    self._emit_event("status", message="Model loaded")

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
                    message=(
                        "Connection to Godot timed out. "
                        "Is the editor running with the API server?"
                    ),
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

    def _load_model(self, model_path: Path, algo_name: Optional[str] = None):
        from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
        from stable_baselines3.common.save_util import load_from_zip_file

        algo_map = {
            "ppo": PPO,
            "a2c": A2C,
            "dqn": DQN,
            "sac": SAC,
            "td3": TD3,
            "ddpg": DDPG,
        }

        if algo_name:
            algo_cls = algo_map.get(algo_name.lower())
            if algo_cls is None:
                raise ValueError(f"Unknown SB3 algorithm: {algo_name}")
            return algo_cls.load(model_path, device="cpu")

        loaded = load_from_zip_file(model_path)
        for part in loaded:
            if isinstance(part, dict) and "algo" in part:
                algo_cls = algo_map.get(str(part["algo"]).lower())
                if algo_cls:
                    return algo_cls.load(model_path, device="cpu")

        for candidate, algo_cls in algo_map.items():
            try:
                return algo_cls.load(model_path, device="cpu")
            except Exception:
                continue

        raise ValueError("Unable to load SB3 model; specify --algo to disambiguate.")

    def _predict_action(self, observation):
        if self.model is None:
            raise RuntimeError("SB3 model not loaded")
        action, _ = self.model.predict(observation, deterministic=True)
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
                action = [self._predict_action(obs) for obs in observations]
                obs, rewards, done_flags, _, infos = env.step(list(zip(*action)))
                done = _bool_from_any(done_flags)

                agents = []
                for idx in range(len(rewards)):
                    agents.append(
                        {
                            "agent": f"agent_{idx}",
                            "policy": "sb3",
                            "observation": observations[idx],
                            "action": action[idx],
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
        agent_policy_names = getattr(env, "agent_policy_names", {}) or {}
        episode = 0
        while not self._stop_requested:
            episode += 1
            self._emit_event("episode_start", mode="multi", episode=episode)
            observations, _ = self._safe_reset_pz(env, self.settings.seed + episode)
            step = 0
            while env.agents and not self._stop_requested:
                step += 1
                actions = {}
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    actions[agent_id] = self._predict_action(obs)

                step_result = env.step(actions)
                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = step_result

                rows = []
                for agent_id, action in actions.items():
                    rows.append(
                        {
                            "agent": agent_id,
                            "policy": agent_policy_names.get(agent_id, "sb3"),
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
    parser = argparse.ArgumentParser(description="Run a raw SB3 agent against Godot.")
    parser.add_argument("model_path", type=Path, help="Path to the SB3 .zip model.")
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
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
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "a2c", "dqn", "sac", "td3", "ddpg"],
        help="Algorithm type if auto-detection fails.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = InterfaceSettings(
        model_path=args.model_path,
        mode=args.mode,
        seed=args.seed,
        step_delay=max(args.step_delay, 0.0),
        restart_delay=max(args.restart_delay, 0.0),
        auto_restart=not args.no_auto_restart,
        log_tracebacks=args.log_tracebacks,
        action_repeat=max(args.action_repeat, 1),
        speedup=max(args.speedup, 1),
        algo=args.algo,
    )
    runner = SB3InterfaceRunner(settings)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
