#!/usr/bin/env python3
"""
Random-action simulator for Godot RL environments.

This helper keeps reconnecting to the Godot environment and streams
structured output that the Rust TUI can render as a "Simulator" tab.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from godot_rl.core.godot_env import GodotEnv
    from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
except ImportError as exc:  # pragma: no cover - dependency check
    print(
        "This script requires the 'godot-rl' package that ships with the controller.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

EVENT_PREFIX = "@SIM_EVENT "
ACTION_PREFIX = "@SIM_ACTION "


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
class SimulatorSettings:
    mode: str  # "single" or "multi"
    env_config: dict[str, Any]
    show_window: bool
    seed: int
    step_delay: float
    restart_delay: float
    max_episodes: Optional[int]
    max_steps: Optional[int]
    auto_restart: bool
    log_tracebacks: bool
    
    def __str__(self) -> str:
        return (
            f"SimulatorSettings(mode={self.mode}, "
            f"env_path={self.env_config.get('env_path')}, "
            f"show_window={self.show_window}, "
            f"seed={self.seed}, "
            f"step_delay={self.step_delay}, "
            f"restart_delay={self.restart_delay}, "
            f"max_episodes={self.max_episodes}, "
            f"max_steps={self.max_steps}, "
            f"auto_restart={self.auto_restart}, "
            f"log_tracebacks={self.log_tracebacks}, "
        )


class SimulationRunner:
    def __init__(self, settings: SimulatorSettings) -> None:
        self.settings = settings
        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._emit_event("settings", **settings.__dict__)

    def run(self) -> None:
        attempt = 0
        while not self._stop_requested:
            attempt += 1
            env = None
            try:
                env = self._create_env()
                self._emit_event(
                    "connected",
                    attempt=attempt,
                    mode=self.settings.mode,
                    env_path=self.settings.env_config.get("env_path"),
                )
                if self.settings.mode == "multi":
                    assert isinstance(env, GDRLPettingZooEnv)
                    self._run_multi(env)
                else:
                    assert isinstance(env, GodotEnv)
                    self._run_single(env)
            except KeyboardInterrupt:
                self._emit_event("status", message="Interrupted by user, shutting down.")
                self._stop_requested = True
            except TimeoutError as exc:
                self._emit_event(
                    "status",
                    message=(
                        "Connection to Godot environment timed out. "
                        "Is the environment running and reachable?"
                    ),
                )
                continue
            except Exception as exc:  # pragma: no cover - exercised at runtime
                traceback.print_exc()
                self._emit_error(exc)
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass

            if self._stop_requested or not self.settings.auto_restart:
                break

            self._emit_event(
                "status",
                message=(
                    f"Restarting simulator in {self.settings.restart_delay:.1f}s "
                    "after error or environment exit."
                ),
            )
            self._sleep(self.settings.restart_delay)

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
        if self.settings.mode == "multi":
            return GDRLPettingZooEnv(
                config=self.settings.env_config,
                show_window=self.settings.show_window,
                seed=self.settings.seed,
            )
        env_kwargs = dict(self.settings.env_config)
        env_kwargs.pop("show_window", None)
        return GodotEnv(
            **env_kwargs,
            show_window=self.settings.show_window,
        )

    def _run_single(self, env: GodotEnv) -> None:
        episode = 0
        while not self._stop_requested:
            if self._limit_hit(episode):
                break
            episode += 1
            self._emit_event("episode_start", mode="single", episode=episode)
            try:
                env.reset(seed=self.settings.seed + episode)
            except TypeError:
                env.reset()

            step = 0
            done = False
            while not done and not self._stop_requested:
                if self.settings.max_steps and step >= self.settings.max_steps:
                    self._emit_event(
                        "status",
                        message=f"Max steps reached in episode {episode}, resetting.",
                    )
                    break
                step += 1
                action = [env.action_space.sample() for _ in range(env.num_envs or 1)]
                # action = list(zip(*action))
                obs, rewards, done_flags, _, infos = env.step(list(zip(*action)))
                done = _bool_from_any(done_flags)
                agents = [
                    # {
                    #     "agent": "env",
                    #     "action": action,
                    #     "reward": reward,
                    #     "done": done_flag,
                    #     "info": info,
                    # }
                ]
                for idx in range(len(rewards)):
                    agents.append(
                        {
                            "agent": f"agent_{idx}",
                            "policy": "random",
                            "action": action[idx],
                            "reward": rewards[idx],
                            "done": done_flags[idx],
                            "info": infos[idx],
                        }
                    )
                self._emit_action(episode, step, agents)
                if self.settings.step_delay > 0:
                    self._sleep(self.settings.step_delay)
            self._emit_event("episode_end", episode=episode, steps=step)

    def _run_multi(self, env: GDRLPettingZooEnv) -> None:
        agent_policy_names: Dict[Any, Any] = getattr(
            env, "agent_policy_names", {}
        ) or {}
        print(agent_policy_names)
        episode = 0
        while not self._stop_requested:
            if self._limit_hit(episode):
                break
            episode += 1
            self._emit_event("episode_start", mode="multi", episode=episode)
            observations, _ = self._safe_reset_pz(env, self.settings.seed + episode)
            step = 0
            while env.agents and not self._stop_requested:
                if self.settings.max_steps and step >= self.settings.max_steps:
                    self._emit_event(
                        "status",
                        message=f"Max steps reached in episode {episode}, resetting.",
                    )
                    break
                step += 1
                actions = {
                    agent: env.action_space(agent).sample() for agent in env.agents
                }
                # env.godot_env.connection
                step_result = env.step(actions)
                # if len(step_result) == 5:
                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = step_result
                # else:  # pragma: no cover - compatibility with older pettingzoo
                #     observations, rewards, terminations, infos = step_result
                #     truncations = {agent: False for agent in rewards}
                panel_rows: List[dict[str, Any]] = []
                for agent_id, action in actions.items():
                    row = {
                        "agent": agent_id,
                        "policy": agent_policy_names[agent_id],
                        "action": action,
                        "reward": rewards.get(agent_id),
                        "terminated": terminations.get(agent_id, False),
                        "truncated": truncations.get(agent_id, False),
                        "info": infos.get(agent_id, {}),
                    }
                    panel_rows.append(row)
                self._emit_action(episode, step, panel_rows)
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

    def _limit_hit(self, episode: int) -> bool:
        if self.settings.max_episodes and episode >= self.settings.max_episodes:
            self._emit_event("status", message="Reached max episodes, stopping.")
            return True
        return False


def _positive_or_none(value: Optional[int]) -> Optional[int]:
    if value is None or value <= 0:
        return None
    return value


def _build_settings(args: argparse.Namespace) -> SimulatorSettings:
    external_config: dict[str, Any] = {}

    env_config = dict(external_config.get("config", {}).get("env_config") or {})

    resolved_env_path: Optional[Path] = None
    if args.env_path:
        resolved_env_path = Path(args.env_path)

    env_config["env_path"] = str(resolved_env_path.expanduser()) if resolved_env_path else None

    if args.action_repeat is not None:
        env_config["action_repeat"] = args.action_repeat
    else:
        env_config["action_repeat"] = env_config.get("action_repeat", 1)
    if args.speedup is not None:
        env_config["speedup"] = args.speedup
    else:
        env_config["speedup"] = env_config.get("speedup", 1)

    show_window = env_config.get("show_window", False)
    if args.show_window is not None:
        show_window = args.show_window
    env_config["show_window"] = show_window

    mode = args.mode
    if mode == "auto":
        if external_config:
            mode = "multi" if external_config.get("env_is_multiagent") else "single"
        else:
            mode = "single"

    if mode not in {"single", "multi"}:
        raise SystemExit(f"Unsupported mode '{mode}'.")

    max_episodes = _positive_or_none(args.max_episodes)
    max_steps = _positive_or_none(args.max_steps)

    return SimulatorSettings(
        mode=mode,
        env_config=env_config,
        show_window=show_window,
        seed=args.seed,
        step_delay=max(args.step_delay, 0.0),
        restart_delay=max(args.restart_delay, 0.0),
        max_episodes=max_episodes,
        max_steps=max_steps,
        auto_restart=not args.no_auto_restart,
        log_tracebacks=args.log_tracebacks,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Random-action simulator for Godot RL environments."
    )
    parser.add_argument(
        "--env-path",
        type=str,
        help="Path to the exported Godot environment (overrides config file).",
    )

    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Simulation mode. 'auto' reads env_is_multiagent from the config file.",
    )

    window_group = parser.add_mutually_exclusive_group()
    window_group.add_argument(
        "--show-window",
        dest="show_window",
        action="store_const",
        const=True,
        help="Force the Godot window to be visible.",
    )
    window_group.add_argument(
        "--headless",
        dest="show_window",
        action="store_const",
        const=False,
        help="Force headless mode even if the config enables rendering.",
    )
    parser.set_defaults(show_window=None)

    parser.add_argument("--action-repeat", type=int, help="Override action_repeat.")
    parser.add_argument("--speedup", type=int, help="Override speedup.")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to the env.")
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Optional pause (seconds) between environment steps.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=2.0,
        help="Delay before reconnecting after an error or env shutdown.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Stop after this many episodes (<=0 means unlimited).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Stop or reset an episode after this many steps (<=0 means unlimited).",
    )
    parser.add_argument(
        "--no-auto-restart",
        action="store_true",
        help="Exit immediately if the env exits or errors instead of reconnecting.",
    )
    parser.add_argument(
        "--log-tracebacks",
        action="store_true",
        help="Include Python tracebacks in error events (useful for debugging).",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = _build_settings(args)
    runner = SimulationRunner(settings)
    runner.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
