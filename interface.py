#!/usr/bin/env python3
"""
Agent interfacing script for Godot RL environments.

This script loads either a Stable Baselines3 (SB3) .zip model or an RLlib checkpoint,
then runs the agent within the Godot editor using the API system (no bin file).
It streams structured output that the Rust TUI can render as an "Interface" tab,
showing observations, rewards, done flags, and actions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
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
except ImportError as exc:
    print(
        "This script requires the 'godot-rl' package.",
        file=sys.stderr,
    )
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)
    
from custom_models import register_rllib_models


# Event prefixes for structured output
EVENT_PREFIX = "@INTERFACE_EVENT "
ACTION_PREFIX = "@INTERFACE_ACTION "

# Agent type constants
AGENT_TYPE_SB3 = "sb3"
AGENT_TYPE_RLLIB = "rllib"


def _json_default(value: Any) -> Any:
    """Convert numpy/scalar containers into JSON-friendly types."""
    try:
        import numpy as np
    except ModuleNotFoundError:
        np = None

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
        except Exception:
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
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except Exception:
            pass
    return bool(value)


@dataclass
class InterfaceSettings:
    agent_type: str  # "sb3" or "rllib"
    agent_path: str  # Path to .zip (SB3) or checkpoint directory (RLlib)
    mode: str  # "single" or "multi"
    env_config: dict[str, Any]
    show_window: bool
    seed: int
    step_delay: float
    restart_delay: float
    auto_restart: bool
    log_tracebacks: bool
    # RLlib-specific settings
    rllib_checkpoint_number: Optional[int]
    rllib_policy_id: Optional[str]
    # SB3-specific settings
    sb3_algo: Optional[str]
    
    def __str__(self) -> str:
        return (
            f"InterfaceSettings(agent_type={self.agent_type}, "
            f"agent_path={self.agent_path}, "
            f"mode={self.mode}, "
            f"show_window={self.show_window})"
        )


class AgentLoader:
    """Loads SB3 or RLlib agents."""
    
    @staticmethod
    def load_sb3_agent(model_path: Path, algo_name: Optional[str] = None):
        """Load a Stable Baselines3 model from a .zip file."""
        from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
        from stable_baselines3.common.save_util import load_from_zip_file
        
        ALGO_CLASSES = {
            "ppo": PPO,
            "a2c": A2C,
            "dqn": DQN,
            "sac": SAC,
            "td3": TD3,
            "ddpg": DDPG,
        }
        
        if algo_name:
            algo_cls = ALGO_CLASSES.get(algo_name.lower())
            if not algo_cls:
                raise ValueError(f"Unknown algorithm: {algo_name}")
            return algo_cls.load(model_path, device="cpu")
        
        # Try to auto-detect algorithm
        loaded = load_from_zip_file(model_path)
        for part in loaded:
            if isinstance(part, dict) and "algo" in part:
                algo_cls = ALGO_CLASSES.get(str(part["algo"]).lower())
                if algo_cls:
                    return algo_cls.load(model_path, device="cpu")
        
        # Try common algorithms in order
        for algo_name, algo_cls in ALGO_CLASSES.items():
            try:
                return algo_cls.load(model_path, device="cpu")
            except Exception:
                continue
        
        raise ValueError("Could not load SB3 model. Please specify --algo.")
    
    @staticmethod
    def load_rllib_agent(checkpoint_path: Path, checkpoint_number: Optional[int] = None, multiagent: bool = True):
        """Load an RLlib agent from a checkpoint."""
        from ray import tune
        from ray.rllib.algorithms.algorithm import Algorithm
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
        
        # Register env
        def env_creator(env_config):
            index = env_config.worker_index + env_config.vector_index
            port = index + GodotEnv.DEFAULT_PORT
            seed = index
            if multiagent:
                env_config["env_path"] = None
                return ParallelPettingZooEnv(
                    GDRLPettingZooEnv(config=env_config, port=port, seed=seed)
                )
            else:
                return RayVectorGodotEnv(config=env_config, port=port, seed=seed)
        
        tune.register_env("godot", env_creator)
        
        # Resolve checkpoint path
        if checkpoint_number is not None and not checkpoint_path.name.startswith("checkpoint_"):
            # Find specific checkpoint
            padded_name = f"checkpoint_{checkpoint_number:06d}"
            simple_name = f"checkpoint_{checkpoint_number}"
            
            if (checkpoint_path / padded_name).is_dir():
                checkpoint_path = checkpoint_path / padded_name
            elif (checkpoint_path / simple_name).is_dir():
                checkpoint_path = checkpoint_path / simple_name
            else:
                # Search for any matching checkpoint
                for entry in checkpoint_path.iterdir():
                    if entry.is_dir() and entry.name.startswith("checkpoint_"):
                        suffix = entry.name.split("checkpoint_")[-1]
                        try:
                            if int(suffix) == checkpoint_number:
                                checkpoint_path = entry
                                break
                        except ValueError:
                            continue
                else:
                    raise FileNotFoundError(f"Checkpoint {checkpoint_number} not found in {checkpoint_path}")
        elif not checkpoint_path.name.startswith("checkpoint_"):
            # Find latest checkpoint
            checkpoints = list(checkpoint_path.glob("checkpoint_*"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
            checkpoints.sort(key=lambda p: int(p.name.split("_")[-1]))
            checkpoint_path = checkpoints[-1]
        
        # Load algorithm
        algo = Algorithm.from_checkpoint(str(checkpoint_path.resolve()))
        return algo


class InterfaceRunner:
    def __init__(self, settings: InterfaceSettings) -> None:
        self.settings = settings
        self._stop_requested = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        self._emit_event("settings", **{
            "agent_type": settings.agent_type,
            "agent_path": settings.agent_path,
            "mode": settings.mode,
        })
        
        # Create environment first
        self.env = None
        self._emit_event("status", message="Creating environment...")
        self.env = self._create_env()
        self._emit_event("status", message="Environment created successfully")
        
        # Load agent
        self.agent = None
        self.rllib_policies = None
        self._load_agent()

    def _load_agent(self):
        """Load the agent based on agent_type."""
        agent_path = Path(self.settings.agent_path)
        
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent path not found: {agent_path}")
        
        self._emit_event("status", message=f"Loading {self.settings.agent_type} agent from {agent_path}...")
        
        if self.settings.agent_type == AGENT_TYPE_SB3:
            self.agent = AgentLoader.load_sb3_agent(agent_path, self.settings.sb3_algo)
            self._emit_event("status", message=f"SB3 agent loaded successfully")
        elif self.settings.agent_type == AGENT_TYPE_RLLIB:
            multiagent = (self.settings.mode == "multi")
            self.agent = AgentLoader.load_rllib_agent(
                agent_path,
                self.settings.rllib_checkpoint_number,
                multiagent
            )
            # Get policy IDs
            try:
                self.rllib_policies = list(self.agent.env_runner.policy_map.keys()) # type: ignore
            except:
                try:
                    config_dict = self.agent.config if isinstance(self.agent.config, dict) else {}
                    multiagent_config = config_dict.get("multiagent", {}) if config_dict else {}
                    policies_config = multiagent_config.get("policies", {}) if isinstance(multiagent_config, dict) else {}
                    self.rllib_policies = list(policies_config.keys()) if policies_config else ["default_policy"]
                except:
                    self.rllib_policies = ["default_policy"]
            
            self._emit_event("status", message=f"RLlib agent loaded successfully. Policies: {self.rllib_policies}")
        else:
            raise ValueError(f"Unknown agent type: {self.settings.agent_type}")

    def run(self) -> None:
        attempt = 0
        while not self._stop_requested:
            attempt += 1
            try:
                # Use the environment created in __init__
                if self.env is None:
                    self._emit_event("status", message="Recreating environment after error...")
                    self.env = self._create_env()
                
                self._emit_event(
                    "connected",
                    attempt=attempt,
                    mode=self.settings.mode,
                )
                if self.settings.mode == "multi":
                    assert isinstance(self.env, GDRLPettingZooEnv)
                    self._run_multi(self.env)
                else:
                    assert isinstance(self.env, GodotEnv)
                    self._run_single(self.env)
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
                # Close and recreate environment on timeout
                if self.env is not None:
                    try:
                        self.env.close()
                    except Exception:
                        pass
                    self.env = None
                continue
            except Exception as exc:
                traceback.print_exc()
                self._emit_error(exc)
                # Close and recreate environment on error
                if self.env is not None:
                    try:
                        self.env.close()
                    except Exception:
                        pass
                    self.env = None

            if self._stop_requested or not self.settings.auto_restart:
                break

            self._emit_event(
                "status",
                message=(
                    f"Restarting interface in {self.settings.restart_delay:.1f}s "
                    "after error or environment exit."
                ),
            )
            self._sleep(self.settings.restart_delay)
        
        # Clean up environment on exit
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass

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
            self.settings.env_config["env_path"] = None
            env = GDRLPettingZooEnv(
                config=self.settings.env_config,
                show_window=self.settings.show_window,
                seed=self.settings.seed,
            )
            env.godot_env.port = GodotEnv.DEFAULT_PORT
            return env
        env_kwargs = dict(self.settings.env_config)
        env_kwargs.pop("show_window", None)
        return GodotEnv(
            **env_kwargs,
            show_window=self.settings.show_window,
        )

    def _predict_sb3_action(self, observation):
        """Get action from SB3 agent."""
        if self.agent is None:
            raise RuntimeError("SB3 agent not loaded")
        
        action, _ = self.agent.predict(observation, deterministic=True)
        return action

    def _predict_rllib_action(self, agent_id: str, observation):
        """Get action from RLlib agent for a specific agent."""
        if self.agent is None:
            raise RuntimeError("RLlib agent not loaded")
        
        if self.rllib_policies is None:
            raise RuntimeError("RLlib policies not available")
        
        # Get the policy for this agent
        policy_id = self.settings.rllib_policy_id or self.rllib_policies[0]
        if len(self.rllib_policies) > 1 and agent_id in self.rllib_policies:
            policy_id = agent_id
        
        policy = self.agent.get_policy(policy_id)
        action = policy.compute_single_action(observation)[0]
        return action, policy_id

    def _run_single(self, env: GodotEnv) -> None:
        episode = 0
        while not self._stop_requested:
            episode += 1
            self._emit_event("episode_start", mode="single", episode=episode)
            try:
                obs_tuple = env.reset(seed=self.settings.seed + episode)
                if isinstance(obs_tuple, tuple):
                    observations = obs_tuple[0]
                else:
                    observations = obs_tuple
            except TypeError:
                observations = env.reset()

            step = 0
            done = False
            while not done and not self._stop_requested:
                step += 1
                
                # Get actions from the agent (not random)
                if self.settings.agent_type == AGENT_TYPE_SB3:
                    action = [self._predict_sb3_action(obs) for obs in observations]
                else:
                    # For RLlib single agent, use the first/default policy
                    action = []
                    for idx, obs in enumerate(observations):
                        act, policy_id = self._predict_rllib_action(f"agent_{idx}", obs)
                        action.append(act)
                
                obs, rewards, done_flags, _, infos = env.step(list(zip(*action)))
                done = _bool_from_any(done_flags)
                
                agents = []
                for idx in range(len(rewards)):
                    policy_name = "sb3" if self.settings.agent_type == AGENT_TYPE_SB3 else self.settings.rllib_policy_id or "default"
                    agents.append(
                        {
                            "agent": f"agent_{idx}",
                            "policy": policy_name,
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
        agent_policy_names: Dict[Any, Any] = getattr(
            env, "agent_policy_names", {}
        ) or {}
        
        episode = 0
        while not self._stop_requested:
            episode += 1
            self._emit_event("episode_start", mode="multi", episode=episode)
            observations, _ = self._safe_reset_pz(env, self.settings.seed + episode)
            step = 0
            while env.agents and not self._stop_requested:
                step += 1
                
                # Get actions from the agent (not random)
                actions = {}
                policy_ids = {}
                for agent_id in env.agents:
                    obs = observations[agent_id]
                    if self.settings.agent_type == AGENT_TYPE_SB3:
                        action = self._predict_sb3_action(obs)
                        policy_ids[agent_id] = "sb3"
                    else:
                        action, policy_id = self._predict_rllib_action(agent_id, obs)
                        policy_ids[agent_id] = policy_id
                    actions[agent_id] = action
                
                step_result = env.step(actions)
                (
                    observations,
                    rewards,
                    terminations,
                    truncations,
                    infos,
                ) = step_result
                
                panel_rows: List[dict[str, Any]] = []
                for agent_id, action in actions.items():
                    row = {
                        "agent": agent_id,
                        "policy": policy_ids.get(agent_id, agent_policy_names.get(agent_id, "unknown")),
                        "observation": observations.get(agent_id),
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

def _build_settings(args: argparse.Namespace) -> InterfaceSettings:
    env_config: dict[str, Any] = {}

    if args.action_repeat is not None:
        env_config["action_repeat"] = args.action_repeat
    else:
        env_config["action_repeat"] = 1
    if args.speedup is not None:
        env_config["speedup"] = args.speedup
    else:
        env_config["speedup"] = 1

    show_window = args.show_window if args.show_window is not None else False
    env_config["show_window"] = show_window

    return InterfaceSettings(
        agent_type=args.agent_type,
        agent_path=args.agent_path,
        mode=args.mode,
        env_config=env_config,
        show_window=show_window,
        seed=args.seed,
        step_delay=max(args.step_delay, 0.0),
        restart_delay=max(args.restart_delay, 0.0),
        auto_restart=not args.no_auto_restart,
        log_tracebacks=args.log_tracebacks,
        rllib_checkpoint_number=args.checkpoint_number,
        rllib_policy_id=args.policy,
        sb3_algo=args.algo,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent interface for Godot RL environments."
    )
    parser.add_argument(
        "agent_type",
        choices=["sb3", "rllib"],
        help="Type of agent to load (sb3 for Stable Baselines3, rllib for RLlib).",
    )
    parser.add_argument(
        "agent_path",
        type=str,
        help="Path to the agent (.zip for SB3, checkpoint directory for RLlib).",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Environment mode.",
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
        help="Force headless mode.",
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
        "--no-auto-restart",
        action="store_true",
        help="Exit immediately if the env exits or errors instead of reconnecting.",
    )
    parser.add_argument(
        "--log-tracebacks",
        action="store_true",
        help="Include Python tracebacks in error events (useful for debugging).",
    )
    
    # RLlib-specific options
    parser.add_argument(
        "--checkpoint-number",
        type=int,
        help="RLlib: Select a specific checkpoint number.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="RLlib: Select a specific policy ID to use.",
    )
    
    # SB3-specific options
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "a2c", "dqn", "sac", "td3", "ddpg"],
        help="SB3: Algorithm type if auto-detection fails.",
    )
    
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    register_rllib_models()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    settings = _build_settings(args)
    runner = InterfaceRunner(settings)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
