"""
Quick RLlib PPO smoke test to exercise controller range validation with a
lightweight PettingZoo environment instead of Godot.

The script builds PPO configs that reflect the controller's advanced settings
(workers, batch sizes, learning hyperparameters, etc.) and runs short training
loops to ensure the combinations fit within RLlib expectations. Use ``--sweep``
to iterate over min/max boundary cases that mirror the UI validation ranges.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Tuple

import ray
from pettingzoo.classic import rps_v2
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


@dataclass
class RangeRunConfig:
    """Configuration values that mirror the controller's advanced RLlib fields."""

    # Environment / rollout settings
    num_workers: int = 0
    num_envs_per_worker: int = 1
    rollout_fragment_length: int = 200
    batch_mode: str = "truncate_episodes"

    # Training hyperparameters
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 256
    num_sgd_iter: int = 10
    lr: float = 0.0003
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.0
    vf_loss_coeff: float = 1.0
    grad_clip: float = 0.5

    # Model specifics
    framework: str = "torch"
    activation: str = "tanh"
    use_lstm: bool = False
    lstm_cell_size: int = 256
    lstm_num_layers: int = 1
    lstm_max_seq_len: int = 128

    # Run control
    iterations: int = 1


def _make_env(_config=None):
    return ParallelPettingZooEnv(rps_v2.parallel_env())


def build_rllib_config(values: RangeRunConfig) -> PPOConfig:
    """Translate :class:`RangeRunConfig` into a RLlib ``PPOConfig`` instance."""

    model_cfg: Dict[str, object] = {"fcnet_hiddens": [128, 128], "activation": values.activation}
    if values.use_lstm:
        model_cfg.update(
            {
                "use_lstm": True,
                "lstm_cell_size": values.lstm_cell_size,
                "max_seq_len": values.lstm_max_seq_len,
                "lstm_use_prev_action": True,
                "lstm_use_prev_reward": True,
                "num_layers": values.lstm_num_layers,
            }
        )

    config = (
        PPOConfig()
        .environment(env=_make_env)
        .framework(values.framework)
        .env_runners(
            num_env_runners=values.num_workers,
            num_envs_per_env_runner=values.num_envs_per_worker,
        )
        .rollouts(
            batch_mode=values.batch_mode,
            rollout_fragment_length=values.rollout_fragment_length,
        )
        .training(
            train_batch_size=values.train_batch_size,
            sgd_minibatch_size=values.sgd_minibatch_size,
            num_sgd_iter=values.num_sgd_iter,
            lr=values.lr,
            gamma=values.gamma,
            lambda_=values.lam,
            clip_param=values.clip_param,
            entropy_coeff=values.entropy_coeff,
            vf_loss_coeff=values.vf_loss_coeff,
            grad_clip=values.grad_clip,
            model=model_cfg,
        )
        .resources(num_gpus=0)
    )
    return config


def iter_sweep_configs(base: RangeRunConfig) -> Iterable[Tuple[str, RangeRunConfig]]:
    """Yield a set of configs that hit the min/max validation ranges."""

    seeds = {
        "lower_bounds": dict(
            num_workers=0,
            num_envs_per_worker=1,
            rollout_fragment_length=50,
            train_batch_size=1000,
            sgd_minibatch_size=64,
            num_sgd_iter=1,
            lr=0.00001,
            gamma=0.9,
            lam=0.8,
            clip_param=0.1,
            entropy_coeff=0.0,
            vf_loss_coeff=0.1,
            grad_clip=0.1,
        ),
        "upper_bounds": dict(
            num_workers=64,
            num_envs_per_worker=16,
            rollout_fragment_length=1000,
            train_batch_size=100_000,
            sgd_minibatch_size=1024,
            num_sgd_iter=50,
            lr=0.01,
            gamma=0.9999,
            lam=1.0,
            clip_param=0.3,
            entropy_coeff=0.1,
            vf_loss_coeff=1.0,
            grad_clip=10.0,
        ),
        "divisible_batch": dict(
            train_batch_size=max(2048, base.sgd_minibatch_size * 4),
            sgd_minibatch_size=base.sgd_minibatch_size,
            rollout_fragment_length=max(50, base.rollout_fragment_length),
        ),
        "lstm_long_sequence": dict(
            use_lstm=True,
            lstm_cell_size=base.lstm_cell_size,
            lstm_num_layers=base.lstm_num_layers,
            lstm_max_seq_len=128,
        ),
    }

    for label, overrides in seeds.items():
        cfg = asdict(base)
        cfg.update(overrides)
        yield label, RangeRunConfig(**cfg)


def run_trial(name: str, values: RangeRunConfig) -> None:
    print(f"\n=== Running {name} with settings: {values}")
    config = build_rllib_config(values)
    algo = config.build()
    try:
        for idx in range(values.iterations):
            result = algo.train()
            training_iteration = result.get("training_iteration", idx + 1)
            reward = result.get("episode_reward_mean")
            print(
                f"Iteration {training_iteration}: reward_mean={reward}",
                flush=True,
            )
    finally:
        with contextlib.suppress(Exception):
            algo.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", action="store_true", help="run predefined boundary sweeps")
    parser.add_argument("--iterations", type=int, default=1, help="training iterations per run")
    parser.add_argument("--num-workers", type=int, default=0, dest="num_workers")
    parser.add_argument("--num-envs-per-worker", type=int, default=1, dest="num_envs_per_worker")
    parser.add_argument("--rollout-fragment-length", type=int, default=200, dest="rollout_fragment_length")
    parser.add_argument("--train-batch-size", type=int, default=4000, dest="train_batch_size")
    parser.add_argument("--sgd-minibatch-size", type=int, default=256, dest="sgd_minibatch_size")
    parser.add_argument("--num-sgd-iter", type=int, default=10, dest="num_sgd_iter")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda", type=float, default=0.95, dest="lam")
    parser.add_argument("--clip-param", type=float, default=0.2, dest="clip_param")
    parser.add_argument("--entropy-coeff", type=float, default=0.0, dest="entropy_coeff")
    parser.add_argument("--vf-loss-coeff", type=float, default=1.0, dest="vf_loss_coeff")
    parser.add_argument("--grad-clip", type=float, default=0.5, dest="grad_clip")
    parser.add_argument("--batch-mode", choices=["truncate_episodes", "complete_episodes"], default="truncate_episodes")
    parser.add_argument("--framework", choices=["torch", "tf2"], default="torch")
    parser.add_argument("--activation", choices=["relu", "tanh", "elu"], default="tanh")
    parser.add_argument("--use-lstm", action="store_true")
    parser.add_argument("--lstm-cell-size", type=int, default=256, dest="lstm_cell_size")
    parser.add_argument("--lstm-num-layers", type=int, default=1, dest="lstm_num_layers")
    parser.add_argument("--lstm-max-seq-len", type=int, default=128, dest="lstm_max_seq_len")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = RangeRunConfig(**vars(args))
    ray.init(_temp_dir=str(ray._private.utils.get_user_temp_dir()))
    try:
        if args.sweep:
            for name, cfg in iter_sweep_configs(base):
                run_trial(name, cfg)
        else:
            run_trial("custom", base)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
