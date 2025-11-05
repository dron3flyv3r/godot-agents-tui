#!/usr/bin/env python3
"""
Demo training script to test output streaming and metrics display.
This script simulates a training process with live output and emits metrics
in the format expected by the controller.
"""

import time
import sys
import json
import random
from datetime import datetime, timezone

METRIC_PREFIX = "@METRIC "

# ========================================
# CONFIGURATION: Change this to test different numbers of policies
NUM_POLICIES = 12  # Change this number (e.g., 2, 5, 10, 12, etc.)
# ========================================

def emit_metric(iteration, timestep, episodes):
    """Emit a metric in the format expected by the controller."""
    # Simulate improving metrics over time
    base_reward = -10.0 + (iteration * 0.5)
    reward_variance = max(0.5, 3.0 - (iteration * 0.1))
    
    episode_reward_mean = base_reward + random.uniform(-0.5, 0.5)
    episode_reward_min = episode_reward_mean - random.uniform(0.5, reward_variance)
    episode_reward_max = episode_reward_mean + random.uniform(0.5, reward_variance)
    
    episode_len = 400 + random.randint(-50, 50)
    throughput = 800 + random.randint(-100, 200)
    
    # Generate policies dynamically based on NUM_POLICIES
    policies = {}
    policy_names = []
    
    if NUM_POLICIES == 2:
        policy_names = ["left_policy", "right_policy"]
    else:
        policy_names = [f"policy_{i+1}" for i in range(NUM_POLICIES)]
    
    for policy_name in policy_names:
        policy_reward = base_reward + random.uniform(-1.0, 1.0)
        policies[policy_name] = {
            "reward_mean": round(policy_reward, 4),
            "reward_min": round(policy_reward - random.uniform(0.5, 2.0), 4),
            "reward_max": round(policy_reward + random.uniform(0.5, 2.0), 4),
            "episode_len_mean": float(episode_len + random.randint(-20, 20)),
            "completed_episodes": random.randint(4, 8),
            "learner": {
                "cur_kl_coeff": round(random.uniform(0.18, 0.22), 4),
                "cur_lr": 0.0003,
                "total_loss": round(random.uniform(0.5, 1.2), 4),
                "policy_loss": round(random.uniform(-0.02, 0.01), 6),
                "vf_loss": round(random.uniform(0.8, 2.0), 4),
                "vf_explained_var": round(random.uniform(0.05, 0.25), 4),
                "kl": round(random.uniform(0.005, 0.02), 6),
                "entropy": round(random.uniform(0.9, 1.2), 4),
                "entropy_coeff": 0.0001,
                "grad_gnorm": round(random.uniform(0.4, 0.8), 4),
            },
            "custom_metrics": {
                "action_rate": round(random.uniform(0.5, 0.9), 3),
                "success_rate": round(random.uniform(0.3, 0.7), 3),
            }
        }
    
    payload = {
        "source": "demo",
        "kind": "iteration",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_iteration": iteration,
        "timesteps_total": timestep,
        "time_this_iter_s": random.uniform(2.0, 4.0),
        "time_total_s": iteration * 3.2,
        "episodes_total": episodes,
        "episodes_this_iter": random.randint(8, 15),
        "episode_reward_mean": round(episode_reward_mean, 4),
        "episode_reward_min": round(episode_reward_min, 4),
        "episode_reward_max": round(episode_reward_max, 4),
        "episode_len_mean": float(episode_len),
        "num_env_steps_sampled": timestep,
        "num_env_steps_trained": timestep,
        "num_agent_steps_sampled": timestep * 2,
        "num_agent_steps_trained": timestep * 2,
        "env_steps_this_iter": 2048,
        "env_throughput": float(throughput),
        "num_workers": 4,
        "custom_metrics": {
            "total_interactions": round(random.uniform(100, 500), 2),
            "avg_performance": round(random.uniform(0.4, 0.8), 3),
        },
        "policies": policies
    }
    
    print(f"{METRIC_PREFIX}{json.dumps(payload)}", flush=True)

def main():
    print("Initializing demo training environment...")
    time.sleep(0.5)
    
    print("Loading virtual environment...")
    time.sleep(0.5)
    
    print("Setting up simulated PPO algorithm...")
    time.sleep(0.5)
    
    print(f"Creating {NUM_POLICIES} policies for multi-agent training")
    if NUM_POLICIES == 2:
        print("  - left_policy")
        print("  - right_policy")
    else:
        for i in range(NUM_POLICIES):
            print(f"  - policy_{i+1}")
    time.sleep(0.5)
    
    print("\n" + "="*60)
    print(f"Starting training with {NUM_POLICIES} policies and metrics emission...")
    print("="*60 + "\n")
    
    iterations = 50
    timestep = 0
    episodes = 0
    
    for iteration in range(1, iterations + 1):
        timestep += 2048
        episodes += random.randint(8, 15)
        
        print(f"Iteration {iteration}/{iterations} - Timesteps: {timestep}")
        
        # Emit metric for this iteration
        emit_metric(iteration, timestep, episodes)
        
        # Simulate some training output
        if iteration % 5 == 0:
            print(f"  Checkpoint saved at iteration {iteration}")
        
        if iteration % 10 == 0:
            avg_reward = -10.0 + (iteration * 0.5)
            print(f"  Progress: Avg reward improving to {avg_reward:.2f}")
        
        # Simulate training time
        time.sleep(0.8)
    
    print("\n" + "="*60)
    print("Demo training completed successfully!")
    print(f"Final timesteps: {timestep}")
    print(f"Final episodes: {episodes}")
    print("Metrics were emitted in RLlib format for controller testing")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo training interrupted by user!", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during demo: {e}", file=sys.stderr)
        sys.exit(1)
