"""Utility script to convert RLlib checkpoints to ONNX format.

This script exports RLlib trained policies to ONNX format, compatible with 
the Godot RL Agents integration. It follows the same export pattern as 
convert_sb3_to_onnx.py for Stable-Baselines3 models.

Examples:
    Export all policies from a specific checkpoint:
        python convert_rllib_onnx.py path/to/checkpoint_000008
    
    Export all policies with custom output directory:
        python convert_rllib_onnx.py path/to/checkpoint_000008 -o my_models
    
    Export only a specific policy:
        python convert_rllib_onnx.py path/to/checkpoint_000008 --policy left_policy

    Auto-find latest checkpoint in a directory:
        python convert_rllib_onnx.py path/to/training_run_dir

    Select a specific checkpoint number from a directory:
        python convert_rllib_to_onnx.py path/to/training_run_dir --checkpoint-number 12

    Use different ONNX opset version:
        python convert_rllib_onnx.py path/to/checkpoint_000008 --opset 15
"""
import argparse
import os
import pathlib
import glob
from typing import Optional
import torch
import onnx
from onnx import helper
from onnx.version_converter import convert_version

from godot_rl.core.godot_env import GodotEnv
from godot_rl.wrappers.petting_zoo_wrapper import GDRLPettingZooEnv
from godot_rl.wrappers.ray_wrapper import RayVectorGodotEnv
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from custom_models import register_rllib_models


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value must be a non-negative integer")
    return number


class RLlibOnnxablePolicy(torch.nn.Module):
    """Wrapper to make RLlib policy exportable to ONNX with proper input/output structure."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, obs, state_ins):
        """
        Forward pass matching the expected ONNX signature.
        Args:
            obs: Observation tensor
            state_ins: State input tensor (for recurrent policies, unused for feedforward)
        Returns:
            output: Action logits or values
            state_outs: State output tensor (passthrough for feedforward policies)
        """
        # Get the policy output
        try:
            # Try with dict input (some RLlib models expect this)
            output, _ = self.model({"obs": obs}, [], None)
        except:
            try:
                # Try with direct tensor input
                output, _ = self.model(obs, [], None)
            except:
                # Fallback to simple forward
                output = self.model(obs)
        
        # Return both output and state (state is just passed through for feedforward networks)
        return output, state_ins


def export_rllib_policy_to_onnx(
    policy,
    export_path: str,
    *,
    opset_version: int = 13,
    target_ir_version: int = 9,
) -> None:
    """
    Export an RLlib policy to ONNX format following the same pattern as SB3 export.
    
    Args:
        policy: RLlib policy to export
        export_path: Path where to save the ONNX model
        opset_version: Target ONNX opset version
        target_ir_version: Target ONNX IR version
    """
    # Get the model and move to CPU
    model = policy.model
    model.eval()
    model.cpu()
    
    # Create dummy input based on observation space
    obs_space = policy.observation_space
    if hasattr(obs_space, 'shape'):
        dummy_obs = torch.randn(1, *obs_space.shape).float()
    else:
        # Fallback: try to infer from model
        dummy_obs = torch.randn(1, policy.model.obs_space.shape[0]).float()
    
    # Create dummy state input (for compatibility with recurrent policies)
    dummy_state = torch.zeros(1).float()
    
    # Wrap the model in our ONNX-compatible wrapper
    onnxable_model = RLlibOnnxablePolicy(model)
    onnxable_model.eval()
    
    print(f"  Exporting to ONNX with opset {opset_version}...")
    
    # Export to ONNX with proper input/output names and dynamic axes
    # Disable dynamo to use the old stable exporter
    with torch.no_grad():
        torch.onnx.export(
            onnxable_model,
            args=(dummy_obs, dummy_state),
            f=export_path,
            opset_version=opset_version,
            input_names=["obs", "state_ins"],
            output_names=["output", "state_outs"],
            dynamic_axes={
                "obs": {0: "batch_size"},
                "state_ins": {0: "batch_size"},
                "output": {0: "batch_size"},
                "state_outs": {0: "batch_size"},
            },
            do_constant_folding=True,
            export_params=True,
            dynamo=False,  # Use old exporter for better compatibility
        )
    
    print(f"  Post-processing ONNX model...")
    postprocess_onnx_file(export_path, target_opset=opset_version, target_ir=target_ir_version)
    
    print(f"✓ Successfully exported to: {export_path}")


def postprocess_onnx_file(export_path: str, *, target_opset: int, target_ir: int) -> None:
    """
    Post-process the exported ONNX file to ensure compatibility.
    This matches the post-processing done in convert_sb3_to_onnx.py
    """
    model = onnx.load(export_path)
    current_opsets = [entry.version for entry in model.opset_import]
    needs_opset_downgrade = any(version > target_opset for version in current_opsets)

    if needs_opset_downgrade:
        try:
            print(f"  Converting to opset {target_opset}...")
            model = convert_version(model, target_opset)
        except Exception as exc:
            raise RuntimeError(f"Failed to convert ONNX model to opset {target_opset}: {exc}") from exc

    if target_ir and model.ir_version > target_ir:
        print(f"  Setting IR version to {target_ir}...")
        model.ir_version = target_ir

    ensure_state_input_name(model)
    ensure_state_output_exists(model)

    onnx.save(model, export_path)


def ensure_state_input_name(model: onnx.ModelProto) -> None:
    """Ensure the state input is named 'state_ins' (not 'state_outs')."""
    input_names = {value_info.name for value_info in model.graph.input}
    
    print(f"  Current input names: {input_names}")
    
    if "state_ins" in input_names:
        print("  'state_ins' already exists, no renaming needed")
        return

    target_input = None
    for value_info in model.graph.input:
        if value_info.name == "state_outs":
            target_input = value_info
            break

    if target_input is None:
        print("  Warning: Could not find 'state_outs' input to rename to 'state_ins'")
        return

    print(f"  Renaming input 'state_outs' to 'state_ins'")
    rename_input_only(model, target_input, "state_ins")


def rename_input_only(model: onnx.ModelProto, input_value_info: onnx.ValueInfoProto, new_name: str) -> None:
    """Rename an input and update all references to it."""
    old_name = input_value_info.name
    input_value_info.name = new_name

    for node in model.graph.node:
        node.input[:] = [new_name if name == old_name else name for name in node.input]


def ensure_state_output_exists(model: onnx.ModelProto) -> None:
    """Ensure state_outs exists and is properly connected."""
    output_names = {value_info.name for value_info in model.graph.output}
    if "state_outs" not in output_names:
        return

    produced_outputs = {name for node in model.graph.node for name in node.output}
    if "state_outs" in produced_outputs:
        return

    # Add Identity node to pass through state_ins to state_outs
    print("  Adding state passthrough node...")
    model.graph.node.append(
        helper.make_node("Identity", inputs=["state_ins"], outputs=["state_outs"], name="StatePassthrough")
    )

def find_latest_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """
    Find the latest checkpoint in a directory.

    Args:
        checkpoint_dir: Directory containing checkpoints or a specific checkpoint path
        
    Returns:
        Path to the latest checkpoint
    """
    
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
    
    # If it's already a checkpoint directory, return it
    if checkpoint_dir.name.startswith("checkpoint_"):
        return checkpoint_dir
    
    # Otherwise, find the latest checkpoint in the directory
    checkpoint_paths = list(checkpoint_dir.glob("checkpoint_*"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    print(f"Found {len(checkpoint_paths)} checkpoints in {checkpoint_dir}")
    # Sort by checkpoint number
    checkpoint_paths.sort(key=lambda p: int(p.name.split("_")[-1]))
    return checkpoint_paths[-1]


def find_checkpoint_by_number(checkpoint_dir: pathlib.Path, checkpoint_number: int) -> pathlib.Path:
    """Locate a specific checkpoint directory by number."""
    if checkpoint_number < 0:
        raise ValueError("Checkpoint number must be non-negative")

    for name in (f"checkpoint_{checkpoint_number:06d}", f"checkpoint_{checkpoint_number}"):
        candidate = checkpoint_dir / name
        if candidate.is_dir():
            return candidate

    matches = []
    for path in checkpoint_dir.glob("checkpoint_*"):
        if not path.is_dir():
            continue
        suffix = path.name.split("checkpoint_")[-1]
        try:
            number = int(suffix)
        except ValueError:
            continue
        if number == checkpoint_number:
            matches.append(path)

    if matches:
        matches.sort()
        return matches[0]

    raise FileNotFoundError(
        f"Checkpoint checkpoint_{checkpoint_number:06d} not found in {checkpoint_dir}"
    )


def resolve_checkpoint_path(
    checkpoint_path: pathlib.Path, checkpoint_number: Optional[int]
) -> pathlib.Path:
    """Resolve the actual checkpoint directory to export."""

    is_specific_checkpoint = checkpoint_path.name.startswith("checkpoint_")

    if is_specific_checkpoint:
        if checkpoint_number is not None:
            print(
                "Ignoring --checkpoint-number because a specific checkpoint directory was provided."
            )
        return checkpoint_path

    if checkpoint_number is not None:
        print(f"Selecting checkpoint number: {checkpoint_number}")
        return find_checkpoint_by_number(checkpoint_path, checkpoint_number)

    return find_latest_checkpoint(checkpoint_path)

def setup_env_registration(is_multiagent: bool) -> None:
    # Register env
    env_name = "godot"

    def env_creator(env_config):
        index = (
            env_config.worker_index
            + env_config.vector_index
        )
        port = index + GodotEnv.DEFAULT_PORT
        seed = index
        if is_multiagent:
            return ParallelPettingZooEnv(
                GDRLPettingZooEnv(config=env_config, port=port, seed=seed)
            )
        else:
            return RayVectorGodotEnv(config=env_config, port=port, seed=seed)

    tune.register_env(env_name, env_creator)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        type=pathlib.Path,
        help="Path to the RLlib checkpoint directory or parent directory containing checkpoints. "
             "If a parent directory is provided, the latest checkpoint will be used automatically.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        help="Output directory for ONNX models. Defaults to './onnx_models'.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="Target ONNX opset version. Default: 13",
    )
    parser.add_argument(
        "--ir-version",
        type=int,
        default=9,
        help="Set the ONNX IR version on the exported model. Default: 9",
    )
    parser.add_argument(
        "--multiagent",
        action="store_true",
        default=True,
        help="Whether the environment is multiagent. Default: True",
    )
    parser.add_argument(
        "--no-multiagent",
        dest="multiagent",
        action="store_false",
        help="Disable multiagent mode.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        help="Export only a specific policy by ID. If not specified, all policies are exported.",
    )
    parser.add_argument(
        "--checkpoint-number",
        type=non_negative_int,
        help="Select a specific checkpoint number when providing a directory of checkpoints.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        help="Prefix for naming exported ONNX files <prefix>_<policy_id>.onnx. Default is no prefix.",
    )
    return parser.parse_args()


def main():
    """Main function to export RLlib policies to ONNX."""
    args = parse_args()
    
    checkpoint_input = args.checkpoint_path
    if not checkpoint_input.exists():
        raise FileNotFoundError(f"Path not found: {checkpoint_input}")
    if not checkpoint_input.is_dir():
        raise NotADirectoryError(
            f"Checkpoint path must be a directory: {checkpoint_input}"
        )

    setup_env_registration(args.multiagent)

    checkpoint_path = resolve_checkpoint_path(checkpoint_input, args.checkpoint_number)
    print(f"Using checkpoint: {checkpoint_path}")

    abs_path = checkpoint_path.resolve()
    print(f"Loading checkpoint from: {abs_path}")

    # Load the algorithm
    algo = Algorithm.from_checkpoint(str(abs_path))

    # Get policy IDs from the algorithm
    # Note: RLlib API varies across versions, so we try multiple approaches
    policy_ids = []
    try:
        # Try to get policies from workers (older RLlib)
        policy_ids = list(algo.env_runner.policy_map.keys())  # type: ignore
    except:
        try:
            # Try from config
            config_dict = algo.config if isinstance(algo.config, dict) else {}
            multiagent_config = config_dict.get("multiagent", {}) if config_dict else {}  # type: ignore
            policies_config = multiagent_config.get("policies", {}) if isinstance(multiagent_config, dict) else {}  # type: ignore
            policy_ids = list(policies_config.keys()) if policies_config else []
            if not policy_ids:
                policy_ids = ["default_policy"]
        except:
            policy_ids = ["default_policy"]

    print(f"Found policies: {policy_ids}")
    
    # Filter by specific policy if requested
    if args.policy:
        if args.policy not in policy_ids:
            raise ValueError(f"Policy '{args.policy}' not found. Available policies: {policy_ids}")
        policy_ids = [args.policy]
        print(f"Exporting only policy: {args.policy}")

    # Setup export directory
    export_dir = args.output or pathlib.Path("onnx_models")
    export_dir = pathlib.Path(export_dir).resolve()
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export settings
    opset_version = args.opset
    target_ir_version = args.ir_version
    prefix = args.prefix + "_" if args.prefix else ""

    # Export each policy
    for policy_id in policy_ids:
        print(f"\nExporting policy: {policy_id}")
        try:
            policy = algo.get_policy(policy_id)
            export_path = export_dir / f"{prefix}{policy_id}.onnx"

            export_rllib_policy_to_onnx(
                policy,
                str(export_path),
                opset_version=opset_version,
                target_ir_version=target_ir_version,
            )
        except Exception as e:
            print(f"✗ Failed to export policy '{policy_id}': {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nExport completed! ONNX files saved to: {export_dir}")


if __name__ == "__main__":
    register_rllib_models()
    main()
