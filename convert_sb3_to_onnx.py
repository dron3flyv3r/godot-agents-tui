"""Utility script to convert a saved Stable-Baselines3 model (.zip) to ONNX."""
import argparse
import pathlib
from typing import Dict, Type

import onnx
from onnx import helper
import torch
from gymnasium import spaces
from onnx.version_converter import convert_version
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.save_util import load_from_zip_file

from godot_rl.wrappers.onnx.stable_baselines_export import OnnxablePolicy, verify_onnx_export

# Mapping from algorithm id stored in the SB3 zip to the corresponding class
ALGO_CLASSES: Dict[str, Type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def resolve_algorithm(algo_name: str) -> Type[BaseAlgorithm]:
    try:
        return ALGO_CLASSES[algo_name.lower()]
    except KeyError as exc:
        known = ", ".join(sorted(ALGO_CLASSES))
        msg = f"Unsupported algorithm '{algo_name}'. Known algorithms: {known}."
        raise ValueError(msg) from exc


def extract_metadata(loaded: tuple) -> dict | None:
    for part in loaded:
        if isinstance(part, dict) and "policy_class" in part:
            return part
    return None


def candidate_algorithms(metadata: dict | None) -> list[str]:
    if not metadata:
        return list(ALGO_CLASSES)

    if metadata.get("replay_buffer_class") is not None:
        if metadata.get("target_entropy") is not None:
            return ["sac"]
        if metadata.get("policy_delay") is not None or metadata.get("target_noise_clip") is not None:
            return ["td3"]
        if metadata.get("action_noise") is not None:
            return ["ddpg", "td3"]
        return ["dqn"]

    if "n_epochs" in metadata or "clip_range" in metadata or "clip_range_vf" in metadata:
        return ["ppo"]

    return ["a2c"]


def load_model(model_path: pathlib.Path, algo_name: str | None) -> BaseAlgorithm:
    if algo_name:
        algo_cls = resolve_algorithm(algo_name)
        return algo_cls.load(model_path, device="cpu")

    loaded = load_from_zip_file(model_path)
    for part in loaded:
        if isinstance(part, dict) and "algo" in part:
            algo_cls = resolve_algorithm(str(part["algo"]))
            return algo_cls.load(model_path, device="cpu")

    metadata = extract_metadata(loaded)
    candidates = candidate_algorithms(metadata)

    errors: Dict[str, Exception] = {}
    for candidate in candidates:
        algo_cls = ALGO_CLASSES[candidate]
        try:
            return algo_cls.load(model_path, device="cpu")
        except Exception as exc:
            errors[candidate] = exc

    extra = "; ".join(f"{name}: {err}" for name, err in errors.items())
    hint = "Could not infer algorithm automatically. Please pass --algo."
    message = f"{extra}. {hint}" if extra else hint
    raise ValueError(message)


def export_model_to_onnx(
    model: BaseAlgorithm,
    export_path: str,
    *,
    opset_version: int,
    verify: bool,
    use_obs_array: bool,
    target_ir_version: int,
) -> None:
    obs_keys = ["obs"]
    policy = model.policy.to("cpu")

    dummy_input = None
    onnxable_model = None

    if isinstance(model, SAC):
        if not use_obs_array:
            raise ValueError("SAC export requires --use-obs-array.")
        onnxable_model = OnnxablePolicy(actor=policy.actor)
        dummy_input = torch.randn(1, *model.observation_space.shape) # type: ignore
    elif isinstance(model, PPO):
        onnxable_model = OnnxablePolicy(
            obs_keys,
            policy.features_extractor,
            policy.mlp_extractor,
            policy.action_net,
            policy.value_net,
            use_obs_array,
        )
        if use_obs_array:
            dummy_input = torch.unsqueeze(torch.tensor(model.observation_space.sample()), 0)
        else:
            dummy_input = dict(model.observation_space.sample())
            for key, value in dummy_input.items():
                dummy_input[key] = torch.from_numpy(value).unsqueeze(0)
            dummy_input = [value for value in dummy_input.values()]
    else:
        raise ValueError("ONNX export is currently supported only for PPO and SAC models.")

    torch.onnx.export(
        onnxable_model,
        args=(dummy_input, torch.zeros(1).float()),
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
    )

    postprocess_onnx_file(export_path, target_opset=opset_version, target_ir=target_ir_version)

    if verify and isinstance(model, PPO) and not isinstance(model.action_space, spaces.MultiDiscrete):
        verify_onnx_export(model, export_path, use_obs_array=use_obs_array)


def export_zip_to_onnx(
    model_path: pathlib.Path,
    export_path: pathlib.Path,
    algo_name: str | None,
    opset_version: int,
    verify: bool,
    use_obs_array: bool,
    target_ir_version: int,
) -> None:
    model = load_model(model_path, algo_name)
    export_model_to_onnx(
        model,
        str(export_path),
        opset_version=opset_version,
        verify=verify,
        use_obs_array=use_obs_array,
        target_ir_version=target_ir_version,
    )


def postprocess_onnx_file(export_path: str, *, target_opset: int, target_ir: int) -> None:
    model = onnx.load(export_path)
    current_opsets = [entry.version for entry in model.opset_import]
    needs_opset_downgrade = any(version > target_opset for version in current_opsets)

    if needs_opset_downgrade:
        try:
            model = convert_version(model, target_opset)
        except Exception as exc:
            raise RuntimeError(f"Failed to convert ONNX model to opset {target_opset}: {exc}") from exc

    if target_ir and model.ir_version > target_ir:
        model.ir_version = target_ir

    ensure_state_input_name(model)
    ensure_state_output_exists(model)

    onnx.save(model, export_path)


def ensure_state_input_name(model: onnx.ModelProto) -> None:
    input_names = {value_info.name for value_info in model.graph.input}
    if "state_ins" in input_names:
        return

    target_input = None
    for value_info in model.graph.input:
        if value_info.name == "state_outs":
            target_input = value_info
            break

    if target_input is None:
        raise RuntimeError(
            "Exported model is missing the expected 'state_ins' input. "
            "Please re-export with a newer godot-rl package or pass --use-obs-array."
        )

    rename_input_only(model, target_input, "state_ins")


def rename_input_only(model: onnx.ModelProto, input_value_info: onnx.ValueInfoProto, new_name: str) -> None:
    old_name = input_value_info.name
    input_value_info.name = new_name

    for node in model.graph.node:
        node.input[:] = [new_name if name == old_name else name for name in node.input]


def ensure_state_output_exists(model: onnx.ModelProto) -> None:
    output_names = {value_info.name for value_info in model.graph.output}
    if "state_outs" not in output_names:
        return

    produced_outputs = {name for node in model.graph.node for name in node.output}
    if "state_outs" in produced_outputs:
        return

    model.graph.node.append(
        helper.make_node("Identity", inputs=["state_ins"], outputs=["state_outs"], name="StatePassthrough")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_zip",
        type=pathlib.Path,
        help="Path to the saved Stable-Baselines3 model (.zip).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        help="Where to write the ONNX model. Defaults to the input name with an .onnx suffix.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=sorted(ALGO_CLASSES),
        help="Algorithm used to train the model. If omitted, the script tries to infer it from the zip metadata.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=13,
        help="Target ONNX opset. Values below 13 may not be supported by the converter.",
    )
    parser.add_argument(
        "--ir-version",
        type=int,
        default=9,
        help="Set the ONNX IR version on the exported model.",
    )
    parser.add_argument(
        "--use-obs-array",
        action="store_true",
        help="Export using a single array observation (required for SAC).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip running the post-export ONNX verification step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_zip
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    export_path = args.output or model_path.with_suffix(".onnx")
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_zip_to_onnx(
        model_path,
        export_path,
        args.algo,
        opset_version=args.opset,
        verify=not args.no_verify,
        use_obs_array=args.use_obs_array,
        target_ir_version=args.ir_version,
    )
    print(f"ONNX model exported to {export_path.resolve()}")


if __name__ == "__main__":
    main()
