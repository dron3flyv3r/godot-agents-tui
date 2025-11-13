"""RLlib custom Torch models mirroring the selectable policy types."""

from __future__ import annotations

from typing import Sequence

import torch
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
from torch import nn

from .common import Conv1dBody, GrnBody, LstmBody, MLPBody, FeatureBody, flatten_observation


def _build_head(input_dim: int, layers: Sequence[int]) -> tuple[nn.Module, int]:
    body = MLPBody(input_dim, layers)
    return body, body.output_dim


class _AdaptiveTorchModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name,
        body: FeatureBody,
        head_layers: Sequence[int],
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.body = body
        head, head_dim = _build_head(body.output_dim, head_layers)
        self.head = head
        self.policy_branch = nn.Linear(head_dim, num_outputs)
        self.value_branch = nn.Linear(head_dim, 1)
        self._last_features: torch.Tensor | None = None

    def forward(self, input_dict, state, seq_lens):  # type: ignore[override]
        obs = flatten_observation(input_dict["obs"])
        features = self.body(obs)
        features = self.head(features)
        self._last_features = features
        logits = self.policy_branch(features)
        return logits, state

    def value_function(self):  # type: ignore[override]
        if self._last_features is None:
            raise RuntimeError("value_function() called before forward().")
        return torch.squeeze(self.value_branch(self._last_features), -1)


class CnnPolicyModel(_AdaptiveTorchModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs,
    ):
        cfg = {**model_config.get("custom_model_config", {}), **custom_model_kwargs}
        channels: Sequence[int] = cfg.get("channels", [32, 64, 64])
        fcnet: Sequence[int] = cfg.get("fcnet_hiddens", [256, 256])
        body = Conv1dBody(_infer_input_dim(obs_space), channels)
        super().__init__(obs_space, action_space, num_outputs, model_config, name, body, fcnet)


class LstmPolicyModel(_AdaptiveTorchModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs,
    ):
        cfg = {**model_config.get("custom_model_config", {}), **custom_model_kwargs}
        hidden_size = int(cfg.get("hidden_size", 256))
        num_layers = int(cfg.get("num_layers", 1))
        fcnet: Sequence[int] = cfg.get("fcnet_hiddens", [256, 256])
        body = LstmBody(_infer_input_dim(obs_space), hidden_size, num_layers)
        super().__init__(obs_space, action_space, num_outputs, model_config, name, body, fcnet)


class GrnPolicyModel(_AdaptiveTorchModel):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **custom_model_kwargs,
    ):
        cfg = {**model_config.get("custom_model_config", {}), **custom_model_kwargs}
        hidden_size = int(cfg.get("hidden_size", 256))
        fcnet: Sequence[int] = cfg.get("fcnet_hiddens", [256, 256])
        body = GrnBody(_infer_input_dim(obs_space), hidden_size)
        super().__init__(obs_space, action_space, num_outputs, model_config, name, body, fcnet)


def _infer_input_dim(space) -> int:
    import numpy as np

    if hasattr(space, "shape") and space.shape is not None:
        return int(np.prod(space.shape))
    if hasattr(space, "spaces"):
        return sum(_infer_input_dim(subspace) for subspace in space.spaces.values())
    raise ValueError("Unsupported observation space for adaptive model")


def register_rllib_models() -> None:
    """Register all custom policy models with RLlib."""

    ModelCatalog.register_custom_model("tui_cnn", CnnPolicyModel)
    ModelCatalog.register_custom_model("tui_lstm", LstmPolicyModel)
    ModelCatalog.register_custom_model("tui_grn", GrnPolicyModel)
