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
        self.include_prev_actions = bool(cfg.get("include_prev_actions", True))
        obs_dim = _infer_input_dim(obs_space)
        self._prev_action_dim = _infer_input_dim(action_space) if self.include_prev_actions else 0
        body = LstmBody(obs_dim + self._prev_action_dim, hidden_size, num_layers)
        super().__init__(obs_space, action_space, num_outputs, model_config, name, body, fcnet)

    def get_initial_state(self):  # type: ignore[override]
        if not isinstance(self.body, LstmBody):
            raise RuntimeError("get_initial_state() called on non-LSTM model.")
        num_layers = self.body.lstm.num_layers
        hidden_size = self.body.lstm.hidden_size
        zeros = torch.zeros(num_layers, hidden_size)
        return [zeros, zeros.clone()]

    def forward(self, input_dict, state, seq_lens):  # type: ignore[override]
        if not isinstance(self.body, LstmBody):
            raise RuntimeError("forward() called on non-LSTM model.")
        flat = flatten_observation(input_dict["obs"])
        flat = self._append_prev_actions(flat, input_dict.get("prev_actions"))
        batch_size = len(seq_lens) if seq_lens is not None and len(seq_lens) > 0 else flat.size(0)
        if batch_size == 0:
            return flat, state

        total = flat.size(0)
        if total % batch_size != 0:
            raise RuntimeError(
                f"Batch/time reshape mismatch: got total frames {total} and batch size {batch_size}"
            )
        max_seq_len = max(1, total // batch_size)
        seq = flat.view(batch_size, max_seq_len, -1)

        lstm_state = None
        if state and len(state) == 2:
            h_state = self._reshape_state_for_lstm(state[0], batch_size)
            c_state = self._reshape_state_for_lstm(state[1], batch_size)
            lstm_state = (h_state, c_state)

        out, (h_out, c_out) = self.body.forward_with_state(seq, lstm_state)
        # Flatten back to [B*T, H] so logits align with the incoming flattened obs/actions.
        features = out.reshape(-1, out.size(-1))
        features = self.head(features)
        self._last_features = features
        logits = self.policy_branch(features)
        new_state = [
            h_out.permute(1, 0, 2).contiguous(),
            c_out.permute(1, 0, 2).contiguous(),
        ]
        return logits, new_state

    def _append_prev_actions(
        self, flat_obs: torch.Tensor, prev_actions: torch.Tensor | None
    ) -> torch.Tensor:
        if not self.include_prev_actions or self._prev_action_dim == 0:
            return flat_obs

        if prev_actions is None:
            prev_flat = torch.zeros(
                flat_obs.size(0),
                self._prev_action_dim,
                dtype=flat_obs.dtype,
                device=flat_obs.device,
            )
        else:
            prev_tensor = prev_actions.float()
            if prev_tensor.dim() == 1:
                prev_tensor = prev_tensor.unsqueeze(-1)
            prev_flat = prev_tensor.view(flat_obs.size(0), -1)
            if prev_flat.size(1) != self._prev_action_dim:
                raise RuntimeError(
                    f"Expected prev_actions dimension {self._prev_action_dim}, "
                    f"got {prev_flat.size(1)}."
                )
        return torch.cat([flat_obs, prev_flat], dim=-1)

    def _reshape_state_for_lstm(self, state: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert RLlib state tensors to (num_layers, batch, hidden_size)."""
        if not isinstance(self.body, LstmBody):
            raise RuntimeError("_reshape_state_for_lstm() called on non-LSTM model.")
        num_layers = self.body.lstm.num_layers
        hidden_size = self.body.lstm.hidden_size

        if state.dim() == 2:
            tensor = state
            if tensor.size(0) != batch_size:
                tensor = tensor.expand(batch_size, -1)
            tensor = tensor.unsqueeze(1)
        elif state.dim() == 3:
            tensor = state
            if tensor.size(0) != batch_size:
                tensor = tensor.expand(batch_size, -1, -1)
        else:
            tensor = state.view(batch_size, -1, hidden_size)

        if tensor.size(1) != num_layers:
            tensor = tensor.expand(batch_size, num_layers, hidden_size)

        return tensor.permute(1, 0, 2).contiguous()


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
        subspaces = space.spaces
        if isinstance(subspaces, dict):
            values = subspaces.values()
        else:
            values = subspaces
        return sum(_infer_input_dim(subspace) for subspace in values)
    if hasattr(space, "n"):
        return 1
    raise ValueError("Unsupported observation space for adaptive model")


def register_rllib_models() -> None:
    """Register all custom policy models with RLlib."""

    ModelCatalog.register_custom_model("tui_cnn", CnnPolicyModel)
    ModelCatalog.register_custom_model("tui_lstm", LstmPolicyModel)
    ModelCatalog.register_custom_model("tui_grn", GrnPolicyModel)
