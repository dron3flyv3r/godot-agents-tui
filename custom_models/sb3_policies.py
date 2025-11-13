"""Custom SB3 feature extractors that can be selected from the TUI."""

from __future__ import annotations

from typing import Sequence

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor

from .common import Conv1dBody, FeatureBody, GrnBody, LstmBody

SUPPORTED_POLICY_TYPES = ("mlp", "cnn", "lstm", "grn")


class _AdaptiveExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, body: FeatureBody):
        super().__init__(observation_space, features_dim=body.output_dim)
        self.flatten = FlattenExtractor(observation_space)
        self.body = body

    def forward(self, observations):  # type: ignore[override]
        flat = self.flatten(observations)
        return self.body(flat)


class CnnFeatureExtractor(_AdaptiveExtractor):
    def __init__(self, observation_space: spaces.Space, channels: Sequence[int]):
        flatten = FlattenExtractor(observation_space)
        body = Conv1dBody(flatten.features_dim, channels)
        super().__init__(observation_space, body)


class LstmFeatureExtractor(_AdaptiveExtractor):
    def __init__(self, observation_space: spaces.Space, hidden_size: int, num_layers: int):
        flatten = FlattenExtractor(observation_space)
        body = LstmBody(flatten.features_dim, hidden_size, num_layers)
        super().__init__(observation_space, body)


class GrnFeatureExtractor(_AdaptiveExtractor):
    def __init__(self, observation_space: spaces.Space, hidden_size: int):
        flatten = FlattenExtractor(observation_space)
        body = GrnBody(flatten.features_dim, hidden_size)
        super().__init__(observation_space, body)


def build_policy_kwargs(
    *,
    policy_type: str,
    mlp_layers: Sequence[int],
    cnn_channels: Sequence[int],
    lstm_hidden_size: int,
    lstm_num_layers: int,
    grn_hidden_size: int,
) -> dict:
    policy_type = policy_type.lower()
    if policy_type not in SUPPORTED_POLICY_TYPES:
        raise ValueError(f"Unsupported policy type '{policy_type}'.")

    kwargs: dict = {"net_arch": [dict(pi=list(mlp_layers), vf=list(mlp_layers))]}
    if policy_type == "cnn":
        kwargs["features_extractor_class"] = CnnFeatureExtractor
        kwargs["features_extractor_kwargs"] = {"channels": list(cnn_channels)}
    elif policy_type == "lstm":
        kwargs["features_extractor_class"] = LstmFeatureExtractor
        kwargs["features_extractor_kwargs"] = {
            "hidden_size": int(lstm_hidden_size),
            "num_layers": int(lstm_num_layers),
        }
    elif policy_type == "grn":
        kwargs["features_extractor_class"] = GrnFeatureExtractor
        kwargs["features_extractor_kwargs"] = {"hidden_size": int(grn_hidden_size)}

    return kwargs
