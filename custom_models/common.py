"""Shared neural network helpers used by SB3 and RLlib integrations."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def _activation() -> nn.Module:
    return nn.ReLU()


class FeatureBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.output_dim = 0


class MLPBody(FeatureBody):
    def __init__(self, input_dim: int, layers: Sequence[int]):
        super().__init__()
        modules: list[nn.Module] = []
        last = input_dim
        for size in layers:
            modules.append(nn.Linear(last, size))
            modules.append(_activation())
            last = size
        if modules:
            self.net = nn.Sequential(*modules)
            output_dim = last
        else:
            self.net = nn.Identity()
            output_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class Conv1dBody(FeatureBody):
    def __init__(self, input_dim: int, channels: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        last_channels = 1
        for size in channels:
            layers.append(
                nn.Conv1d(
                    in_channels=last_channels,
                    out_channels=size,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(_activation())
            last_channels = size
        if not layers:
            layers.append(nn.Identity())
        self.net = nn.Sequential(*layers)
        self.output_dim = last_channels * input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        seq = x.unsqueeze(1)
        features = self.net(seq)
        return torch.flatten(features, start_dim=1)


class LstmBody(FeatureBody):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=max(1, num_layers),
            batch_first=True,
        )
        self.output_dim = hidden_size

    def _as_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input is shaped [B, T, F] for the LSTM."""

        if x.dim() == 2:
            return x.unsqueeze(1)
        if x.dim() == 3:
            return x
        # Fallback: flatten everything but the batch dim, keep a single time step.
        return torch.flatten(x, start_dim=1).unsqueeze(1)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
        seq = self._as_sequence(x)
        out, new_state = self.lstm(seq, state)
        features = out[:, -1, :]
        if state is None:
            return features
        return features, new_state

    def forward_with_state(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the LSTM and return the full sequence plus new hidden state."""

        seq = self._as_sequence(x)
        out, new_state = self.lstm(seq, state)
        return out, new_state


class GrnBody(FeatureBody):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        self.candidate = nn.Linear(input_dim, hidden_size)
        self.context = nn.Linear(input_dim, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ELU()
        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        candidate = self.activation(self.candidate(x))
        context = self.context(x)
        gate = torch.sigmoid(self.gate(candidate))
        return gate * candidate + (1 - gate) * context


def flatten_observation(obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten RLlib observation dictionaries into a single tensor."""

    if isinstance(obs, dict):
        flattened = [flatten_observation(value) for value in obs.values()]
        return torch.cat(flattened, dim=-1)
    return torch.flatten(obs.float(), start_dim=1)
