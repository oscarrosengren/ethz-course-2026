"""Model definitions for SO-100 imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch."""
        pred = self.forward(state)
        return nn.functional.mse_loss(pred, action_chunk)

    @abc.abstractmethod
    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""
        return self.forward(state)


# TODO: Students implement ObstaclePolicy here.
class ObstaclePolicy(BasePolicy):

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int, hidden_dim: int = 256):
        super().__init__(state_dim, action_dim, chunk_size)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        
        flat_actions = self.mlp(state) 
        
        
        action_chunks = flat_actions.view(-1, self.chunk_size, self.action_dim)
        return action_chunks

    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:

        predicted_chunk = self.forward(state)
        
        return nn.functional.mse_loss(predicted_chunk, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


class MultiTaskPolicy(BasePolicy):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        processed_input_dim = 10
        self.network = nn.Sequential(
            nn.Linear(processed_input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def _process_state(self, state: torch.Tensor, augment: bool = False) -> torch.Tensor:
        
        p_red = state[:, 0:3]
        p_green = state[:, 7:10]
        p_blue = state[:, 14:17]
        p_ee = state[:, 21:24]
        p_bin = state[:, 24:27]
        goal_oh = state[:, 27:30]
        gripper = state[:, 30:31]

        goal_oh_clean = torch.round(goal_oh)

        p_target = (
            goal_oh_clean[:, 0:1] * p_red
            + goal_oh_clean[:, 1:2] * p_green
            + goal_oh_clean[:, 2:3] * p_blue
        )
        # ------------------------

        if augment and self.training:
            # did not work as intended - nosie added
            p_target = p_target + torch.randn_like(p_target) * 0.00
            p_bin = p_bin + torch.randn_like(p_bin) * 0.00

        rel_target = p_target - p_ee
        rel_bin = p_bin - p_ee

       
        return torch.cat([rel_target, rel_bin, goal_oh_clean, gripper], dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        processed = self._process_state(state, augment=False)
        flat_actions = self.network(processed)
        return flat_actions.view(-1, self.chunk_size, self.action_dim)
    
    def compute_loss(self, state: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        processed = self._process_state(state, augment=True)
        flat_actions = self.network(processed)
        pred = flat_actions.view(-1, self.chunk_size, self.action_dim)
        return nn.functional.mse_loss(pred, action_chunk)

    def sample_actions(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


PolicyType: TypeAlias = Literal["obstacle", "multitask"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    **kwargs
) -> BasePolicy:
 
    hidden_dim = 256 
    
    if policy_type == "obstacle":
        return ObstaclePolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            hidden_dim=hidden_dim
        )
    if policy_type == "multitask":
        return MultiTaskPolicy(
            action_dim=action_dim,
            state_dim=state_dim,
            chunk_size=chunk_size,
            hidden_dim=512
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
