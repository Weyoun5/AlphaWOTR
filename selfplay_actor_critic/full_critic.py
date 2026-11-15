"""Full-state value network with auxiliary predictions.

The whole observation goes through the value network's trunk (per-region
graph embeddings + global features -> a 256-wide state embedding). Three
heads read that embedding:

  value_head   own-side value in [-1, 1]: the number used for decision credit.
  race_head    AUXILIARY: which of the four victory tracks ends the game and
               when (discrete hazards per clock, race_critic.RaceHead).
  future_head  AUXILIARY: how the race read-out of the position (stronghold
               control and strength around it, Fellowship progress,
               corruption, companions, hunt, dice, political track, forces)
               will have CHANGED at the side's first decision of the next
               turn and of the turn after.

The auxiliary heads exist to teach the trunk game dynamics from labels every
finished game already contains (KataGo-style); they are predictions, never
rewards. One game gives one win/loss bit but hundreds of future-state numbers.
Only value_head is used outside training.
"""
import copy

import torch
from torch import nn

from .race_critic import RaceFeatures, RaceHead, RaceLayout


class FullStateCritic(nn.Module):
    def __init__(self, trunk: nn.Module, side: str, layout: RaceLayout, *, horizon: int = 24, future_steps: int = 2):
        super().__init__()
        self.side = side
        self.trunk = trunk
        hidden = int(next(m for m in trunk.value_head.modules() if isinstance(m, nn.Linear)).in_features)
        # Starts as an exact copy of the trunk's own value head, so a critic
        # built from an existing full-state value network computes the same values.
        self.value_head = copy.deepcopy(trunk.value_head)
        self.race_head = RaceHead(hidden, side, hidden=64, horizon=horizon)
        self.race_features = RaceFeatures(layout)
        self.future_steps = int(future_steps)
        self.future_head = nn.Linear(hidden, self.future_steps * layout.feature_count)
        nn.init.zeros_(self.future_head.weight)
        nn.init.zeros_(self.future_head.bias)

    @property
    def layout(self) -> RaceLayout:
        return self.race_features.layout()

    def forward_backbone(self, x: torch.Tensor):
        return self.trunk.forward_backbone(x)[0], None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.forward_backbone(x)[0]).reshape(-1)

    def future(self, hidden: torch.Tensor) -> torch.Tensor:
        """[N, future_steps, features]: predicted change of the race read-out."""
        return self.future_head(hidden).reshape(hidden.shape[0], self.future_steps, -1)

    @staticmethod
    def is_full_state(state: dict) -> bool:
        return any(k.startswith("trunk.") for k in state)

    @staticmethod
    def from_state(trunk: nn.Module, state: dict, side: str, *, scalar_names=None) -> "FullStateCritic":
        layout = RaceFeatures.from_state(state, prefix="race_features.", scalar_names=scalar_names).layout()
        horizon = int(state["race_head.hazard.weight"].shape[0]) // 4
        steps = int(state["future_head.weight"].shape[0]) // layout.feature_count
        model = FullStateCritic(trunk, side, layout, horizon=horizon, future_steps=steps)
        model.load_state_dict(state, strict=True)
        return model
