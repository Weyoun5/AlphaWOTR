"""Race critic: a small value model over the victory-race scalars.

The game ends when the FIRST of four clocks fires (src/game_logic.py
check_all_victory_conditions / should_game_end):

  0 sh_ring      corruption reaches 12            immediate, checked first
  1 fp_ring      Mordor step 5 with corruption<12 immediate
  2 sh_military  Shadow holds 10 VP               end-of-turn check, Shadow first
  3 fp_military  Free Peoples hold 4 VP           end-of-turn check

That order inside one turn is the engine's: ring victories end the game
mid-action, military victories only in the end-of-turn check, and Shadow is
checked before the Free Peoples in both. The critic predicts a discrete-time
hazard for every clock over the next K turns from a few dozen race scalars
read out of the observation vector, and the value is the closed-form
probability that an own clock fires before any enemy clock. Nothing in it
rewards an action; at a terminal position it equals the terminal return.

Labels come free from every finished game (survival analysis): at a position
turns_left turns before the end, the winning clock fired at turns_left and
the other three are censored there (ordered before the winner inside the
turn: censored through turns_left inclusive; after it: through turns_left-1).
Games without a known clock (truncation, resignation with an unknown route)
censor every clock and contribute only their outcome, when known.

The module is deliberately tiny (a few thousand parameters): with ~500
self-play games per generation a 1.3M-parameter critic memorises game
identities (in-sample 0.15 vs held-out 0.6-0.7 outcome MSE); a model that
only sees the race cannot, and the search cannot steer it into
position-specific blind spots.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CLOCKS = ("sh_ring", "fp_ring", "sh_military", "fp_military")
# Within-turn resolution order (index = position in CLOCKS).
CLOCK_ORDER = (0, 1, 2, 3)
CLOCK_OF_ROUTE = {("sh", "ring_corruption"): 0, ("fp", "ring_destroyed"): 1,
                  ("sh", "military"): 2, ("fp", "military"): 3}
FP_CLOCKS = (1, 3)
SH_CLOCKS = (0, 2)
SIDE_SIGN = {"fp": 1., "sh": -1.}
# Row labels: clock -2 = no race label (outcome only), -1 = no clock fired
# (truncated / resigned), 0..3 = the clock that ended the game.
UNLABELED = -2
NO_CLOCK = -1


@dataclass(frozen=True)
class RaceLayout:
    """Where the race scalars live inside the observation vector.

    scalars: direct read-outs (name, observation index, scale divisor).
    Region blocks: observation[region_base + r * region_stride + offset] for
    region r; fp_strength / sh_strength are (offset, weight) pairs summed
    into an army-strength number per region for each side.
    Settlements: region index, VP value (city 1 / stronghold 2), owner
    (0 = Free Peoples settlement, scores for Shadow when captured; 1 = Shadow
    settlement), the observation index of its "captured by the enemy" flag
    (-1 if none), and its neighbouring region indices (the threat ring).
    """
    width: int
    scalar_names: tuple
    scalar_index: tuple
    scalar_scale: tuple
    region_base: int
    region_count: int
    region_stride: int
    fp_strength: tuple
    sh_strength: tuple
    settlement_region: tuple
    settlement_vp: tuple
    settlement_owner: tuple
    settlement_captured_index: tuple
    settlement_neighbors: tuple

    def __post_init__(self):
        n = len(self.scalar_index)
        if not (len(self.scalar_names) == len(self.scalar_scale) == n):
            raise ValueError("scalar name/index/scale lengths differ")
        if any(not 0 <= int(i) < self.width for i in self.scalar_index):
            raise ValueError("scalar index outside the observation")
        if self.region_count > 0:
            last = self.region_base + self.region_count * self.region_stride
            if self.region_base < 0 or last > self.width:
                raise ValueError("region blocks outside the observation")
            for offset, _ in tuple(self.fp_strength) + tuple(self.sh_strength):
                if not 0 <= int(offset) < self.region_stride:
                    raise ValueError("strength offset outside the region block")
        m = len(self.settlement_region)
        if not (len(self.settlement_vp) == len(self.settlement_owner) == len(self.settlement_captured_index)
                == len(self.settlement_neighbors) == m):
            raise ValueError("settlement table lengths differ")
        for r in self.settlement_region:
            if not 0 <= int(r) < self.region_count:
                raise ValueError("settlement region outside the board")
        for i in self.settlement_captured_index:
            if not -1 <= int(i) < self.width:
                raise ValueError("captured-flag index outside the observation")
        for ring in self.settlement_neighbors:
            if any(not 0 <= int(r) < self.region_count for r in ring):
                raise ValueError("neighbour outside the board")

    @property
    def feature_count(self) -> int:
        # scalars + per settlement (own inside, enemy inside, enemy adjacent,
        # captured) + per-owner aggregates (6 each).
        return len(self.scalar_index) + 4 * len(self.settlement_region) + 12

    @property
    def feature_names(self) -> tuple:
        names = list(self.scalar_names)
        for k in range(len(self.settlement_region)):
            names += [f"settlement{k}_own_inside", f"settlement{k}_enemy_inside",
                      f"settlement{k}_enemy_adjacent", f"settlement{k}_captured"]
        for owner in ("fp_settlements", "sh_settlements"):
            names += [f"{owner}_vp_captured", f"{owner}_vp_enemy_inside", f"{owner}_vp_threatened",
                      f"{owner}_max_enemy_adjacent", f"{owner}_threatened_count", f"{owner}_vp_remaining"]
        return tuple(names)


class RaceFeatures(nn.Module):
    """Buffer-only read-out of the race scalars from observation rows."""

    def __init__(self, layout: RaceLayout):
        super().__init__()
        self.register_buffer("width", torch.tensor(int(layout.width), dtype=torch.long))
        self.register_buffer("scalar_index", torch.tensor(layout.scalar_index, dtype=torch.long))
        self.register_buffer("scalar_scale", torch.tensor(layout.scalar_scale, dtype=torch.float32))
        self.register_buffer("region_shape", torch.tensor([layout.region_base, layout.region_count, layout.region_stride],
                                                          dtype=torch.long))
        stride = max(int(layout.region_stride), 1)
        fp = torch.zeros(stride)
        for offset, weight in layout.fp_strength:
            fp[int(offset)] += float(weight)
        sh = torch.zeros(stride)
        for offset, weight in layout.sh_strength:
            sh[int(offset)] += float(weight)
        self.register_buffer("fp_strength", fp)
        self.register_buffer("sh_strength", sh)
        m = len(layout.settlement_region)
        self.register_buffer("settlement_region", torch.tensor(layout.settlement_region, dtype=torch.long))
        self.register_buffer("settlement_vp", torch.tensor(layout.settlement_vp, dtype=torch.float32))
        self.register_buffer("settlement_owner", torch.tensor(layout.settlement_owner, dtype=torch.long))
        self.register_buffer("settlement_captured_index", torch.tensor(layout.settlement_captured_index, dtype=torch.long))
        adjacency = torch.zeros(m, max(int(layout.region_count), 1))
        for k, ring in enumerate(layout.settlement_neighbors):
            for r in ring:
                adjacency[k, int(r)] = 1.
        self.register_buffer("adjacency", adjacency)
        self.feature_count = layout.feature_count
        self.feature_names = layout.feature_names

    def layout(self) -> RaceLayout:
        base, count, stride = (int(v) for v in self.region_shape.tolist())
        return RaceLayout(
            width=int(self.width), scalar_names=tuple(self.feature_names[:len(self.scalar_index)]),
            scalar_index=tuple(int(i) for i in self.scalar_index.tolist()),
            scalar_scale=tuple(float(s) for s in self.scalar_scale.tolist()),
            region_base=base, region_count=count, region_stride=stride,
            fp_strength=tuple((i, float(w)) for i, w in enumerate(self.fp_strength.tolist()) if w),
            sh_strength=tuple((i, float(w)) for i, w in enumerate(self.sh_strength.tolist()) if w),
            settlement_region=tuple(int(r) for r in self.settlement_region.tolist()),
            settlement_vp=tuple(int(v) for v in self.settlement_vp.tolist()),
            settlement_owner=tuple(int(o) for o in self.settlement_owner.tolist()),
            settlement_captured_index=tuple(int(i) for i in self.settlement_captured_index.tolist()),
            settlement_neighbors=tuple(tuple(int(r) for r in torch.nonzero(row).reshape(-1).tolist())
                                       for row in self.adjacency))

    @staticmethod
    def from_state(state: dict, prefix: str = "features.", scalar_names=None) -> "RaceFeatures":
        """Rebuild from a saved state dict (shapes come from the tensors; the
        scalar names are not tensors, so callers pass them from the payload)."""
        base, count, stride = (int(v) for v in state[prefix + "region_shape"].tolist())
        scalar_index = state[prefix + "scalar_index"].tolist()
        if scalar_names is not None and len(scalar_names) != len(scalar_index):
            raise ValueError("scalar names do not match the saved scalar index")
        layout = RaceLayout(
            width=int(state[prefix + "width"].item()),
            scalar_names=tuple(f"scalar{i}" for i in range(len(scalar_index))) if scalar_names is None else tuple(scalar_names),
            scalar_index=tuple(int(i) for i in scalar_index),
            scalar_scale=tuple(float(s) for s in state[prefix + "scalar_scale"].tolist()),
            region_base=base, region_count=count, region_stride=stride,
            fp_strength=tuple((i, float(w)) for i, w in enumerate(state[prefix + "fp_strength"].tolist()) if w),
            sh_strength=tuple((i, float(w)) for i, w in enumerate(state[prefix + "sh_strength"].tolist()) if w),
            settlement_region=tuple(int(r) for r in state[prefix + "settlement_region"].tolist()),
            settlement_vp=tuple(int(v) for v in state[prefix + "settlement_vp"].tolist()),
            settlement_owner=tuple(int(o) for o in state[prefix + "settlement_owner"].tolist()),
            settlement_captured_index=tuple(int(i) for i in state[prefix + "settlement_captured_index"].tolist()),
            settlement_neighbors=tuple(tuple(int(r) for r in torch.nonzero(row).reshape(-1).tolist())
                                       for row in state[prefix + "adjacency"]))
        return RaceFeatures(layout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != int(self.width):
            raise ValueError("Race features expect observation rows [N, width]")
        parts = [x[:, self.scalar_index] / self.scalar_scale]
        base, count, stride = (int(v) for v in self.region_shape.tolist())
        m = int(self.settlement_region.numel())
        if count > 0 and m > 0:
            blocks = x[:, base:base + count * stride].reshape(x.shape[0], count, stride)
            fp = blocks @ self.fp_strength            # [N, R] Free Peoples strength per region
            sh = blocks @ self.sh_strength            # [N, R] Shadow strength per region
            owner = self.settlement_owner.to(x.dtype).unsqueeze(0)          # [1, M] 1 = Shadow settlement
            fp_at = fp[:, self.settlement_region]
            sh_at = sh[:, self.settlement_region]
            own_inside = (1 - owner) * fp_at + owner * sh_at
            enemy_inside = (1 - owner) * sh_at + owner * fp_at
            fp_adjacent = fp @ self.adjacency.T
            sh_adjacent = sh @ self.adjacency.T
            enemy_adjacent = (1 - owner) * sh_adjacent + owner * fp_adjacent
            captured = torch.zeros_like(own_inside)
            has_flag = self.settlement_captured_index >= 0
            if bool(has_flag.any()):
                captured[:, has_flag] = x[:, self.settlement_captured_index[has_flag]]
            scale = 5.
            parts.append(torch.stack((own_inside / scale, enemy_inside / scale, enemy_adjacent / scale, captured),
                                     dim=-1).reshape(x.shape[0], 4 * m))
            vp = self.settlement_vp.unsqueeze(0)
            threatened = ((enemy_inside + enemy_adjacent) > 0).to(x.dtype) * (1 - captured)
            for owner_value in (0., 1.):
                mask = (self.settlement_owner.to(x.dtype) == owner_value).to(x.dtype).unsqueeze(0)
                parts.append(torch.stack((
                    (vp * captured * mask).sum(-1) / 10.,
                    (vp * (enemy_inside > 0).to(x.dtype) * (1 - captured) * mask).sum(-1) / 10.,
                    (vp * threatened * mask).sum(-1) / 10.,
                    (enemy_adjacent * mask).amax(-1) / scale,
                    (threatened * mask).sum(-1) / 5.,
                    (vp * (1 - captured) * mask).sum(-1) / 10.), dim=-1))
        else:
            parts.append(torch.zeros(x.shape[0], 4 * m + 12, dtype=x.dtype, device=x.device))
        return torch.cat(parts, dim=-1)


def clock_log_mass(logits: torch.Tensor):
    """Discrete-time hazards -> (log P(fire at k) [N,C,K], log P(no fire within K) [N,C])."""
    log_h = F.logsigmoid(logits)
    log_1mh = F.logsigmoid(-logits)
    cumulative = torch.cumsum(log_1mh, dim=-1)
    survive_before = torch.cat((torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]), dim=-1)
    return log_h + survive_before, cumulative[..., -1]


def fp_win_probability(logits: torch.Tensor, beyond_logit: torch.Tensor) -> torch.Tensor:
    """P(an FP clock fires before any Shadow clock) from per-clock hazards.

    logits [N, 4, K] in CLOCKS order; beyond_logit [N] gives P(FP wins) when
    no clock fires within the horizon. Clocks are placed on a global time
    axis tau = 4k + order so that no two can tie; computed in float64."""
    n, c, k = logits.shape
    if c != len(CLOCKS):
        raise ValueError("hazard logits need one row per clock")
    log_mass, _ = clock_log_mass(logits.double())
    mass = log_mass.exp()                                    # [N, 4, K]
    axis = torch.zeros(n, c, 4 * k, dtype=mass.dtype, device=mass.device)
    for clock, order in enumerate(CLOCK_ORDER):
        axis[:, clock, order::4] = mass[:, clock]
    cumulative = torch.cumsum(axis, dim=-1).clamp(max=1.)   # P(tau_c <= tau)
    survive = 1. - cumulative                                # P(tau_c > tau)
    survive_fp = survive[:, list(FP_CLOCKS)].prod(dim=1)    # [N, 4K]
    survive_sh = survive[:, list(SH_CLOCKS)].prod(dim=1)
    earlier = torch.cat((torch.ones_like(survive_fp[:, :1]), survive_fp[:, :-1]), dim=-1)
    within = ((earlier - survive_fp) * survive_sh).sum(-1)
    neither = survive_fp[:, -1] * survive_sh[:, -1]
    return (within + neither * torch.sigmoid(beyond_logit.double())).clamp(1e-7, 1 - 1e-7)


class RaceHead(nn.Module):
    """features -> own-side value; hazards() exposes the per-clock logits."""

    def __init__(self, feature_count: int, side: str, *, hidden: int = 64, horizon: int = 24):
        super().__init__()
        if side not in SIDE_SIGN:
            raise ValueError("side must be 'fp' or 'sh'")
        self.horizon = int(horizon)
        self.register_buffer("sign", torch.tensor(SIDE_SIGN[side]))
        self.body = nn.Sequential(nn.Linear(feature_count, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.hazard = nn.Linear(hidden, len(CLOCKS) * self.horizon)
        self.beyond = nn.Linear(hidden, 1)
        with torch.no_grad():
            # Start from a flat, low hazard (about 1/12 per turn per clock) so
            # the untrained race is long and balanced rather than instant.
            self.hazard.weight.mul_(0.1)
            self.hazard.bias.fill_(-2.4)
            self.beyond.weight.zero_()
            self.beyond.bias.zero_()

    def hazards(self, features: torch.Tensor):
        hidden = self.body(features)
        return self.hazard(hidden).reshape(features.shape[0], len(CLOCKS), self.horizon), self.beyond(hidden).reshape(-1)

    def probability(self, features: torch.Tensor) -> torch.Tensor:
        logits, beyond = self.hazards(features)
        return fp_win_probability(logits, beyond)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = 2. * self.probability(features) - 1.
        return (self.sign.double() * value).to(features.dtype).unsqueeze(-1)


class RaceCritic(nn.Module):
    """Drop-in critic: forward_backbone(x) -> (race features, None); value_head(features) -> [N, 1]."""

    def __init__(self, layout: RaceLayout, side: str, *, hidden: int = 64, horizon: int = 24):
        super().__init__()
        self.side = side
        self.features = RaceFeatures(layout)
        self.value_head = RaceHead(layout.feature_count, side, hidden=hidden, horizon=horizon)

    @property
    def horizon(self) -> int:
        return self.value_head.horizon

    @property
    def layout(self) -> RaceLayout:
        return self.features.layout()

    def forward_backbone(self, x: torch.Tensor):
        return self.features(x), None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.features(x)).reshape(-1)

    @staticmethod
    def is_race_state(state: dict) -> bool:
        return any(k.startswith("features.") for k in state)

    @staticmethod
    def from_state(state: dict, side: str, *, scalar_names=None) -> "RaceCritic":
        layout = RaceFeatures.from_state(state, scalar_names=scalar_names).layout()
        hidden = int(state["value_head.body.0.weight"].shape[0])
        horizon = int(state["value_head.hazard.weight"].shape[0]) // len(CLOCKS)
        model = RaceCritic(layout, side, hidden=hidden, horizon=horizon)
        model.load_state_dict(state, strict=True)
        return model


@dataclass(frozen=True)
class RaceLabel:
    """Survival label of one row: the clock that ended its game (or NO_CLOCK /
    UNLABELED) and how many turns after the row's turn the game ended."""
    clock: int
    turns_left: float


def race_label(winner, route, end_turn, row_turn) -> RaceLabel:
    """Clock + turns_left for a row at row_turn of a game that ended at end_turn."""
    if winner is None or route is None:
        return RaceLabel(NO_CLOCK, float(max(0, int(end_turn) - int(row_turn))))
    clock = CLOCK_OF_ROUTE.get((winner, route))
    if clock is None:
        return RaceLabel(NO_CLOCK, float(max(0, int(end_turn) - int(row_turn))))
    return RaceLabel(clock, float(max(0, int(end_turn) - int(row_turn))))


def race_loss(model: RaceCritic, observations: torch.Tensor, outcome: torch.Tensor, clock: torch.Tensor,
              turns_left: torch.Tensor, *, outcome_weight: float = 1.):
    """Hazard NLL (where the race label is known) + outcome BCE (where the
    outcome is known, 0 = unknown). outcome is the FP-perspective result
    (+1 FP won, -1 Shadow won) regardless of the model's side."""
    features, _ = model.forward_backbone(observations)
    return race_head_loss(model.value_head, features, outcome, clock, turns_left, outcome_weight=outcome_weight)


def race_head_loss(head: RaceHead, features: torch.Tensor, outcome: torch.Tensor, clock: torch.Tensor,
                   turns_left: torch.Tensor, *, outcome_weight: float = 1.):
    """race_loss on precomputed race features (the fitter caches them)."""
    logits, beyond = head.hazards(features)
    n, c, k = logits.shape
    log_h = F.logsigmoid(logits.double())
    log_1mh = F.logsigmoid(-logits.double())
    survive_cum = torch.cumsum(log_1mh, dim=-1)                                   # log S(k+1) = sum_{j<=k}
    survive_before = torch.cat((torch.zeros_like(survive_cum[..., :1]), survive_cum[..., :-1]), dim=-1)
    labeled = clock >= NO_CLOCK
    steps = turns_left.clamp(min=0).round().long().clamp(max=k - 1)             # event/censor index
    beyond_horizon = turns_left.round().long() >= k
    nll = torch.zeros(n, dtype=torch.float64, device=logits.device)
    if bool(labeled.any()):
        idx = steps.unsqueeze(-1).unsqueeze(-1).expand(n, c, 1)
        # Censored through turns_left-1: log S(turns_left) = survive_before[k*]
        cens_before = survive_before.gather(-1, idx).squeeze(-1)                  # [N, C]
        # Censored through turns_left inclusive: log S(turns_left+1)
        cens_through = survive_cum.gather(-1, idx).squeeze(-1)
        event = (log_h + survive_before).gather(-1, idx).squeeze(-1)             # log P(fire at k*)
        clock_ids = torch.arange(c, device=logits.device).unsqueeze(0)
        order = torch.tensor(CLOCK_ORDER, device=logits.device).unsqueeze(0)
        own = clock.unsqueeze(-1) == clock_ids
        winner_order = torch.where(clock >= 0, order.expand(n, -1).gather(-1, clock.clamp(min=0).unsqueeze(-1)).squeeze(-1),
                                   torch.full_like(clock, -1))
        before_winner = (order < winner_order.unsqueeze(-1)) & (clock >= 0).unsqueeze(-1)
        term = torch.where(own, event, torch.where(before_winner, cens_through, cens_before))
        # Beyond the horizon everything is censored at K.
        term = torch.where(beyond_horizon.unsqueeze(-1), survive_cum[..., -1], term)
        nll = -(term * labeled.unsqueeze(-1).double()).sum(-1)
    probability = fp_win_probability(logits, beyond)
    known = outcome != 0
    target = (outcome > 0).double()
    bce = -(target * probability.log() + (1 - target) * (1 - probability).log()) * known.double()
    labeled_count = labeled.double().sum().clamp(min=1.)
    known_count = known.double().sum().clamp(min=1.)
    loss = nll.sum() / labeled_count + outcome_weight * bce.sum() / known_count
    value = head.sign.double() * (2 * probability - 1)
    own_outcome = head.sign.double() * outcome.double()
    mse = (.5 * (value.detach() - own_outcome).square() * known.double()).sum() / known_count
    metrics = dict(value_only_loss=loss.detach(), value_only_outcome_mse=mse,
                   hazard_nll=(nll.detach().sum() / labeled_count), outcome_bce=(bce.detach().sum() / known_count),
                   labeled_fraction=labeled.double().mean())
    return loss.to(torch.float32), metrics


def race_value_loss(model: RaceCritic, value_samples):
    """value_only_loss replacement for race critics over ValueSample rows
    (fields: observation, outcome, clock, turns_left)."""
    device = next(model.parameters()).device
    observations = torch.from_numpy(np.stack([s.observation for s in value_samples]).astype(np.float32)).to(device)
    sign = float(model.value_head.sign)
    # ValueSample.outcome is the OWN-side return; the race loss wants FP perspective.
    outcome = torch.tensor([sign * float(s.outcome) for s in value_samples], dtype=torch.float64, device=device)
    clock = torch.tensor([int(getattr(s, "clock", UNLABELED)) for s in value_samples], dtype=torch.long, device=device)
    turns_left = torch.tensor([float(getattr(s, "turns_left", float("nan"))) for s in value_samples],
                              dtype=torch.float64, device=device)
    turns_left = torch.where(torch.isnan(turns_left), torch.zeros_like(turns_left), turns_left)
    return race_loss(model, observations, outcome, clock, turns_left)
