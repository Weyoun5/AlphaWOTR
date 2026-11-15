"""A BOTH-VIEWS (centralised) critic: the value network sees the position as the Shadow sees it AND as the Free
Peoples see it, at the same decision.

AlphaStar's recipe: the critic is used only in training (baselines, advantages, value deltas), never at play
time, so it may read everything: the Fellowship's true position, both hands, the hunt knowledge. That is not a
leak into the player. The hidden state at a decision is not caused by the move taken, so an advantage computed
from the full state is an unbiased signal for the policy that acts on its own view; and a value that knows the
hidden state no longer averages over it, so its deltas for hunt, Nazgul and eye decisions are sharper.

valuedata.py already stores both views of every decision: `observations` (the mover's) and `watch` (the other
side's), so no new games are needed. The model: the two single-view critics of --init-critic-dir (full_critic.py,
one per side) applied as frozen towers to their own view, their hidden states concatenated and fused by a small
MLP into the Shadow's expected result (tanh). Trained by mean-squared error on the Shadow's result at every
stored decision of either side; games held out by seed. The report compares it, by turn, with the Shadow tower's
own single-view value on the same held-out rows.

  python -m selfplay_actor_critic.full_state_critic --run selfplay_actor_critic/runs/big_g5_g3_a \
      --output selfplay_actor_critic/seeds/both_views_r1 --epochs 4

Memory: the towers' features are streamed to float16 memmaps on disk (<output>/features/<run>.f16, about 1 KB
per decision; reused on a rerun with the same towers) and the fusion head trains by gathering batches from them,
so the process stays at a few GB whatever the number of games. Workers only read games (numpy, no engine) and
at most 2 x workers chunks of --chunksize games are in flight.

Saved as <output>/critic.pt (schema both_views_critic_v1, self-contained: towers + fusion) with a marker file
BOTH_VIEWS; pg_fit.py --critic-dir <output> then computes its baseline from both views
(load_full_state_critic(path, device).values(obs_mine, obs_other, side)).

NOTE on the name: full_critic.py's "FullStateCritic" is a SINGLE-view network with auxiliary heads ("full state"
there means the whole observation as opposed to the race scalars). This module's critic is the one that sees
both sides; its schema is both_views_critic_v1 to keep the two apart.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from .report_metrics import metric_text, r2_score

ROOT = Path(__file__).resolve().parents[1]
MARKER = "BOTH_VIEWS"
SCHEMA = "both_views_critic_v1"


def _engine():
    os.environ.setdefault("WOTR_NATIVE", "0")
    from .generation import require_generation
    require_generation("full_state_critic._engine")
    for _sys_path_entry in (str(ROOT), str(ROOT / "src")):  # R15-PRODUCT-SYSPATH-GROWTH (once, not per call)
        if _sys_path_entry not in sys.path:
            sys.path.insert(0, _sys_path_entry)
    cwd = os.getcwd()
    import ai
    os.chdir(cwd)
    return ai


def views_by_side(observations, watch, sides):
    """(Shadow view, Free Peoples view) per row from the mover's view, the other side's view and the mover."""
    sh = np.asarray(sides) == "sh"
    sh_view = np.where(sh[:, None], observations, watch)
    fp_view = np.where(sh[:, None], watch, observations)
    return sh_view, fp_view


def game_folders(run, max_games=0):
    folders = sorted(glob.glob(os.path.join(run, "batches", "*", "training_workers", "worker_*", "training", "raw-*")))
    return folders[:max_games] if max_games else folders


def read_game(folder):
    """One stored game as (sh_view, fp_view) float16 rows, the Shadow's result, turns, sides, seed; None if unfinished
    or without the other side's view."""
    try:
        meta = json.loads(Path(folder, "episode.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    end = meta.get("end") or {}
    if end.get("status") != "terminated" or not meta.get("rows"):
        return None
    with np.load(os.path.join(folder, "arrays.npz")) as d:
        if "watch" not in d.files:
            return None
        observations, watch = d["observations"], d["watch"]
    rows = meta["rows"]
    if len(observations) != len(rows) or len(watch) != len(rows):
        return None
    sides = [r["side"] for r in rows]
    sh_view, fp_view = views_by_side(observations, watch, sides)
    return dict(sh_view=sh_view.astype(np.float16), fp_view=fp_view.astype(np.float16),
                z=1. if end.get("winner") == "sh" else -1., turn=np.array([r["turn"] for r in rows], np.int16),
                side=np.array(sides), seed=int(meta["seed"]))


def _torch():
    import torch
    return torch


def _fusion_module():
    torch = _torch()
    nn = torch.nn

    class Fusion(nn.Module):
        """tanh( mix_sh * u_sh + mix_fp * u_fp + mlp(h_sh, h_fp) ): starts as the Shadow tower's own value
        (mix = (1, 0), mlp's last layer zero) and learns what the Free Peoples' view adds."""

        def __init__(self, hidden, width):
            super().__init__()
            self.mix = nn.Parameter(torch.tensor([1., 0.]))
            self.mlp = nn.Sequential(nn.Linear(2 * hidden, width), nn.ReLU(), nn.Linear(width, 64), nn.ReLU(), nn.Linear(64, 1))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

        def forward(self, x):
            u = self.mix[0] * x[:, -2] + self.mix[1] * x[:, -1] + self.mlp(x[:, :-2]).reshape(-1)
            return torch.tanh(u)
    return Fusion


def Fusion(hidden, width):
    return _fusion_module()(hidden, width)


def _pre_tanh(tower, h):
    """The tower's value before its final tanh (value_head is a Sequential ending in Tanh)."""
    head = tower.value_head
    return head[:-1](h).reshape(-1) if len(head) > 1 else head(h).reshape(-1)


class BothViewsCritic:
    """Two frozen single-view towers (one per side, full_critic.py critics) and a fusion head."""

    def __init__(self, ai, towers, fusion, hidden, *, meta=None):
        self.ai = ai
        self.towers = towers              # {"sh": critic applied to the Shadow view, "fp": critic applied to the FP view}
        self.fusion = fusion
        self.hidden = int(hidden)
        self.meta = meta or {}
        self.device = "cpu"

    @staticmethod
    def build_fusion(hidden, width=256):
        return Fusion(hidden, width)

    @classmethod
    def from_single_view_critics(cls, ai, critic_dir, *, width=256):
        from .policy import load_critic
        torch = _torch()
        towers, payloads = {}, {}
        for side in ("sh", "fp"):
            path = Path(critic_dir) / f"{side}_critic.pt"
            towers[side] = load_critic(ai, path, side)
            payloads[side] = torch.load(path, map_location="cpu", weights_only=True)
        hidden = cls._hidden_width(towers["sh"])
        if cls._hidden_width(towers["fp"]) != hidden:
            raise ValueError("The two towers have different hidden widths")
        fusion = cls.build_fusion(hidden, width)
        meta = dict(init_critic_dir=str(critic_dir), width=width,
                    towers={s: dict(scalar_names=payloads[s].get("scalar_names"), schema=payloads[s].get("schema")) for s in towers})
        return cls(ai, towers, fusion, hidden, meta=meta)

    @staticmethod
    def _hidden_width(critic):
        torch = _torch()
        return int(next(m for m in critic.value_head.modules() if isinstance(m, torch.nn.Linear)).in_features)

    def to(self, device):
        for tower in self.towers.values():
            tower.to(device)
        self.fusion.to(device)
        self.device = device
        return self

    def tower_features(self, sh_view, fp_view, *, batch=4096):
        """[N, 2 * hidden + 2] float32 on the device: the towers' hidden states and their own values before the
        tanh (Shadow tower on the Shadow view, FP tower on the FP view; sign as each tower sees it). No gradient."""
        torch = _torch()
        parts = []
        with torch.inference_mode():
            for start in range(0, len(sh_view), batch):
                a = torch.from_numpy(np.asarray(sh_view[start:start + batch], np.float32)).to(self.device)
                b = torch.from_numpy(np.asarray(fp_view[start:start + batch], np.float32)).to(self.device)
                ha, hb = self.towers["sh"].forward_backbone(a)[0], self.towers["fp"].forward_backbone(b)[0]
                parts.append(torch.cat([ha, hb, _pre_tanh(self.towers["sh"], ha)[:, None], -_pre_tanh(self.towers["fp"], hb)[:, None]], dim=1))
        return torch.cat(parts) if parts else torch.zeros((0, 2 * self.hidden + 2), device=self.device)

    def single_view_values(self, sh_view, *, batch=4096):
        """The Shadow tower's own value of the Shadow view (what a single-view critic says), numpy."""
        torch = _torch()
        out = []
        with torch.inference_mode():
            for start in range(0, len(sh_view), batch):
                a = torch.from_numpy(np.asarray(sh_view[start:start + batch], np.float32)).to(self.device)
                tower = self.towers["sh"]
                out.append(tower.value_head(tower.forward_backbone(a)[0]).reshape(-1).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def value_shadow(self, sh_view, fp_view, *, batch=4096):
        """V for the Shadow in (-1, 1) from the two views, numpy."""
        torch = _torch()
        out = []
        with torch.inference_mode():
            for start in range(0, len(sh_view), batch):
                h = self.tower_features(sh_view[start:start + batch], fp_view[start:start + batch], batch=batch)
                out.append(self.fusion(h).reshape(-1).double().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def values(self, obs_mine, obs_other, side, *, batch=4096):
        """V in the MOVER's perspective (+ for the Shadow's result when side == 'sh', negated for 'fp') from the
        mover's view and the other side's view of the same decisions."""
        sides = np.full(len(obs_mine), side)
        sh_view, fp_view = views_by_side(np.asarray(obs_mine), np.asarray(obs_other), sides)
        v = self.value_shadow(sh_view, fp_view, batch=batch)
        return v if side == "sh" else -v

    def state(self):
        return dict(schema=SCHEMA, hidden=self.hidden, meta=self.meta,
                    towers={s: {k: v.detach().cpu() for k, v in t.state_dict().items()} for s, t in self.towers.items()},
                    fusion_state={k: v.detach().cpu() for k, v in self.fusion.state_dict().items()})

    def save(self, folder):
        torch = _torch()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        from .runtime_io import _atomic_write, atomic_bytes
        _atomic_write(folder / "critic.pt", lambda stream: torch.save(self.state(), stream))
        atomic_bytes(folder / MARKER, (SCHEMA + "\n").encode("utf-8"))


def is_both_views_dir(path):
    return (Path(path) / MARKER).exists()


def load_full_state_critic(path, device="cpu", ai=None):
    """Load a both-views critic from its folder (or critic.pt). Returns a BothViewsCritic on `device`
    exposing values(obs_mine, obs_other, side)."""
    from .policy import build_critic_model
    torch = _torch()
    path = Path(path)
    file = path / "critic.pt" if path.is_dir() else path
    payload = torch.load(file, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{file}: not a {SCHEMA} file")
    ai = ai or _engine()
    towers = {}
    for side in ("sh", "fp"):
        state = payload["towers"][side]
        names = (payload.get("meta", {}).get("towers", {}).get(side) or {}).get("scalar_names")
        model = build_critic_model(ai, side, state, scalar_names=names)
        model.load_state_dict(state, strict=True)
        towers[side] = model.eval().requires_grad_(False)
    hidden = int(payload["hidden"])
    fusion = BothViewsCritic.build_fusion(hidden, int(payload.get("meta", {}).get("width", 256)))
    fusion.load_state_dict(payload["fusion_state"], strict=True)
    fusion.eval().requires_grad_(False)
    critic = BothViewsCritic(ai, towers, fusion, hidden, meta=payload.get("meta"))
    return critic.to(device)


def explained(z, v):
    """Share of the result's variance explained by v after the best linear map (the measure pg_fit uses)."""
    z = np.asarray(z, np.float64); v = np.asarray(v, np.float64)
    if len(z) < 2 or np.all(z == z[0]):
        return None
    A = np.column_stack((np.ones(len(z)), v))
    fit = A @ np.linalg.lstsq(A, z, rcond=None)[0]
    return r2_score(z, fit)


def by_turn_report(z, v, turn, label):
    cells = {}
    for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 30)):
        m = (turn >= lo) & (turn <= hi)
        if m.sum() > 500:
            cells[f"turns {lo}-{hi}"] = explained(z[m], v[m])
    cells["all"] = explained(z, v)
    print(f"  {label:34s} " + "  ".join(f"{k} {metric_text(x, '5.1f', scale=100, suffix='%')}" for k, x in cells.items()), flush=True)
    return cells


def stored_rows(folder):
    """Number of stored decisions of a finished game (0 = skip), from episode.json alone."""
    try:
        meta = json.loads(Path(folder, "episode.json").read_text(encoding="utf-8"))
    except Exception:
        return 0
    end = meta.get("end") or {}
    if end.get("status") != "terminated":
        return 0
    return int(end.get("transition_count") or len(meta.get("rows", [])))


def read_games(folders):
    """Worker entry (top level, silent): the games of one chunk, unfinished ones as None."""
    return [read_game(f) for f in folders]


def bounded_map(pool, fn, items, *, chunksize, inflight):
    """pool.map with at most `inflight` chunks submitted at a time (results in order). ProcessPoolExecutor.map
    submits everything up front; with tens of thousands of multi-megabyte results that floods the call queue and
    the parent's memory."""
    from collections import deque
    chunks = [items[i:i + chunksize] for i in range(0, len(items), chunksize)]
    pending = deque()
    nxt = 0
    while nxt < len(chunks) and len(pending) < inflight:
        pending.append(pool.submit(fn, chunks[nxt]))
        nxt += 1
    while pending:
        result = pending.popleft().result()
        if nxt < len(chunks):
            pending.append(pool.submit(fn, chunks[nxt]))
            nxt += 1
        yield result


def feature_store(output, run, critic, folders, *, workers=6, chunksize=4, log_every=2000):
    """Stream or reuse a content-bound atomic feature-store generation."""
    from .critic_feature_cache import feature_store as store
    return store(output, run, critic, folders, workers=workers, chunksize=chunksize, log_every=log_every)


def collect_features(critic, output, runs, *, max_games=0, workers=6, chunksize=4):
    """All runs' feature stores concatenated by index (a list of memmaps and offsets) with their labels."""
    stores, parts, games = [], [], 0
    for run in runs:
        folders = game_folders(os.path.abspath(run), max_games)
        store, labels, n = feature_store(output, run, critic, folders, workers=workers, chunksize=chunksize)
        stores.append(store)
        parts.append(labels)
        games += n
        print(f"  {Path(run).name}: {n} games, {len(store)} rows, store {store.nbytes / 1e9:.2f} GB on disk", flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in ("z", "turn", "seed", "sh_row")}
    data["turn"] = data["turn"].astype(np.int64)
    data["h"] = Stores(stores)
    data["single"] = np.tanh(np.concatenate([np.asarray(st[:, -2], np.float32) for st in stores]))   # the Shadow tower's own value
    data["games"] = games
    return data


class Stores:
    """Several row-aligned memmaps read as one array by integer index (sorted gathers, one store at a time)."""

    def __init__(self, stores):
        self.stores = stores
        self.offsets = np.concatenate([[0], np.cumsum([len(s) for s in stores])]).astype(np.int64)
        self.width = int(stores[0].shape[1]) if stores else 0

    def __len__(self):
        return int(self.offsets[-1])

    @property
    def nbytes(self):
        return sum(s.nbytes for s in self.stores)

    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        order = np.argsort(idx, kind="stable")
        out = np.empty((len(idx), self.width), np.float16)
        sorted_idx = idx[order]
        which = np.searchsorted(self.offsets, sorted_idx, side="right") - 1
        for k, store in enumerate(self.stores):
            m = which == k
            if m.any():
                out[order[m]] = store[sorted_idx[m] - self.offsets[k]]
        return out


def train_fusion(critic, data, *, epochs, batch_size, learning_rate, holdout_modulo=10, weight_decay=1e-5, seed=0):
    """Fit the fusion head on the cached tower features; games with seed % holdout_modulo == 0 are held out."""
    if epochs < 1 or batch_size < 1 or holdout_modulo < 2:
        raise ValueError("Fusion fitting requires positive epochs and batch size, and holdout_modulo >= 2")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Fusion fitting requires a finite positive learning rate")
    hold = (data["seed"] % holdout_modulo) == 0
    train_idx, hold_idx = np.nonzero(~hold)[0], np.nonzero(hold)[0]
    if not len(train_idx):
        raise ValueError("No admitted training rows outside the holdout; critic checkpoint was not updated")
    torch = _torch()
    rng = np.random.default_rng(seed)
    H = data["h"]                                      # Stores: float16 memmaps on disk, gathered per batch
    Z = torch.from_numpy(data["z"])
    device = critic.device
    optimizer = torch.optim.Adam(critic.fusion.parameters(), lr=learning_rate, weight_decay=weight_decay)
    critic.fusion.train()
    for epoch in range(epochs):
        order = rng.permutation(train_idx)
        total, count, start = 0., 0, time.time()
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            x = torch.from_numpy(H.take(idx)).to(device).float()
            y = Z[torch.from_numpy(idx)].to(device)
            v = critic.fusion(x).reshape(-1)
            loss = torch.mean((v - y) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx); count += len(idx)
        critic.fusion.eval()
        vh = predict_fusion(critic, H, hold_idx) if len(hold_idx) else np.zeros(0)
        held = explained(data["z"][hold_idx], vh) if len(hold_idx) else None
        print(f"  epoch {epoch + 1}: train mse {total / max(count, 1):.4f}, held-out variance explained {metric_text(held, '5.1f', scale=100, suffix='%')}  ({time.time() - start:.0f}s)", flush=True)
        critic.fusion.train()
    critic.fusion.eval()
    return train_idx, hold_idx


def predict_fusion(critic, stores, idx, batch=65536):
    """The fusion head's values of the rows `idx` of the stores."""
    torch = _torch()
    out = []
    with torch.inference_mode():
        for s in range(0, len(idx), batch):
            x = torch.from_numpy(stores.take(idx[s:s + batch])).to(critic.device).float()
            out.append(critic.fusion(x).reshape(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", nargs="+", required=True, help="valuedata.py runs (games with the other side's view stored)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--init-critic-dir", default=str(ROOT / "selfplay_actor_critic/seeds/full_critic_r2"),
                        help="single-view critics used as the frozen towers (sh_critic.pt / fp_critic.pt)")
    parser.add_argument("--max-games", type=int, default=0, help="per run (0 = all)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6, help="game readers (numpy only, no engine)")
    parser.add_argument("--chunksize", type=int, default=4, help="games per worker result; at most 2 x workers chunks are in flight")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch = _torch()
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    ai = _engine()
    critic = BothViewsCritic.from_single_view_critics(ai, args.init_critic_dir, width=args.width).to(device)
    print(f"{len(args.run)} run(s); towers from {args.init_critic_dir} (hidden {critic.hidden}); device {device}; "
          f"features stream to {Path(args.output) / 'features'}", flush=True)
    started = time.time()
    data = collect_features(critic, args.output, args.run, max_games=args.max_games, workers=args.workers, chunksize=args.chunksize)
    print(f"  {data['games']} finished games, {len(data['z'])} decisions, features {data['h'].nbytes / 1e9:.2f} GB on disk ({time.time() - started:.0f}s)", flush=True)
    train_idx, hold_idx = train_fusion(critic, data, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate)
    report = dict(args=vars(args), games=int(data["games"]), decisions=int(len(data["z"])), held_out_games=int(len(np.unique(data["seed"][hold_idx]))))
    if len(hold_idx):
        vh = predict_fusion(critic, data["h"], hold_idx)
        zh, th, sh_rows = data["z"][hold_idx], data["turn"][hold_idx], data["sh_row"][hold_idx]
        print("held-out games, variance of the Shadow's result explained (all decisions of either side):")
        report["both_views"] = by_turn_report(zh, vh, th, "both-views critic")
        report["single_view_shadow_tower"] = by_turn_report(zh, data["single"][hold_idx], th, "Shadow tower alone (Shadow view)")
        print("held-out games, Shadow decisions only (what pg_fit --side sh uses):")
        report["both_views_sh_rows"] = by_turn_report(zh[sh_rows], vh[sh_rows], th[sh_rows], "both-views critic")
        report["single_view_sh_rows"] = by_turn_report(zh[sh_rows], data["single"][hold_idx][sh_rows], th[sh_rows], "Shadow tower alone (Shadow view)")
    critic.save(args.output)
    Path(args.output, "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"written {args.output} (critic.pt + {MARKER})", flush=True)


if __name__ == "__main__":
    main()
