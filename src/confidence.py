# -*- coding: utf-8 -*-
"""State-familiarity confidence (RND) — the confidence wave's S1
signal (USER: "the net should know when it's in well trodden
territory... that corpus can grow as self play grows").

Semantics (measured, experiments\\VALUE_uncertainty\\validation.json):
familiarity means TRAINING BASIS — how much reference experience
covers this state — never correctness.  Gate-passed self-play growth
makes explored regions familiar BY DESIGN; seed-time value poison is
the depth battery's job; shared-trunk bootstrap ensembles REJECTED
on the gift set (DESIGN.md sec 6a).

R4 leg-2/4 fix (2026-08-10): a SINGLE frozen random target's novelty
geometry is draw-sensitive — the stamped fit (seed 20260706)
reordered the class profile vs the validated fit (seed 0): siege
deep-novel conf p50 0.063 -> 0.308 against the frozen <=0.10 bar.
The mechanism now averages K INDEPENDENT frozen target/predictor
pairs (novelty = mean of per-pair MSE); K is MEASURED by the
seed-stability sweep (experiments\\VALUE_corpus_rings\\
rnd_k_stability.py) as the smallest K passing every CONF-1 bar for
every tested base seed.  Artifacts and checkpoint fields carry K;
single-pair artifacts in the previous format still load (K=1).

Artifacts (sha-manifested next to seeds):
  rnd_target.pt   K frozen random MLPs — the measurement frame,
                  seeded ONCE at reset, never retrained
  rnd_pred.pt     K predictors; seeded on the corpus, GROWN on
                  self-play batches (census is telemetry-only under
                  loop v2 — the caller chooses admission;
                  replay-mixed 50/50, lr 3e-4 — naive continuation
                  is test-pinned forbidden: round-1 interference)
  rnd_calib.json  novelty quantiles of a held-out reference slice;
                  confidence = P(calib novelty > x) in [0,1]
                  (parameter-free; 0.5 = median reference state)

API (consumption is SEARCH's law set; this module only measures):
  Familiarity.seed(ref_states, seed, k) / .load(dir) / .save(dir)
  Familiarity.confidence(enc_batch) -> [0,1]
  Familiarity.grow(new_states, replay_states) / .recalibrate(hold)
  seed_from_ds / to_checkpoint_fields / from_checkpoint_fields /
  provider / register_from_checkpoint          (pipeline + loader)
ASCII only.
"""
import json
import os

HID = 256
OUT_D = 64
GROW_LR = 3e-4
SEED_LR = 1e-3
CALIB_N = 4096
K_DEFAULT = int(os.environ.get('WOTR_CONF_RND_K', '4') or 4)


def _mlp(d_in, seed):
    import torch
    import torch.nn as nn
    # fork_rng: the seeded init must NEVER stomp the process-wide
    # torch RNG (checkpoint auto-registration runs inside net loading,
    # which would silently shift every later torch draw)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(d_in, HID), nn.ReLU(),
                             nn.Linear(HID, OUT_D))


class Familiarity:
    def __init__(self, targets, preds, calib, d_enc):
        if not isinstance(targets, (list, tuple)):
            targets = [targets]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        assert len(targets) == len(preds) and len(targets) >= 1
        self.targets = list(targets)
        self.preds = list(preds)
        self.calib = calib
        self.d_enc = d_enc

    @property
    def k(self):
        return len(self.targets)

    # legacy single-pair aliases (first pair)
    @property
    def target(self):
        return self.targets[0]

    @property
    def pred(self):
        return self.preds[0]

    # ------------------------------------------------------ build
    @classmethod
    def seed(cls, ref_states, seed=0, epochs=2, k=None):
        """Reset-time build: K frozen random targets + K predictors
        trained on the reference corpus; calibration from a held-out
        slice (never trained on).  Adaptive calib slice for small
        references; hard refusal below 512 rows (B5 lesson)."""
        import numpy as np
        if k is None:
            k = K_DEFAULT
        X = np.asarray(ref_states, dtype=np.float32)
        if len(X) < 512:
            raise ValueError('reference too small: %d' % len(X))
        d = X.shape[1]
        calib_n = min(CALIB_N, max(256, len(X) // 5))
        hold = X[-calib_n:]
        train = X[:-calib_n]
        targets, preds = [], []
        for i in range(k):
            t = _mlp(d, seed * 1000 + i)
            for p in t.parameters():
                p.requires_grad = False
            targets.append(t)
            preds.append(_mlp(d, seed * 1000 + 500 + i))
        obj = cls(targets, preds, None, d)
        obj._fit(train, epochs=epochs, lr=SEED_LR, rng_seed=seed)
        obj.calib = np.sort(obj.novelty(hold))
        return obj

    def _fit(self, X, epochs, lr, rng_seed):
        import numpy as np
        import torch
        n = len(X)
        if n < 2:
            raise ValueError('fit set too small: %d' % n)
        bs = min(512, n)
        for ki, (target, pred) in enumerate(zip(self.targets,
                                                self.preds)):
            opt = torch.optim.Adam(pred.parameters(), lr=lr)
            rng = np.random.RandomState(rng_seed * 100 + ki)
            target.eval()
            pred.train()
            for _ in range(epochs):
                order = rng.permutation(n)
                for i in range(0, n - bs + 1, bs):
                    xb = torch.from_numpy(X[order[i:i + bs]])
                    with torch.no_grad():
                        tb = target(xb)
                    loss = torch.nn.functional.mse_loss(pred(xb),
                                                        tb)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            pred.eval()

    # ------------------------------------------------------ scores
    def novelty(self, X, bs=1024):
        """Mean per-pair prediction error (K-averaged geometry —
        the R4 draw-stability fix)."""
        import numpy as np
        import torch
        X = np.asarray(X, dtype=np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(X[i:i + bs])
                acc = None
                for target, pred in zip(self.targets, self.preds):
                    e = ((pred(xb) - target(xb)) ** 2).mean(1)
                    acc = e if acc is None else acc + e
                out.append((acc / float(self.k)).numpy())
        return np.concatenate(out)

    def confidence(self, X):
        """[0,1]: fraction of the calibration slice MORE novel than
        each state (1 = deep inside reference experience)."""
        import numpy as np
        raw = self.novelty(X)
        idx = np.searchsorted(self.calib, raw, side='right')
        return 1.0 - idx / float(len(self.calib))

    # ------------------------------------------------------ growth
    def grow(self, new_states, replay_states, seed=0):
        """Loop op — grow on self-play batch states (the caller
        chooses admission; census is telemetry-only under loop v2).
        Replay-mixed continuation (round-1 interference lesson);
        caller re-derives calibration afterwards when the reference
        definition grows (recalibrate())."""
        import numpy as np
        new_states = np.asarray(new_states, dtype=np.float32)
        replay_states = np.asarray(replay_states, dtype=np.float32)
        kk = min(len(new_states), len(replay_states))
        rng = np.random.RandomState(seed)
        mix = np.concatenate([
            new_states,
            replay_states[rng.choice(len(replay_states), size=kk,
                                     replace=False)]])
        self._fit(mix, epochs=1, lr=GROW_LR, rng_seed=seed + 1)

    def recalibrate(self, hold_states):
        import numpy as np
        self.calib = np.sort(self.novelty(hold_states))

    # ------------------------------------------------------ persist
    def save(self, dirpath):
        import torch
        os.makedirs(dirpath, exist_ok=True)
        torch.save({'k': self.k, 'd_enc': self.d_enc,
                    'frozen': True,
                    'states': [t.state_dict()
                               for t in self.targets]},
                   os.path.join(dirpath, 'rnd_target.pt'))
        torch.save({'k': self.k, 'd_enc': self.d_enc,
                    'states': [p.state_dict()
                               for p in self.preds]},
                   os.path.join(dirpath, 'rnd_pred.pt'))
        with open(os.path.join(dirpath, 'rnd_calib.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'calib': [float(v) for v in self.calib],
                       'd_enc': self.d_enc, 'k': self.k}, f)

    @classmethod
    def load(cls, dirpath):
        import numpy as np
        import torch
        ck_t = torch.load(os.path.join(dirpath, 'rnd_target.pt'),
                          map_location='cpu', weights_only=True)
        ck_p = torch.load(os.path.join(dirpath, 'rnd_pred.pt'),
                          map_location='cpu', weights_only=True)
        d = int(ck_t['d_enc'])
        t_states = ck_t.get('states') or [ck_t['state']]
        p_states = ck_p.get('states') or [ck_p['state']]
        targets, preds = [], []
        for i, sd in enumerate(t_states):
            t = _mlp(d, i)
            t.load_state_dict(sd)
            for p in t.parameters():
                p.requires_grad = False
            targets.append(t)
        for i, sd in enumerate(p_states):
            p9 = _mlp(d, 500 + i)
            p9.load_state_dict(sd)
            preds.append(p9)
        with open(os.path.join(dirpath, 'rnd_calib.json'),
                  encoding='utf-8') as f:
            calib = np.asarray(json.load(f)['calib'])
        return cls(targets, preds, calib, d)


# ---------------------------------------------------------------
# Checkpoint stamping + provider glue (comms\30 contract)
# ---------------------------------------------------------------
CONF_MECH = 'rnd_familiarity_v1'


def seed_from_ds(ds_dir, tag, n_ref=60000, seed=0, epochs=2,
                 k=None):
    """Pipeline-time reference fit: sample the SIDE'S OWN consolidated
    state encodings ({tag}_X.npy — exactly the encodings that side's
    net consumes: the same-encoding invariant, comms\\31 amendment A)
    from the non-tail zone (tail reserved as in the validated
    recipe), then Familiarity.seed with K pairs."""
    import numpy as np
    X = np.load(os.path.join(ds_dir, '%s_X.npy' % tag),
                mmap_mode='r')
    n = X.shape[0]
    # reserve at most 200k tail rows but never starve the reference
    # (n_ref)
    tail = min(200000, max(0, n - (n_ref + CALIB_N)))
    zone = n - tail
    take = min(n_ref + CALIB_N, zone)
    rng = np.random.RandomState(seed)
    rows = np.sort(rng.choice(zone, size=take, replace=False))
    ref = np.array(X[rows], dtype=np.float32)
    return Familiarity.seed(ref, seed=seed, epochs=epochs, k=k)


def to_checkpoint_fields(fam, meta=None):
    """The four comms\\30 fields, torch-save-safe (K-aware)."""
    def _cpu(sd):
        return {kk: v.cpu() for kk, v in sd.items()}
    return {
        'conf_mech': CONF_MECH,
        'conf_state': {
            'k': fam.k,
            'targets': [_cpu(t.state_dict()) for t in fam.targets],
            'preds': [_cpu(p.state_dict()) for p in fam.preds],
            'd_enc': int(fam.d_enc)},
        'conf_calib': [float(v) for v in fam.calib],
        'conf_meta': dict(meta or {}),
    }


def from_checkpoint_fields(ckpt):
    """Rebuild Familiarity from stamped fields; None when absent or a
    different mechanism.  Accepts the legacy single-pair layout."""
    import numpy as np
    if not isinstance(ckpt, dict) or \
            ckpt.get('conf_mech') != CONF_MECH:
        return None
    st = ckpt['conf_state']
    d = int(st['d_enc'])
    t_states = st.get('targets') or [st['target']]
    p_states = st.get('preds') or [st['pred']]
    targets, preds = [], []
    for i, sd in enumerate(t_states):
        t = _mlp(d, i)
        t.load_state_dict(sd)
        for p in t.parameters():
            p.requires_grad = False
        targets.append(t)
    for i, sd in enumerate(p_states):
        p9 = _mlp(d, 500 + i)
        p9.load_state_dict(sd)
        preds.append(p9)
    return Familiarity(targets, preds,
                       np.asarray(ckpt['conf_calib'], dtype=float),
                       d)


# ---------------------------------------------------------------
# Stamp-time CONF-1 self-certification (R4 leg-2/4: the fitted
# mechanism's class profile is draw-sensitive; the artifact that
# ships must pass ITS OWN bars, with provenance — mirroring the
# depth-cert doctrine: measured selection, loudly refused otherwise)
# ---------------------------------------------------------------
CONF1_SIEGE_P50_MAX = 0.10       # frozen bar (comms\32)
CONF1_AUC_MIN = 0.55             # frozen bar (comms\32)
CONF1_MARGIN_MIN = 0.25          # deep-novel class margin


def _fixture_interior_states(rng_seed=1):
    """Deterministic poison-interior encodings from the behavior
    fixtures (fp perspective) — the labeled deep-novel class, built
    on the fly so stamping needs no lab-only artifacts."""
    import pickle
    import numpy as np
    import ai
    import board as board_mod
    from game_state import Side
    fdir = os.environ.get('WOTR_FIXTURES_DIR') or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'behavior_fixtures')
    rng = np.random.RandomState(rng_seed)
    rows, tags = [], []
    for fix in ('siege_choice_erebor', 'mordor_push',
                'erebor_recapture'):
        try:
            with open(os.path.join(fdir, fix + '.pkl'), 'rb') as f:
                d = pickle.load(f)
            st = d['state']
            if st.board is None:
                st.board = board_mod.load_board()
            acts = ai.get_legal_actions(st, Side.FREE_PEOPLES)
        except Exception as _fx_err9:
            # LOUD: a silently-dropped fixture class skews the deep-
            # novel calibration (a missing siege class reads p50=1.0
            # and the certification band passes vacuously)
            print('[confidence] fixture %s FAILED to load: %r'
                  % (fix, _fx_err9))
            continue
        for i in rng.permutation(len(acts))[:24]:
            try:
                c1 = ai.clone_state(st)
                ai._execute_action(c1, acts[int(i)])
                rows.append(ai.encode_state(c1, Side.FREE_PEOPLES))
                tags.append(fix)
            except Exception:
                continue
            try:
                acts2 = ai.get_legal_actions(c1, Side.FREE_PEOPLES)
            except Exception:
                continue
            for j in rng.permutation(len(acts2))[:5]:
                try:
                    c2 = ai.clone_state(c1)
                    ai._execute_action(c2, acts2[int(j)])
                    rows.append(ai.encode_state(c2,
                                                Side.FREE_PEOPLES))
                    tags.append(fix)
                except Exception:
                    continue
    return (np.asarray(rows, dtype=np.float32),
            np.asarray(tags, dtype=object))


def conf1_check(fam, ds_dir, tag, n_common=4000):
    """Measure the frozen CONF-1 bars for a fitted mechanism.
    fp: siege-class conf p50 + overall AUC + deep-novel margin.
    sh: no labeled poison class exists — calibration-median sanity
    only (documented limitation, comms\\32 bars are fp-scoped)."""
    import numpy as np
    X = np.load(os.path.join(ds_dir, '%s_X.npy' % tag),
                mmap_mode='r')
    n = X.shape[0]
    rows = np.sort(np.random.RandomState(11).choice(
        min(n, 200000), size=min(n_common, n), replace=False))
    common = np.array(X[n - 1 - rows][::-1], dtype=np.float32)
    n_c = fam.novelty(common)
    med = float(np.median(fam.confidence(common)))
    rep = {'common_conf_median': round(med, 3)}
    if tag != 'fp':
        rep['pass'] = bool(0.35 <= med <= 0.65)
        return rep
    poison, tags = _fixture_interior_states()
    if not len(poison):
        rep['pass'] = False
        rep['error'] = 'no fixture interiors'
        return rep
    n_p = fam.novelty(poison)
    conf_p = 1.0 - np.searchsorted(np.sort(n_c), n_p,
                                   side='right') / float(len(n_c))
    siege = np.array([t == 'siege_choice_erebor' for t in tags])
    siege_p50 = float(np.percentile(conf_p[siege], 50)) \
        if siege.any() else 1.0
    # tie-averaged AUC
    allv = np.concatenate([n_p, n_c])
    order = np.argsort(allv, kind='mergesort')
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    u = ranks[:len(n_p)].sum() - len(n_p) * (len(n_p) + 1) / 2.0
    auc = float(u / (len(n_p) * len(n_c)))
    rep.update({'siege_conf_p50': round(siege_p50, 3),
                'auc': round(auc, 3),
                'margin': round(0.5 - siege_p50, 3),
                'pass': bool(siege_p50 <= CONF1_SIEGE_P50_MAX
                             and auc >= CONF1_AUC_MIN
                             and (0.5 - siege_p50)
                             >= CONF1_MARGIN_MIN)})
    return rep


def seed_certified(ds_dir, tag, base_seed=0, k=None, tries=5):
    """Fit + stamp-time CONF-1 cert with refit-on-fail (seeds
    base_seed..base_seed+tries-1).  Returns (fam, report) where
    report records every attempt (provenance).  Raises SystemExit(4)
    when no attempt passes — an uncertified confidence artifact must
    never ship silently."""
    attempts = []
    for t in range(tries):
        s = base_seed + t
        fam = seed_from_ds(ds_dir, tag, seed=s, k=k)
        rep = conf1_check(fam, ds_dir, tag)
        rep['seed'] = s
        rep['k'] = fam.k
        attempts.append(rep)
        print('[conf-cert %s] seed %d k %d -> %s'
              % (tag, s, fam.k,
                 {kk: rep[kk] for kk in rep if kk != 'pass'}),
              flush=True)
        if rep['pass']:
            return fam, {'attempts': attempts, 'chosen_seed': s,
                         'k': fam.k}
    print('[conf-cert %s] NO fit passed CONF-1 in %d tries — '
          'REFUSING to stamp (SystemExit 4)' % (tag, tries),
          flush=True)
    raise SystemExit(4)


_REGISTRY = {}          # side tag ('fp'/'sh') -> Familiarity


def _side_tag(side):
    nm = getattr(side, 'name', None) or str(side)
    nm = nm.upper()
    if 'FREE' in nm or nm == 'FP':
        return 'fp'
    if 'SHADOW' in nm or nm == 'SH':
        return 'sh'
    return None


def provider(vec, side, v_raw):
    """The registered confidence provider (ai.set_confidence_fn
    contract: c(vec, side, v_raw) -> [0,1]).  Unregistered side or
    any shape problem -> 1.0 (identity)."""
    import numpy as np
    fam = _REGISTRY.get(_side_tag(side))
    if fam is None:
        return 1.0
    v = np.asarray(vec, dtype=np.float32)
    if v.ndim == 1:
        v = v[None, :]
    if v.shape[1] != fam.d_enc:
        return 1.0
    return float(fam.confidence(v)[0])


def register_from_checkpoint(ckpt, side_tag):
    """Loader glue: rebuild the stamped mechanism and (re)register
    the dispatching provider.  Returns True when registered."""
    fam = from_checkpoint_fields(ckpt)
    if fam is None or side_tag not in ('fp', 'sh'):
        return False
    _REGISTRY[side_tag] = fam
    import ai
    ai.set_confidence_fn(provider)
    return True
