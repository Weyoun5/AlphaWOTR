"""Forced-deviation tests: pure functions, statistics with planted effects, and SMALL REAL GAMES on this machine's CPU.

  CUDA_VISIBLE_DEVICES=-1 PYTHONHASHSEED=0 KMP_DUPLICATE_LIB_OK=TRUE python -B test_deviation.py --ckpt DIR --vnet DIR --tmp DIR

--ckpt holds fp_win/fp_seed.pt and sh_win/sh_seed.pt; --vnet holds <side>_<model>/best.pt; --tmp receives every output (never deleted
here).  The local tree tolerates the one pin drift smoke_real.py documents (net_tags.py), in this test only.  Not covered: flux, GPUs,
Lustre, anything at scale; test_flux_logic.sh covers the job scripts' branches with a stub flux.
"""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import unittest

RECIPE = dict(WOTR_ACT_GEN="v53", WOTR_NATIVE="0", WOTR_NET_ARCH="base", WOTR_POLICY_DELTA="1",
              WOTR_REWARD_CONTRACT="predecision_profile_v1", WOTR_VALUE_HEAD_ACT="scaled2", WOTR_PHI_RINGS_RHO="0",
              WOTR_OBSERVATION_TRAINING="full_network_v1", WOTR_OPTIMIZER_STATE="persistent_adam_v1",
              WOTR_OPTIMIZER_ALLOW_FRESH="0", PYTHONHASHSEED="0", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
              WOTR_ADMIT_GENERATIONS="v52", NATIVE_RAW_ALLOW_UNPROVEN="1", CUDA_VISIBLE_DEVICES="-1", KMP_DUPLICATE_LIB_OK="TRUE")
for _k, _v in RECIPE.items():
    os.environ.setdefault(_k, _v)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

import dvcommon as D  # noqa: E402
import dvgen  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("dv_report", HERE / "report.py")     # valuelab has a report.py too
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
V, C = D.V, D.C

ARGS = None
SEED0 = 930000000


class Sink:
    def __init__(self):
        self.items = []

    def put(self, message):
        self.items.append(message)


def tolerate_net_tags():
    import native_env
    verify = native_env.verify
    if getattr(verify, "tolerant", False):
        return

    def tolerant(ai_, pin):
        try:
            verify(ai_, pin)
        except RuntimeError as error:
            if str(error) != "native raw generator pin drift: net_tags.py":
                raise
    tolerant.tolerant = True
    native_env.verify = tolerant


class World:
    """One in-process worker and the engine, shared by the real-game tests."""
    worker = None

    @classmethod
    def get(cls, device="cpu"):
        if cls.worker is None:
            tree = C.setup_paths()
            os.environ["WOTR_TREE"] = str(tree)
            V.allow_search_patches()
            tolerate_net_tags()
            from selfplay_actor_critic.generation import require_generation
            generation = require_generation("dv test")
            import dvworker
            spec = dict(index=0, kind="cpu", device=device, lockstep=2, cpus=None, numa=None, gpu=None, max_rows=None, score="batched")
            cls.worker = dvworker.DeviationWorker(spec, dict(tree=str(tree), generation=generation, actor_cache=4), Sink(), threading.Event(),
                                                  threading.Event(), multiprocessing.Value("q", 0))
            cls.worker.setup_process()
            import torch
            torch.set_num_threads(2)
        return cls.worker

    @classmethod
    def run_job(cls, job):
        w = cls.get()
        w.reset_stats()
        w.channel = Sink()
        w.counter.value = 0
        w.abandon.clear()
        w.stop.clear()
        w.load_job(job)
        w.run()
        return w


def rich_hazards():
    """Hazards much above the production rates so a few real games exercise every path."""
    freq = json.loads((HERE / "census_freq.json").read_text(encoding="utf-8"))
    return D.hazards(freq, dict(band=0.2, kind=0.05, exact_aa=0.5, reseed_aa=0.5, handicap=0.3, candidate=1.0))


def plain_game(seed, force=None, stop=None):
    """A T=0 game with no fork machinery: (winner code, plies, trace from ply `force[0]`)."""
    w = World.get()
    game = w.new_game(seed)
    ply, trace = 0, 0
    while game.next():
        obs, watch, cand, regions, kinds = game.encode()
        p = w.actor(game.side).predict_many([w.ActorInput(obs, cand, regions, w.zeros(len(cand)))])[0]
        lp = np.asarray(p.log_probabilities, np.float64)
        selected = int(np.argmax(lp))
        if stop is not None and ply == stop:
            return game, ply, selected, lp, (obs, watch, cand, regions, kinds)
        if force is not None and ply == force[0]:
            selected = force[1]
        if force is not None and ply >= force[0] + 1:
            trace = D.trace_step(trace, len(cand), selected)
        game.apply(selected)
        ply += 1
    end = game.end
    game.close()
    return D.winner_code(end), ply, trace


class TestPure(unittest.TestCase):
    def test_bands(self):
        self.assertEqual([D.alt_band(p) for p in (5e-5, 1e-4, 9e-4, 1e-3, 0.04, 0.05, 0.19, 0.2, 0.6)], [-1, 0, 0, 1, 2, 3, 3, 4, 4])
        self.assertEqual([D.top_band(p) for p in (0.3, 0.5, 0.79, 0.8, 0.94, 0.95, 1.0)], [0, 1, 1, 2, 2, 3, 3])

    def test_hazards_capped_and_scaled(self):
        freq = dict(decisions=dict(fp=100, sh=100), confident=dict(fp=10, sh=10), band={s: [[0, 0.05, 1, 2, 4]] * 4 for s in D.SIDES},
                    kind={s: {g: 2.0 for g in D.GROUPS} for s in D.SIDES})
        h = D.hazards(freq, dict(band=0.1, kind=0.04))
        self.assertEqual(h["band"]["fp"][0][0], 0.0)
        self.assertEqual(h["band"]["fp"][0][1], 1.0)
        self.assertAlmostEqual(h["band"]["fp"][0][2], 0.1)
        self.assertAlmostEqual(h["kind"]["sh"][3], 0.02)

    def test_all_kinds_seen_map_to_a_named_group(self):
        for kind in ("army_move", "pass", "skip", "muster_elite", "fellowship_hide", "companion_separate", "draw_card", "combat_card_play", "casualty_allocate"):
            self.assertNotEqual(D.GROUPS[D.group_code(kind)], "other", kind)
        self.assertEqual(D.GROUPS[D.group_code("never_heard_of")], "other")

    def test_plan_is_a_function_of_the_past_only(self):
        """Same priors, same u: same plan.  The result does not read the game; a draw count never depends on what fires."""
        lp = np.log(np.array([0.9, 0.05, 0.03, 0.015, 0.004, 0.0008, 0.00002, 0.00018], np.float64))
        hz = dict(floor=D.FLOOR, band={"fp": [[1.0] * 5] * 4}, kind={"fp": [0.0] * D.N_GROUPS}, exact_aa=0.0, reseed_aa=0.0, handicap=0.0, candidate=0.0)
        u = np.zeros(D.N_U) + 0.5
        one = D.plan_decision(7, 3, "fp", lp, 0, np.zeros(8, np.int64), ["army_move"], hz, u)
        two = D.plan_decision(7, 3, "fp", lp, 0, np.zeros(8, np.int64), ["army_move"], hz, u)
        self.assertEqual(one, two)
        forks = one[0]
        self.assertEqual(sorted(f["alt_band"] for f in forks), [0, 1, 2, 3])     # .05 is 5-20%, .03 & .015 1-5%, .004 .1-1%, .0008 & .00018 <.1%
        p = np.exp(lp)
        for f in forks:
            self.assertEqual(D.alt_band(p[f["alt"]]), f["alt_band"])
            self.assertNotEqual(f["alt"], 0)
            self.assertGreaterEqual(p[f["alt"]], D.FLOOR)

    def test_selector_never_touches_a_global_stream(self):
        np.random.seed(5)
        a = np.random.random()
        np.random.seed(5)
        s = D.selector(11)
        s.random(D.N_U)
        D.pick_rng(11, 3, 0).integers(10)
        self.assertEqual(a, np.random.random())

    def test_stream_state_and_determinize_keyed_by_position_and_k(self):
        self.assertEqual(D.stream_state(1, 2, 3), D.stream_state(1, 2, 3))
        self.assertNotEqual(D.stream_state(1, 2, 3), D.stream_state(1, 2, 4))
        self.assertEqual(D.determinize_rng(1, 2, 3).integers(1 << 40), D.determinize_rng(1, 2, 3).integers(1 << 40))

    def test_candidate_arms(self):
        lp = np.log(np.arange(1, 21, dtype=np.float64) / 210.0)
        first, chosen = D.candidate_arms(1, 2, lp, 19)
        self.assertEqual(first, [18, 17, 16, 15, 14])
        self.assertEqual(len(chosen), 2)
        self.assertFalse(set(chosen) & set(first))
        self.assertNotIn(19, chosen)

    def test_cluster_pin_gained_exactly_one_hash(self):
        self.assertEqual(V.SEARCH_PATCHES["ai.py"], D.CLUSTER_AI_SHA256)
        self.assertEqual(len(V.SEARCH_PATCHES), 5)


class TestStatistics(unittest.TestCase):
    """The report on data with a KNOWN answer."""

    def fake_forks(self, n_seeds, per_seed, effect, base_rho=0.6, seed=1):
        r = np.random.default_rng(seed)
        seeds = np.repeat(np.arange(n_seeds), per_seed)
        base = r.random(n_seeds) < 0.5
        s_base = base[np.repeat(np.arange(n_seeds), per_seed)].astype(np.float64)
        same = r.random(len(seeds)) < base_rho
        s_alt = np.where(same, s_base, (r.random(len(seeds)) < 0.5).astype(np.float64))
        # plant the effect by flipping losses to wins
        flip = (s_alt == 0) & (r.random(len(seeds)) < 2 * effect)
        s_alt = np.where(flip, 1.0, s_alt)
        return seeds, s_base, s_alt

    def test_planted_effect_is_recovered_with_coverage(self):
        covered = 0
        for t in range(40):
            seeds, s_base, s_alt = self.fake_forks(2500, 2, 0.03, seed=t)
            m, lo, hi = R.boot_mean(s_alt - s_base, seeds, None, 300, seed=t)
            covered += lo <= (s_alt - s_base).mean() + 0 and lo <= 0.03 <= hi
        self.assertGreaterEqual(covered, 34)   # nominal 38 of 40

    def test_null_covers_zero_and_seed_clustering_widens(self):
        seeds, s_base, s_alt = self.fake_forks(4000, 5, 0.0, seed=3)
        d = s_alt - s_base
        m, lo, hi = R.boot_mean(d, seeds, None, 500)
        self.assertLess(lo, 0.0)
        self.assertGreater(hi, 0.0)
        naive = 1.96 * d.std() / np.sqrt(len(d))
        self.assertGreater((hi - lo) / 2, 1.3 * naive)    # the shared baseline counts

    def test_power_arithmetic(self):
        self.assertLess(abs(R.n_needed(0.707) - 4360), 6)
        self.assertLess(abs(R.mde(0.707, 4360) - 0.03), 1e-3)

    def test_benjamini_hochberg(self):
        np.testing.assert_array_equal(R.benjamini_hochberg([0.001, 0.2, 0.012, 0.9]), [True, False, True, False])

    def test_true_sd_recovers_a_planted_spread_and_ignores_noise(self):
        r = np.random.default_rng(9)
        rows = []
        for pid in range(300):
            for arm in range(4):
                truth = r.normal(-0.02, 0.05)
                n = 64
                odd, even = r.normal(truth, 0.6 / np.sqrt(n / 2), 2)
                rows.append(dict(pid=pid, odd=odd, even=even, mean=(odd + even) / 2))
        t = R.true_sd(rows)
        self.assertLess(abs(t["sd"] - 5.0), 2.0)
        rows2 = [dict(x, odd=r.normal(-0.02, 0.075), even=r.normal(-0.02, 0.075)) for x in rows]
        t2 = R.true_sd(rows2)
        self.assertLess(t2["sd"], 3.0)

    def test_slope_recovers_calibration_and_chance(self):
        r = np.random.default_rng(1)
        x = r.normal(0, 0.05, 20000)
        y = x + r.normal(0, 0.45, 20000)
        s = R.slope(x, y, np.arange(20000))
        self.assertLess(s["lo"], 1.0)
        self.assertGreater(s["hi"], 1.0)
        s2 = R.slope(x, r.normal(0, 0.45, 20000), np.arange(20000))
        self.assertLess(s2["lo"], 0.0)
        self.assertGreater(s2["hi"], 0.0)


def scan_job(out, seeds, hz=None, tag_every=2, mode="scan"):
    config = dict(fp_ckpt=str(Path(ARGS.ckpt) / "fp_win"), sh_ckpt=str(Path(ARGS.ckpt) / "sh_win"), max_turns=40, hazards=hz or rich_hazards(), view_keep=1.0, tag_every=tag_every)
    return dvgen.job_for(config, out, mode, 0, time.strftime("%H%M%S"), seeds), config


class TestRealScan(unittest.TestCase):
    """Real games, real nets, the native engine."""
    out = None

    @classmethod
    def setUpClass(cls):
        cls.out = Path(ARGS.tmp) / ("scan_%s" % time.strftime("%H%M%S"))
        (cls.out / "scan" / "shards").mkdir(parents=True)
        job, config = scan_job(cls.out / "scan", [dict(seed=SEED0 + i) for i in range(ARGS.games)])
        cls.worker = World.run_job(job)
        config.update(games=ARGS.games, seed0=SEED0, temperature=0.0, tag_rates=dict(D.TAG_RATES))
        C.write_json(cls.out / "scan" / "config.json", config)
        cls.games, cls.forks = R.scan_arrays(cls.out)
        cls.config = config

    def test_every_game_written_once(self):
        self.assertEqual(sorted(self.games["seed"].tolist()), [SEED0 + i for i in range(ARGS.games)])

    def test_all_streams_fired(self):
        fired = set(self.forks["stream"].tolist())
        self.assertEqual(fired, {D.BAND, D.KIND, D.EXACT_AA, D.RESEED_AA, D.HANDICAP})

    def test_base_game_is_unperturbed_by_forking(self):
        for seed, winner, plies in zip(self.games["seed"][:6], self.games["winner"][:6], self.games["plies"][:6]):
            w, p, _t = plain_game(int(seed))
            self.assertEqual((w, p), (int(winner), int(plies)), "seed %d" % seed)

    def test_fork_equals_replay_with_the_row_forced(self):
        """The causal claim, directly: a forked arm is the game you would get replaying the seed and forcing that row."""
        rows = np.flatnonzero((self.forks["stream"] == D.BAND) & (self.forks["status"] == D.OK))
        for i in rows[:: max(1, len(rows) // 10)][:10]:
            winner, plies, trace = plain_game(int(self.forks["seed"][i]), force=(int(self.forks["ply"][i]), int(self.forks["alt"][i])))
            self.assertEqual((winner, plies), (int(self.forks["winner"][i]), int(self.forks["end_ply"][i])), "fork %d" % i)
            self.assertEqual(trace, int(self.forks["trace"][i]))

    def test_exact_aa_reproduces_the_game(self):
        aa = (self.forks["stream"] == D.EXACT_AA) & (self.forks["status"] == D.OK)
        self.assertGreater(aa.sum(), 5)
        self.assertTrue(np.all(self.forks["trace"][aa] == self.forks["base_trace"][aa]))
        self.assertTrue(np.all(self.forks["winner"][aa] == self.forks["base_winner"][aa]))
        self.assertTrue(np.all(self.forks["end_ply"][aa] == self.forks["base_plies"][aa]))

    def test_reseeded_aa_differs_and_is_marked(self):
        ra = (self.forks["stream"] == D.RESEED_AA) & (self.forks["status"] == D.OK)
        self.assertGreater(ra.sum(), 5)
        self.assertTrue((self.forks["trace"][ra] != self.forks["base_trace"][ra]).any())
        self.assertTrue(np.mean(self.forks["end_ply"][ra] != self.forks["base_plies"][ra]) > 0.2)

    def test_handicap_arms_play_other_games(self):
        hc = (self.forks["stream"] == D.HANDICAP) & (self.forks["status"] == D.OK)
        self.assertGreater(hc.sum(), 0)
        self.assertGreater(np.mean(self.forks["end_ply"][hc] != self.forks["base_plies"][hc]), 0.3)

    def test_lowest_eligible_is_the_lowest_row_at_the_floor(self):
        lp = np.log(np.array([0.7, 0.2, 0.0999, 0.00009, 0.00001], np.float64))
        self.assertEqual(D.lowest_eligible(lp, 0), 2)
        self.assertEqual(D.lowest_eligible(np.log(np.array([0.99995, 0.00004, 0.00001])), 0), 0)

    def test_forks_are_sane(self):
        real = np.isin(self.forks["stream"], (D.BAND, D.KIND)) & (self.forks["status"] == D.OK)
        self.assertTrue(np.all(self.forks["p_alt"][real] <= self.forks["p_top"][real] + 1e-6))
        self.assertTrue(np.all(self.forks["p_alt"][real] >= D.FLOOR * 0.999))
        self.assertTrue(np.all(self.forks["rank"][real] >= 1))
        self.assertTrue(np.all(self.forks["ply"] < self.forks["base_plies"]))
        band = (self.forks["stream"] == D.BAND)
        self.assertTrue(np.all(np.searchsorted(D.ALT_EDGES, self.forks["p_alt"][band], side="right") - 1 == self.forks["alt_band"][band]))

    def test_a_game_with_no_hazard_has_no_fork_and_the_same_result(self):
        hz = D.hazards(json.loads((HERE / "census_freq.json").read_text(encoding="utf-8")), dict(band=0.0, kind=0.0, exact_aa=0.0, reseed_aa=0.0, handicap=0.0, candidate=0.0))
        out = Path(ARGS.tmp) / ("nofork_%s" % time.strftime("%H%M%S"))
        job, _c = scan_job(out, [dict(seed=SEED0 + 1)], hz=hz, tag_every=0)
        World.run_job(job)
        shard = D.shard_paths(out / "shards")[0]
        g = D.read_table(shard, "games/", D.GAME_FIELDS)
        self.assertEqual(int(g["forks"][0]), 0)
        i = int(np.flatnonzero(self.games["seed"] == SEED0 + 1)[0])
        self.assertEqual(int(g["winner"][0]), int(self.games["winner"][i]))
        self.assertEqual(int(g["plies"][0]), int(self.games["plies"][i]))

    def test_views_line_up_with_what_the_next_decision_shows(self):
        """A kept fork's role 2 view is the mover's view at the arm's next decision."""
        parts = [D.read_table(p, "views/", D.VIEW_FIELDS) for p in D.shard_paths(self.out / "scan" / "shards")]
        v = parts[0]
        self.assertGreater(len(v["a"]), 0)
        rows = np.flatnonzero((v["role"] == 2) & (v["term"] == 0))
        self.assertGreater(len(rows), 0)
        i = int(rows[0])
        seed, fid = int(v["a"][i]), int(v["b"][i])
        m = (self.forks["seed"] == seed) & (self.forks["fid"] == fid)
        ply, alt, side = int(self.forks["ply"][m][0]), int(self.forks["alt"][m][0]), int(self.forks["side"][m][0])
        game, _p, _s, _lp, _inp = plain_game(seed, stop=ply)
        game.apply(alt)
        self.assertTrue(game.next())
        obs, watch, *_rest = game.encode()
        vec = obs if D.SIDE_CODE[game.side] == side else watch
        game.close()
        idx, val = V.sparse(vec)
        a, b = v["o0"][i], v["o0"][i + 1]
        np.testing.assert_array_equal(idx, v["oidx"][a:b])
        np.testing.assert_array_equal(val, v["oval"][a:b])

    def test_check_scan_passes_real_data_and_fails_broken_data(self):
        """Mutation tests of the abort logic: every failure below is a check that fires on the bug it names."""
        fails, _w, fig = dvgen.check_scan(self.out, min_games=ARGS.games + 1)
        hard = [f for f in fails if "T=0 baseline" not in f and "forks per game" not in f]    # a few local games and rich hazards
        self.assertEqual(hard, [], fig)

    def mutated(self, fn):
        games, forks = {k: v.copy() for k, v in self.games.items()}, {k: v for k, v in self.forks.items()}
        forks = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in forks.items()}
        fn(games, forks)
        real_load = dvgen.load_scan_tables
        dvgen.load_scan_tables = lambda out, kinds=False: (games, forks)
        minimum, dvgen.HANDICAP_MIN = dvgen.HANDICAP_MIN, 2
        try:
            return dvgen.check_scan(self.out, min_games=1)[0]
        finally:
            dvgen.load_scan_tables = real_load
            dvgen.HANDICAP_MIN = minimum

    def test_check_catches_a_broken_copy(self):
        def bad(g, f):
            f["trace"][f["stream"] == D.EXACT_AA] += 1
        self.assertTrue(any("exact A/A" in x for x in self.mutated(bad)))

    def test_check_catches_a_blind_instrument(self):
        def bad(g, f):
            m = f["stream"] == D.HANDICAP
            f["winner"][m] = f["base_winner"][m]
        self.assertTrue(any("handicap" in x for x in self.mutated(bad)))

    def test_check_catches_wrong_play(self):
        def bad(g, f):
            g["winner"][:] = 0
        self.assertTrue(any("T=0 baseline" in x for x in self.mutated(bad)))

    def test_check_catches_a_top_row_forced(self):
        def bad(g, f):
            m = f["stream"] == D.BAND
            f["alt"][m] = f["top"][m]
        self.assertTrue(any("equal to the top" in x for x in self.mutated(bad)))

    def test_check_catches_refusals(self):
        def bad(g, f):
            f["status"][:] = D.REFUSED
        self.assertTrue(any("refused" in x for x in self.mutated(bad)))

    def test_abandon_then_resume_loses_and_duplicates_nothing(self):
        out = Path(ARGS.tmp) / ("resume_%s" % time.strftime("%H%M%S"))
        seeds = [SEED0 + 100 + i for i in range(10)]
        job, _c = scan_job(out, [dict(seed=s) for s in seeds])
        job["flush_games"] = 1
        w = World.get()
        w.reset_stats()
        w.channel = Sink()
        w.counter.value = 0
        w.abandon.clear()
        w.load_job(job)
        threading.Timer(4.0, w.abandon.set).start()
        w.run()
        first = set(dvgen.recorded(D.shard_paths(out / "shards")))
        self.assertLess(len(first), len(seeds), "the abandon came too late to test anything")
        job2, _c = scan_job(out, [dict(seed=s) for s in seeds if s not in first])
        job2["session"] = job["session"] + "b"
        World.run_job(job2)
        both = dvgen.recorded(D.shard_paths(out / "shards"))
        self.assertEqual(sorted(both), sorted(seeds))

    def test_no_file_outside_the_output_folder(self):
        self.assertTrue(Path(ARGS.tmp).is_dir())
        self.assertEqual(Path(os.getcwd()).resolve(), Path(ARGS.start_cwd).resolve())


class TestRealRollouts(unittest.TestCase):
    out = None

    @classmethod
    def setUpClass(cls):
        cls.out = Path(ARGS.tmp) / ("roll_%s" % time.strftime("%H%M%S"))
        (cls.out / "scan" / "shards").mkdir(parents=True)
        job, config = scan_job(cls.out / "scan", [dict(seed=SEED0 + 300 + i) for i in range(6)], tag_every=1)
        World.run_job(job)
        config.update(games=6, seed0=SEED0 + 300, temperature=0.0, tag_rates=dict(D.TAG_RATES))
        C.write_json(cls.out / "scan" / "config.json", config)
        ns = types.SimpleNamespace(out=str(cls.out), stage=1, n_general=4, n_named=1, k1=8, k2=8, k_promote=-9.0, max_promoted=4, random_share=0.25, procs=0)
        dvgen.plan_roll(ns)
        body = json.loads((cls.out / "roll1" / "positions.json").read_text(encoding="utf-8"))
        cls.body = body
        # a position with its identical arm is what the rollout A/A needs; force one for a certain test
        c = body["positions"][0]
        if not any(a["source"] == "aa" for a in c["arms"]):
            c["arms"].append(dict(arm=len(c["arms"]), row=c["top"], source="aa", p=c["p_top"], kind=c["top_kind"], label=c["top_label"]))
            C.write_json(cls.out / "roll1" / "positions.json", body)
            dvgen.plan_roll(ns)
        items = json.loads((cls.out / "roll1" / "pending.json").read_text(encoding="utf-8"))
        cls.items = items
        job = dvgen.job_for(config, cls.out / "roll1", "roll", 0, "r1", items)
        cls.worker = World.run_job(job)

    def test_positions_chosen_once_and_stratified(self):
        sides = {c["side"] for c in self.body["positions"]}
        self.assertEqual(sides, {"fp", "sh"})
        self.assertTrue(all(c["p_top"] >= 0.8 for c in self.body["positions"] if c["sample"].startswith("general")))
        self.assertEqual(self.body["k"], 8)

    def test_rollout_results_complete_and_paired(self):
        body, rows, sides = dvgen.read_rolls(self.out / "roll1")
        table = dvgen.pair_table(rows, sides)
        c = body["positions"][0]
        self.assertIn(c["pid"], table)
        self.assertEqual(len(table[c["pid"]]), len(c["arms"]))
        for arm in c["arms"]:
            ks, _s = table[c["pid"]][arm["arm"]]
            self.assertEqual(ks.tolist(), list(range(8)))

    def test_rollout_aa_is_identical_by_construction(self):
        fails, _w, fig = dvgen.check_roll(self.out, "roll1")
        self.assertEqual(fails, [], fig)
        self.assertEqual(fig.get("aa_identical"), 1.0)
        self.assertGreater(fig.get("aa_rollouts", 0), 0)

    def test_rollouts_are_not_all_the_same_game(self):
        _body, rows, _sides = dvgen.read_rolls(self.out / "roll1")
        self.assertGreater(len(set(rows["trace"].tolist())), len(rows["trace"]) / 4)

    def test_replay_ending_before_its_position_is_a_mismatch(self):
        """Real on the cluster: a tie flip can end the replayed game early; finish_arm must not treat it as a rollout."""
        item = dict(self.items[0], id="late:0", ply=10 ** 6)
        job = dvgen.job_for(self.config(), self.out / "late", "roll", 0, "t", [item])
        w = World.run_job(job)
        self.assertEqual(w.n["replay_mismatch"], 1)
        meta = D.read_meta(D.shard_paths(self.out / "late" / "shards")[0])
        self.assertEqual(meta["chunks"], ["late:0!mismatch"])

    def test_replay_mismatch_is_detected_not_silently_used(self):
        item = dict(self.items[0], id="tamper:0", pos="12345")
        job = dvgen.job_for(self.config(), self.out / "tamper", "roll", 0, "t", [item])
        w = World.run_job(job)
        self.assertEqual(w.n["replay_mismatch"], 1)
        meta = D.read_meta(D.shard_paths(self.out / "tamper" / "shards")[0])
        self.assertEqual(meta["chunks"], ["tamper:0!mismatch"])

    def config(self):
        return json.loads((self.out / "scan" / "config.json").read_text(encoding="utf-8"))

    def test_determinize_redeals_only_what_the_mover_cannot_see(self):
        c = self.body["positions"][0]
        game, ply, _s, _lp, _inp = plain_game(c["seed"], stop=c["ply"])
        w = World.get()
        ai = w.ai
        mine, theirs = (ai.gs.Side.FREE_PEOPLES, ai.gs.Side.SHADOW) if c["side"] == "fp" else (ai.gs.Side.SHADOW, ai.gs.Side.FREE_PEOPLES)
        before = game.state(ai)
        info = game.world_info()
        changed = 0
        for k in range(8):
            world = game.world_clone(D.stream_state(c["seed"], c["ply"], k), info[5] - info[2])
            world.world_determinize(D.SIDE_CODE[c["side"]], D.determinize_rng(c["seed"], c["ply"], k))
            after = world.state(ai)
            self.assertEqual(list(before.hands[mine].hand), list(after.hands[mine].hand))
            self.assertEqual(before.fellowship.progress, after.fellowship.progress)
            changed += list(before.hands[theirs].hand) != list(after.hands[theirs].hand)
            world.close()
        game.close()
        self.assertGreater(changed, 0)            # sensitivity: the re-deal does something

    def test_clone_with_the_game_stream_equals_a_copy(self):
        """world_clone exposes forced frames; the loop skips them.  With the same stream it must play the same game as a copy."""
        c = self.body["positions"][0]
        game, ply, top, _lp, _inp = plain_game(c["seed"], stop=c["ply"])
        info = game.world_info()
        a, b = game.world_copy(), game.world_clone(game.rng_state(), info[5] - info[2])
        a.world_apply_from(game, top)
        b.world_apply_from(game, top)
        w = World.get()
        results = []
        for g in (a, b):
            trace = 0
            while True:
                more = g.next()
                while more and g.forced:
                    g.apply(0)
                    more = g.next()
                if not more:
                    break
                obs, watch, cand, regions, kinds = g.encode()
                p = w.actor(g.side).predict_many([w.ActorInput(obs, cand, regions, w.zeros(len(cand)))])[0]
                s = int(np.argmax(p.log_probabilities))
                trace = D.trace_step(trace, len(cand), s)
                g.apply(s)
            results.append((D.winner_code(g.end), trace))
            g.close()
        game.close()
        self.assertEqual(results[0], results[1])

    def test_stage_two_and_the_report(self):
        ns = types.SimpleNamespace(out=str(self.out), stage=2, n_general=4, n_named=1, k1=8, k2=16, k_promote=-9.0, max_promoted=4, random_share=0.25, procs=0)
        self.assertEqual(dvgen.plan_roll(ns), 0)
        items = json.loads((self.out / "roll2" / "pending.json").read_text(encoding="utf-8"))
        body = json.loads((self.out / "roll2" / "positions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(body["positions"]), 4)
        self.assertEqual(sum(1 for c in body["positions"] if c["sample2"] == "random"), 1)
        self.assertGreaterEqual(items[0]["k0"], 1000000)         # fresh rollouts: no k is reused
        self.assertEqual({a["arm"]: a["source"] == "top" for a in body["positions"][0]["arms"]}, {0: False, 1: True})
        dvgen_job = dvgen.job_for(self.config(), self.out / "roll2", "roll", 0, "r2", items)
        World.run_job(dvgen_job)
        self.assertEqual(dvgen.plan_roll(ns), 10)
        fails, _w, fig = dvgen.check_roll(self.out, "roll2")
        self.assertEqual(fails, [], fig)
        import dvvalue
        argv = sys.argv
        sys.argv = ["dvvalue.py", "--out", str(self.out), "--train", ARGS.vnet, "--device", "cpu", "--batch", "256"]
        try:
            self.assertEqual(dvvalue.main(), 0)
        finally:
            sys.argv = argv
        parity = json.loads((self.out / "values" / "PARITY.json").read_text(encoding="utf-8"))
        self.assertLess(parity["max"], 1e-4)
        text, js = R.write_report(self.out, resamples=100)
        for needle in ("Controls", "Mode A", "Mode B", "Named-case", "cannot show", "c200_head", "full_fresh_mc"):
            self.assertIn(needle, text)
        self.assertIn("primary", js)
        self.assertTrue((self.out / "report" / "report.md").is_file())
        print("\n--- report on the tiny real run ---\n" + text[:3000], flush=True)


class TestTwinAA(unittest.TestCase):
    """aa.flux: real games with every firing decision cloned twice; aa_report.py reads them and its reading follows planted answers."""

    def test_twin_end_to_end(self):
        import aa_report
        out = Path(ARGS.tmp) / ("aa_%s" % time.strftime("%H%M%S"))
        (out / "scan" / "shards").mkdir(parents=True)
        freq = json.loads((HERE / "census_freq.json").read_text(encoding="utf-8"))
        hz = D.hazards(freq, dict(band=0, kind=0, exact_aa=0.3, reseed_aa=1.0, handicap=0, candidate=0, reseed_twin=1))
        self.assertEqual(hz["band"]["fp"][0][0], 0.0)
        seeds = [SEED0 + 900 + i for i in range(ARGS.games)]
        job, config = scan_job(out / "scan", [dict(seed=s) for s in seeds], hz=hz, tag_every=1000000)
        config["view_keep"] = 0.0
        job["view_keep"] = 0.0
        World.run_job(job)
        config.update(games=len(seeds), seed0=seeds[0], temperature=0.0, tag_rates=dict(D.TAG_RATES))
        C.write_json(out / "scan" / "config.json", config)
        games, forks = dvgen.load_scan_tables(out)
        a, b = forks["stream"] == D.RESEED_AA, forks["stream"] == D.RESEED_B
        self.assertGreater(a.sum(), 0)
        self.assertEqual(a.sum(), b.sum())                                     # always in twos
        self.assertEqual(sorted(zip(forks["seed"][a].tolist(), forks["ply"][a].tolist())), sorted(zip(forks["seed"][b].tolist(), forks["ply"][b].tolist())))
        self.assertFalse(np.isin(forks["stream"], [D.BAND, D.KIND, D.HANDICAP]).any())
        self.assertTrue((forks["alt"][a | b] == forks["top"][a | b]).all())    # the forced row is the top move
        self.assertGreater(float((forks["trace"][a] != forks["trace"][b]).mean()) + float((forks["end_ply"][a] != forks["end_ply"][b]).mean()), 0)   # two streams give two games
        sys.argv = ["aa_report.py", "--out", str(out)]
        aa_report.main()
        body = json.loads((out / "AA_REPORT.json").read_text(encoding="utf-8"))
        self.assertIn("b_a", body["new"])
        self.assertEqual(body["new"]["b_a"]["B - A all"]["n"], int(a.sum()))


class TestPool(unittest.TestCase):
    """lgnode.Pool with spawned workers, as on a node (one or two CPU workers here)."""

    def test_pool_scan_then_deadline_then_resume(self):
        code = (
            "import os,sys,json,types;"
            "sys.path.insert(0,%r);"
            "import test_deviation as T;"
            "T.pool_main(%r,%r)"
        ) % (str(HERE), str(ARGS.ckpt), str(ARGS.tmp))
        r = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, timeout=600, cwd=str(ARGS.tmp))
        tail = r.stdout[-2500:] + r.stderr[-2500:]
        self.assertEqual(r.returncode, 0, tail)
        self.assertIn("POOL OK", r.stdout, tail)


def serve_tolerant(*args):
    C.setup_paths()
    tolerate_net_tags()
    import dvworker
    return dvworker.serve_main(*args)


def pool_main(ckpt, tmp):
    import batchgen
    import lgnode
    import lgworker
    out = Path(tmp) / ("pool_%s" % time.strftime("%H%M%S"))
    plan = types.SimpleNamespace(out=str(out), games=24, seed0=SEED0 + 500, fp_ckpt=str(Path(ckpt) / "fp_win"), sh_ckpt=str(Path(ckpt) / "sh_win"),
                                 census=str(HERE / "census_freq.json"), view_keep=0.3, tag_every=10, procs=0)
    lgworker.serve_main = serve_tolerant
    tree = C.setup_paths()
    os.environ["WOTR_TREE"] = str(tree)
    from selfplay_actor_critic.generation import require_generation
    generation = require_generation("dv pool test")
    opts = batchgen.own_parser().parse_args(["--device", "cpu", "--cpu-procs", "1", "--cpu-lockstep", "2"])
    pool = lgnode.Pool(batchgen.make_plan(opts)["specs"], dict(tree=str(tree), generation=generation, actor_cache=4))
    try:
        for attempt, stop_after in ((1, 10.0), (2, 0.0)):
            if dvgen.plan_scan(plan) == 10:
                break
            config = json.loads((out / "scan" / "config.json").read_text(encoding="utf-8"))
            pending = np.load(out / "scan" / "pending.npy")
            abandon = out / (".abandon_%d" % attempt)
            if stop_after:
                threading.Timer(stop_after, lambda: abandon.write_text("deadline\n")).start()
            job = dvgen.job_for(config, out / "scan", "scan", 0, "p%d" % attempt, [dict(seed=int(s)) for s in pending])
            job["flush_games"] = 2
            shards, refusals, reports, errors, abandoned = pool.run_job(job, abandon)
            print("attempt %d: %d pending, %d games, errors %s, abandoned %s" % (attempt, len(pending), sum(len(s["games"]) for s in shards), errors, abandoned), flush=True)
            if errors:
                raise SystemExit("pool failed")
    finally:
        pool.close()
    seeds = dvgen.recorded(D.shard_paths(out / "scan" / "shards"))
    if sorted(seeds) != list(range(SEED0 + 500, SEED0 + 524)):
        raise SystemExit("lost or duplicated games: %s" % sorted(seeds))
    print("POOL OK", flush=True)


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vnet", required=True)
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--only", default="")
    ARGS, rest = ap.parse_known_args()
    ARGS.ckpt, ARGS.vnet, ARGS.tmp = str(D.absolute(ARGS.ckpt)), str(D.absolute(ARGS.vnet)), str(D.absolute(ARGS.tmp))
    ARGS.start_cwd = os.getcwd()
    Path(ARGS.tmp).mkdir(parents=True, exist_ok=True)
    watched = [Path(os.getcwd()), HERE, C.tree_root() / "src", C.tree_root()]        # board.py chdirs to src/: nothing may land there
    before = {w: set(os.listdir(w)) for w in watched}
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    for case in (TestPure, TestStatistics, TestRealScan, TestRealRollouts, TestTwinAA, TestPool):
        if not ARGS.only or case.__name__ in ARGS.only.split(","):
            suite.addTests(loader.loadTestsFromTestCase(case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    stray = {str(w): sorted(set(os.listdir(w)) - before[w]) for w in watched if set(os.listdir(w)) - before[w]}
    print("stray files: %s" % (stray or "none"), flush=True)
    return 0 if result.wasSuccessful() and not stray else 1


if __name__ == "__main__":
    raise SystemExit(main())
