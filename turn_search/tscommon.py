"""Shared set-up of the turn-search experiments: the fast engine from the scratch copy (with the "start at a position"
entry), the league worker's net loading, and small helpers."""
from pathlib import Path
import os
import random
import sys

HERE = Path(__file__).resolve().parent
DEVICE = os.environ.get("WOTR_TS_DEVICE", "cpu")     # where the nets run: "cpu" or "cuda" (the engine is always on the CPU)
GPU = os.environ.get("WOTR_TS_GPU", "0")             # which GPU, as CUDA_VISIBLE_DEVICES names it (the service: ts_device.py)
# Where things are.  A PACKED kit (pack_game_kit.py: the copy a game runs) holds the files of the development tree it
# reads in its own folder "tree", at their places in the tree; the kit in the development tree uses the tree it sits
# in.  On the cluster the job sets the three variables (ts_arms_node.sh):
#   WOTR_TS_TREE       the development tree
#   WOTR_TS_ENGINE     the folder whose native_env.py (fast engine) is used; only this kit's scratch copy has the
#                      "start at a position" entry the live service needs, full games need no such entry
#   WOTR_TS_DEVIATION  the folder of test_deviation.py (the league's in-process worker)
PACKED = (HERE / "tree").is_dir()
TREE = Path(os.environ.get("WOTR_TS_TREE") or (HERE / "tree" if PACKED else HERE.parents[1]))
ENGINE = Path(os.environ.get("WOTR_TS_ENGINE") or HERE / "native_w2x").resolve()
for key, val in dict(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", KMP_DUPLICATE_LIB_OK="TRUE", WOTR_ACT_GEN="v53",
                     CUDA_VISIBLE_DEVICES="-1" if DEVICE == "cpu" else GPU, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WOTR_NET_ARCH="base",
                     WOTR_POLICY_DELTA="1", WOTR_TREE=str(TREE)).items():     # WOTR_TREE: the engine wrapper and the league's modules take the tree from it
    os.environ[key] = val
if ENGINE == (HERE / "native_w2x").resolve():
    os.environ["NATIVE_RAW_DEV_ANY_DLL"] = "1"       # the rebuilt DLL of the scratch copy
sys.dont_write_bytecode = True
sys.path.insert(0, str(ENGINE))
import native_env                                   # cached, so every later import gets this one
assert Path(native_env.__file__).resolve().parent == ENGINE
sys.path.insert(0, os.environ.get("WOTR_TS_DEVIATION") or str(TREE / "hpc" / "remote" / "league" / "deviation"))
import test_deviation as T
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if DEVICE == "cpu" else GPU     # that module hides the GPU on import
import dvcommon as D
import numpy as np
import torch

torch.set_num_threads(1)
CKPT = HERE.parent / "c200"
VALUE = TREE / "claude_bugscan_2026-10-21" / "valuenet_1102" / "ckpt"


def world(ckpts=None):
    w = T.World.get(device="cpu" if DEVICE == "cpu" else "cuda:0")
    w.ckpts = ckpts or {s: (str(CKPT / (s + "_win")), s) for s in ("fp", "sh")}
    w.max_turns = 40
    assert Path(w.native_env.__file__).resolve().parent == ENGINE
    return w


def net_table(settings, policy, values):
    """The nets of search_settings.json ("nets": side -> key -> {"policy": folder, "judge": value file | "own"}) as the
    kit uses them -> (nets, judges, seats):
      nets    name -> (folder, side): the world's net table.  A side's "win" net is named like the side - it is the net
              the OTHER side's playouts are answered with - every other net by its folder;
      judges  name -> (side, file): the value that judges for a seat playing that net - a value net's file, or ("own")
              the net's own file, whose value head the league trained on that net's own objective;
      seats   (side, key) -> name."""
    nets, judges, seats = {}, {}, {}
    for side, table in settings["nets"].items():
        for key, spec in table.items():
            name = side if key == "win" else spec["policy"]
            folder = Path(policy) / spec["policy"]
            nets[name] = (str(folder), side)
            judges[name] = (side, str(folder / (side + "_seed.pt")) if spec["judge"] == "own" else str(Path(values) / spec["judge"]))
            seats[(side, key)] = name
    return nets, judges, seats


def advisers_of(seats, side, key):
    """The advisers of a seat: the other nets of its side."""
    return tuple(name for (s, k), name in seats.items() if s == side and k != key)


def logp(w, game, enc=None, net=None):
    """The log priors of the open decision's menu under the mover's own net, or under another net of that side."""
    obs, watch, cand, regions, kinds = enc or game.encode()
    p = w.actor(net or game.side).predict_many([w.ActorInput(obs, cand, regions, w.zeros(len(cand)))])[0]
    return np.asarray(p.log_probabilities, np.float64)


SPLIT = hasattr(native_env.NativeRawGame, "encode_view")     # this kit's engine copy has the parts of the encoding on
                                                             # their own (native_w2x/TURNSEARCH.diff); the certified engine has not


def decision(game):
    """What a net needs to move at the open decision: (the mover's observation, the menu's rows, their regions)."""
    if SPLIT:
        return (game.encode_view(game.side),) + game.encode_menu()
    obs, _watch, cand, regions, _kinds = game.encode()
    return obs, cand, regions


def view(game, side):
    """`side`'s observation of the open decision (what its value net judges)."""
    if SPLIT:
        return game.encode_view(side)
    obs, watch = game.encode()[:2]
    return obs if game.side == side else watch


def load_value(ai, side, kind="full_fresh_mc", path=None):
    path = Path(path) if path else VALUE / (side + "_" + kind) / "best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state", ck.get("model_state_dict", ck))
    net = ai.WOTRValueNet(int(ck.get("input_dim", 4745)), side=side, action_dim=int(ck.get("action_dim", 199)))
    act = ck.get("head_act") or "tanh"
    net.value_head[-1] = ai._ScaledTanhHead(2) if act == "scaled2" else torch.nn.Tanh()
    ai._install_selfplay_policy_modules(net, sd, str(path))
    net.load_state_dict(sd, strict=False)
    net.eval()
    return net if DEVICE == "cpu" else net.to("cuda:0")


def load_values(ai, judges):
    """judges: name -> (side, file) (net_table) -> name -> the value, loaded."""
    return {name: load_value(ai, side, path=path) for name, (side, path) in judges.items()}


def stream(*key):
    return random.Random(hash(tuple(int(k) for k in key)) & 0xFFFFFFFFFFFF).getstate()
