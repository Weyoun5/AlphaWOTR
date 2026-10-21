"""Build and validate a new source-generated candidate without publishing it."""

from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import time

RUNTIME_DATA_FILES = ("board200.jpg", "wotr_map.txt", "card_list.csv")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_candidate(source_root, output_dir, tooling_dir, action_generation="v41"):
    if action_generation not in ("v41", "v45", "v46", "v47", "v48", "v49", "v50", "v51", "v52"):
        raise ValueError("Native candidate action generation must be v41, v45, v46, v47, v48, v49, v50, v51 or v52")
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    tooling_dir = tooling_dir.resolve()
    if (
        not (source_root / "src/ai.py").is_file()
        or not (source_root / "src/board.py").is_file()
    ):
        raise ValueError("source root must contain src/ai.py and src/board.py")
    for reserved in (source_root / "src", source_root / "data"):
        if output_dir == reserved or reserved in output_dir.parents:
            raise ValueError("candidate output cannot be inside source src/data")
    runtime_data = [source_root / "data" / name for name in RUNTIME_DATA_FILES]
    missing_data = [path.name for path in runtime_data if not path.is_file()]
    if missing_data:
        raise FileNotFoundError(
            "required runtime data missing: " + ", ".join(missing_data)
        )
    # Existing candidate directories are never removed or overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    engine = output_dir / "engine"
    src = engine / "src"
    data = engine / "data"
    src.mkdir(parents=True)
    data.mkdir()
    sources = list((source_root / "src").glob("*.py")) + runtime_data
    artifacts = (
        list((source_root / "src").glob("wotr_native*.pyd"))
        + list((source_root / "src").glob("wotr_native*.so"))
        + list((source_root / "src").glob("wotr_native.pyx"))
    )
    before = {str(p): sha(p) for p in sources + artifacts}
    for path in sources:
        shutil.copy2(path, engine / path.relative_to(source_root))
    if not all(
        sha(engine / p.relative_to(source_root)) == before[str(p)] for p in sources
    ):
        raise RuntimeError("snapshot copy mismatch")
    if before != {p: sha(Path(p)) for p in before}:
        raise RuntimeError("source changed while copying")
    tools = output_dir / "tooling"
    tools.mkdir()
    for name in (
        "native_codegen.py",
        "native_build_candidate.py",
        "build_wotr_native.py",
    ):
        shutil.copy2(tooling_dir / name, tools / name)
    manifest = {
        "status": "running",
        "action_generation": action_generation,
        "source_root": str(source_root),
        "source_sha256_before": before,
        "tooling_sha256": {p.name: sha(p) for p in tools.iterdir()},
        "python": sys.version,
        "platform": sys.platform,
        "machine": platform.machine(),
        "extension_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "build_environment": {
            key: os.environ.get(key)
            for key in ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS")
        },
        "compiler_policy": "setuptools chooses the native platform compiler; no Windows suffix or MSVC command is hard-coded",
    }
    start = time.perf_counter()
    env = {k: v for k, v in os.environ.items() if not k.startswith("WOTR_")}
    env.update(
        WOTR_NATIVE="0",
        WOTR_ACT_GEN=action_generation,
        CUDA_VISIBLE_DEVICES="-1",
        KMP_DUPLICATE_LIB_OK="TRUE",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="0",
        PYTHONOPTIMIZE="0",
    )

    def run(label, command, where, native=False):
        childenv = dict(env, WOTR_NATIVE="1" if native else "0")
        result = subprocess.run(
            command, cwd=str(where), env=childenv, capture_output=True, timeout=180
        )
        (output_dir / (label + "_stdout.log")).write_bytes(result.stdout)
        (output_dir / (label + "_stderr.log")).write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{label} failed rc={result.returncode}; see saved logs")

    try:
        run(
            "generate",
            [sys.executable, str(tools / "native_codegen.py"), str(output_dir)],
            engine,
        )
        generation = json.loads((output_dir / "generation.json").read_text())
        if sha(src / "wotr_native.pyx") != generation["generated_pyx_sha256"]:
            raise RuntimeError("generated source changed before compilation")
        run("compile", [sys.executable, str(src / "setup_generated.py")], src)
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not suffix:
            raise RuntimeError("Python did not report an extension suffix")
        binary = src / ("wotr_native" + suffix)
        if not binary.is_file():
            raise RuntimeError(f"Expected native artifact absent: {binary.name}")
        # Standalone board load is verified before ai is imported. Importing ai
        # afterward must perform real binding; fallback cannot pass this check.
        probe = output_dir / "activation_probe.py"
        probe.write_text(
            """from pathlib import Path
import json
import sys

src = Path(sys.argv[1])
sys.path.insert(0, str(src))
import board
import wotr_native as n

assert board.sanitize is n.sanitize and not n._INITED
assert board.sanitize("Southrons & Easterlings") == "southrons and easterlings"
import ai
import game_state as gs
import game_logic as gl

assert ai.NATIVE_BOUND and n._INITED
assert ai.clone_state is n.clone_state and ai.encode_state is n.encode_state
assert ai.encode_state.__module__ == "wotr_native"
s = gs.setup_initial_game(board.load_board())
gl.initialize_decks(s)
gl.begin_fellowship_tail(s)
assert ai.clone_state(s)._eye_allocation_phase
assert len(ai.encode_state(s, gs.Side.FREE_PEOPLES)) == ai.get_feature_dim()
print(
    json.dumps(
        {
            "standalone_board_native": True,
            "native_bound": True,
            "width": ai.get_feature_dim(),
            "action_width": ai.ACTION_ENCODING_DIM,
            "observation_contract": ai.OBSERVATION_CONTRACT,
            "action_generation": ai.ACT_GEN,
            "artifact": n.__file__,
        }
    )
)
"""
        )
        run("activation", [sys.executable, str(probe), str(src)], engine, native=True)
        oracle = output_dir / "oracle_probe.py"
        oracle.write_text(
            """from pathlib import Path
import hashlib
import json
import random
import sys

src = Path(sys.argv[1])
sys.path.insert(0, str(src))
import ai
import board
import game_state as gs
import game_logic as gl

random.seed(9544)
base = gs.setup_initial_game(board.load_board())
gl.initialize_decks(base)
states = [ai.clone_state(base)]
gl.begin_fellowship_tail(base)
states.append(ai.clone_state(base))
for action in ai.get_ai_actions(ai.clone_state(base), gs.Side.FREE_PEOPLES):
    if action.action_type == "choose_guide":
        child = ai.clone_state(base)
        ai._execute_action(child, action)
        states.append(child)


def digest(value):
    return hashlib.sha256(value).hexdigest()


cases = []
for state in states:
    for side in (gs.Side.FREE_PEOPLES, gs.Side.SHADOW):
        actions = ai.get_ai_actions(ai.clone_state(state), side)
        cases.append(
            {
                "observation": digest(ai.encode_state(state, side).tobytes()),
                "actions": [
                    (repr(a), digest(ai.encode_action(a, state, side).tobytes()))
                    for a in actions
                ],
            }
        )
Path(sys.argv[2]).write_text(
    json.dumps({"width": ai.get_feature_dim(), "cases": cases}, sort_keys=True)
)
"""
        )
        for native in (False, True):
            label = "oracle_native" if native else "oracle_python"
            run(
                label,
                [
                    sys.executable,
                    str(oracle),
                    str(src),
                    str(output_dir / (label + ".json")),
                ],
                engine,
                native=native,
            )
        if (output_dir / "oracle_python.json").read_bytes() != (
            output_dir / "oracle_native.json"
        ).read_bytes():
            raise RuntimeError("independent Python/native candidate oracle mismatch")
        if sha(src / "wotr_native.pyx") != generation["generated_pyx_sha256"]:
            raise RuntimeError("generated source changed after compilation")
        manifest.update(
            status="passed",
            artifact=str(binary),
            artifact_sha256=sha(binary),
            generated_pyx_sha256=sha(src / "wotr_native.pyx"),
            independent_python_native_oracle=True,
        )
    except Exception as error:
        manifest.update(status="failed", error=repr(error))
        raise
    finally:
        manifest["source_sha256_after"] = {p: sha(Path(p)) for p in before}
        manifest["originals_unchanged"] = before == manifest["source_sha256_after"]
        manifest["seconds"] = time.perf_counter() - start
        (output_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    if not manifest["originals_unchanged"]:
        raise RuntimeError("source inputs changed during candidate build")
    print(
        json.dumps(
            {
                key: manifest[key]
                for key in (
                    "status",
                    "artifact",
                    "artifact_sha256",
                    "originals_unchanged",
                    "seconds",
                )
            },
            indent=2,
        )
    )
    return manifest
