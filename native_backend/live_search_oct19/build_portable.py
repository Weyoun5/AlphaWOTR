"""Build this box's native library with clang or gcc, for a PC that is not Windows (macOS, Linux).

    python build_portable.py                 build for this PC, into platform/<tag>/ beside this file
    python build_portable.py --check         say whether that build is there and whole; build nothing

Windows keeps the box's own MSVC library (build.py, build/manifest.json).  Every other platform gets a RUNTIME BOX of
its own beside it,

    platform/<tag>/build/manifest.json                    what the loader reads (native_backend/bridge.py library())
    platform/<tag>/build/<identity>/libwotr_native.dylib  (.so on Linux)

where <tag> is sys.platform + "-" + the machine of the Python that will load it (darwin-arm64, darwin-x86_64,
linux-x86_64).  The loaders pick the runtime box by that same tag; nothing of the Windows box is changed.

THE NUMBERS MUST BE THE ENGINE'S: no fused multiply-add (-ffp-contract=off; arm64 has it always and clang uses it by
default), no fast-math, and pow / log taken from the platform's libm, never folded by the compiler
(-fno-builtin-pow -fno-builtin-log) - the flags the Linux build of the sibling box was proven with.  The search's
results on one platform are compared with the Python engine ON THAT PLATFORM, not with another platform's binary.

THE ALLOCATOR IS THE PLATFORM'S.  The Windows library replaces operator new / delete with mimalloc, linked against a
C++ runtime of its own.  Here the C++ runtime is the system's shared one (Apple's libc++): memory the runtime
allocates and this library frees, or the reverse, would cross two allocators, and a replaced operator new can reach
other C++ code in the same process (torch).  So the unit that replaces them (cpp/allocator_mimalloc.cpp) is not
compiled, and mimalloc is not linked: a speed choice of the Windows build, never a part of what is computed.

The manifest lists no sources for the loader to re-hash (as an installed runtime box): the build's own sources are
recorded under "build_sources_sha256", and the library by its sha256, which the loader checks before it loads it.
--cxx / --flags / --tag / --no-load exist for building for another platform than this PC's (a compile check).
"""
import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NUMERIC = ["-ffp-contract=off", "-fno-fast-math", "-fno-builtin-pow", "-fno-builtin-log"]
CXX_FLAGS = ["-std=c++17", "-O2", "-fexceptions", "-fvisibility=hidden", "-fvisibility-inlines-hidden", "-fPIC"]


def tag():
    """This PC's platform, as the loaders name it."""
    return "%s-%s" % (sys.platform, platform.machine().lower())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def units():
    """The box's translation units: the game, the search or raw generator it carries, the random stream (not the
    Windows build's allocator unit: see the top)."""
    middle = "raw_game.cpp" if (HERE / "cpp" / "raw_game.cpp").is_file() else "snapshot.cpp"
    return [HERE / "cpp" / n for n in ("position.cpp", middle, "random_stream.cpp")]


def closure(files):
    """The units and every file of the box they include (as build.py's dependencies())."""
    found, pending = set(), list(files)
    while pending:
        p = pending.pop().resolve()
        if p in found:
            continue
        found.add(p)
        for name in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"', p.read_text(encoding="utf-8"), re.MULTILINE):
            dep = next((x for x in (p.parent / name, HERE / "include" / name) if x.is_file()), None)
            if dep is None:
                raise SystemExit("missing include: %s (in %s)" % (name, p))
            pending.append(dep)
    return found


def sources():
    files = closure(units()) | {Path(__file__).resolve()}
    files.update(p for p in (HERE / "portable").rglob("*") if p.is_file())
    return {p.relative_to(HERE).as_posix(): sha(p) for p in sorted(files)}


def library_name(platform_tag):
    return "wotr_native.dll" if platform_tag.startswith("win") else \
        "libwotr_native.dylib" if platform_tag.startswith("darwin") else "libwotr_native.so"


def loads(path):
    """The library answers its ABI question in this process."""
    lib = ctypes.CDLL(str(path))
    lib.wotr_abi_version.argtypes, lib.wotr_abi_version.restype = [], ctypes.c_uint32
    return int(lib.wotr_abi_version())


def whole(box):
    """-> (the manifest, None) when the runtime box holds a library that is what its manifest says, else (None, why)."""
    try:
        manifest = json.loads((box / "build" / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return None, "no build there (%s)" % type(error).__name__
    binary = box / manifest.get("binary_relative", "")
    if manifest.get("passed") is not True or not binary.is_file():
        return None, "its manifest names no library that is there"
    if sha(binary) != manifest.get("binary_sha256"):
        return None, "the library is not the one its manifest records"
    if manifest.get("build_sources_sha256") != sources():
        return None, "the sources changed since it was built"
    return manifest, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--cxx", default=os.environ.get("CXX") or "c++")
    ap.add_argument("--flags", default="", help="more flags for both compilers (another target)")
    ap.add_argument("--tag", default=tag())
    ap.add_argument("--no-load", action="store_true", help="do not load the result here (it is for another platform)")
    a = ap.parse_args()
    box = HERE / "platform" / a.tag
    if a.check:
        manifest, why = whole(box)
        print("%s: %s" % (box, "built, %s" % manifest["binary_relative"] if manifest else why))
        return 0 if manifest else 1
    cxx, more = shlex.split(a.cxx), shlex.split(a.flags)
    if shutil.which(cxx[0]) is None:
        raise SystemExit("no C++ compiler found (%s).  On a Mac:  xcode-select --install" % cxx[0])
    if a.tag.startswith("darwin") and not more:
        more = ["-arch", a.tag.split("-", 1)[1]]                # the machine of the Python that will load it
    version = subprocess.run(cxx + ["--version"], capture_output=True, text=True).stdout.strip().splitlines()[:1]
    src = sources()
    include = ["-I" + str(HERE / "include")]
    if (HERE / "portable").is_dir():
        include.insert(0, "-I" + str(HERE / "portable"))
    link = ["-shared"]
    if a.tag.startswith("darwin"):
        link += ["-Wl,-undefined,error"]
    elif not a.tag.startswith("win"):
        link += ["-Wl,-z,defs", "-pthread"]
    identity = hashlib.sha256(json.dumps(
        {"sources": src, "tag": a.tag, "compiler": version, "flags": NUMERIC + CXX_FLAGS + more + link},
        sort_keys=True).encode()).hexdigest()[:20]
    out = box / "build" / identity
    out.mkdir(parents=True, exist_ok=True)
    binary = out / library_name(a.tag)
    command = cxx + more + CXX_FLAGS + NUMERIC + include + link + [str(p) for p in units()] + ["-o", str(binary)]
    started = time.perf_counter()
    r = subprocess.run(command, cwd=str(out), capture_output=True, text=True)
    (out / "build.log").write_text(" ".join(command) + "\n" + r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        raise SystemExit("the build failed; see %s\n%s" % (out / "build.log", (r.stdout + r.stderr)[-3000:]))
    abi = None if a.no_load else loads(binary)
    record = {"passed": True, "abi": 1, "snapshot_abi": 1, "sources": {}, "distribution": "runtime",
              "binary_relative": binary.relative_to(box).as_posix(), "binary_sha256": sha(binary),
              "build_sources_sha256": src, "platform": a.tag, "compiler": version, "loaded_abi": abi,
              "command": [c.replace(str(HERE), ".") for c in command],
              "allocator": "the platform's (no replacement of operator new / delete)",
              "seconds": round(time.perf_counter() - started, 1),
              "scope": "portable build of this box (build_portable.py); numerics: " + " ".join(NUMERIC)}
    if (HERE / "cpp" / "raw_game.cpp").is_file():
        record["raw_abi"] = 1
    for path in (out / "manifest.json", box / "build" / "manifest.json"):
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("BUILT %s (%s, %.0f s%s)" % (binary, " ".join(version), record["seconds"],
                                       "" if a.no_load else ", abi %s" % abi))
    return 0


if __name__ == "__main__":
    sys.exit(main())
