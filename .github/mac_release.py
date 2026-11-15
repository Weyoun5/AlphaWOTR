"""Make the Mac download, on a Mac (Apple silicon), from this checkout:  AlphaWOTR-mac.zip

    python3 .github/mac_release.py --out DIR        (DIR must not exist yet)

The Windows zip is made on the Windows PC from the installed game, and the repository tracks that zip's files
without its python\\ runtime.  The Mac download is those tracked files as checked out here, without Windows' start
files, plus what a Mac needs to run them with nothing installed but Java:

    python/                        a Python of its own (RUNTIME: a relocatable build, taken by its sha256) with the
                                   packages of requirements.txt, torch among them
    <box>/platform/darwin-arm64/   the two search libraries, built here by each box's build_portable.py

so that the Mac's process is Windows': unzip, open Play_vs_AI.command.

Checked before the zip is written: every program and library in the folder has an arm64 part that loads on macOS
build_portable.MIN_MACOS (a package's wheel built for a newer macOS stops the build by name).  The zip is then to be
unzipped somewhere else and the game's self-check and self-test run from it with its own Python
(.github/workflows/mac.yml): what is tested is what is downloaded.
"""
import argparse
import hashlib
import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NAME = "AlphaWOTR"
ZIP = "AlphaWOTR-mac.zip"
RUNTIME = ("https://github.com/astral-sh/python-build-standalone/releases/download/20261110/"
           "cpython-3.12.16%2B20261110-aarch64-apple-darwin-install_only.tar.gz",
           "e956e60bc9465d577ef8062059aa05cd3a2665b4b8678a2ed24e432368330b8f")
BOXES = ("native_backend/live_search_oct19", "turn_search/native_w2x/box")
REPO_ONLY = (".gitignore", ".gitattributes", "RELEASE_MANIFEST.tsv")     # the last is the Windows zip's own list
MACH_O = (b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe", b"\xca\xfe\xba\xbf")  # 64-bit, and the two fat headers


def run(cmd, **kw):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def shipped(rel):
    """A tracked file that goes into the Mac download."""
    if rel.startswith(".github/") or rel in REPO_ONLY:
        return False
    return not ("/" not in rel and rel.endswith(".bat"))


def tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=str(REPO), capture_output=True, check=True).stdout
    return [rel for rel in out.decode("utf-8").split("\0") if rel]


def min_macos():
    spec = importlib.util.spec_from_file_location("build_portable", str(REPO / BOXES[0] / "build_portable.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MIN_MACOS


def version(text):
    return tuple(int(x) for x in text.split("."))


def binaries(stage):
    """-> [(path below the folder, the oldest macOS its arm64 part loads on or None when it names none)], and the
    files that are Mach-O without an arm64 part."""
    found, foreign = [], []
    for path in sorted(stage.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        with open(path, "rb") as f:
            if f.read(4) not in MACH_O:
                continue
        rel = path.relative_to(stage).as_posix()
        archs = subprocess.run(["lipo", "-archs", str(path)], capture_output=True, text=True)
        if archs.returncode:                                   # the same first bytes, not a Mach-O (a Java class)
            continue
        if "arm64" not in archs.stdout.split():
            foreign.append("%s (%s)" % (rel, archs.stdout.strip()))
            continue
        build = subprocess.run(["xcrun", "vtool", "-arch", "arm64", "-show-build", str(path)],
                               capture_output=True, text=True).stdout
        seen = re.findall(r"\bminos\s+([0-9.]+)", build)       # LC_BUILD_VERSION; older files: LC_VERSION_MIN_MACOSX
        seen += re.findall(r"LC_VERSION_MIN_MACOSX\s+cmdsize\s+\d+\s+version\s+([0-9.]+)", build)
        found.append((rel, max(seen, key=version) if seen else None))
    return found, foreign


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if sys.platform != "darwin" or platform.machine().lower() != "arm64":
        raise SystemExit("the Mac download is made on an Apple-silicon Mac (this is %s %s)" % (sys.platform, platform.machine()))
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit("%s exists: name a folder that is not there yet" % out)
    stage = out / NAME
    stage.mkdir(parents=True)

    files = [rel for rel in tracked() if shipped(rel)]
    for rel in files:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(REPO / rel), str(dst), follow_symlinks=False)
    print("%d tracked files -> %s" % (len(files), stage), flush=True)

    url, want = RUNTIME
    archive = out / "runtime.tar.gz"
    urllib.request.urlretrieve(url, str(archive))
    if sha(archive) != want:
        raise SystemExit("the runtime that came from %s is not the one named here (sha256 %s)" % (url, sha(archive)))
    run(["tar", "-xzf", archive, "-C", stage])
    py = stage / "python" / "bin" / "python3"
    run([py, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check", "-r", stage / "requirements.txt"])
    run([py, "-c", "import sys, torch, numpy; print('python', sys.version.split()[0], '| torch', torch.__version__, "
                   "'| numpy', numpy.__version__)"])
    for box in BOXES:
        run([py, "-B", stage / box / "build_portable.py"])
        run([py, "-B", stage / box / "build_portable.py", "--check"])

    floor = min_macos()
    found, foreign = binaries(stage)
    newer = ["%s (macOS %s)" % (rel, v) for rel, v in found if v and version(v) > version(floor)]
    unnamed = [rel for rel, v in found if not v]
    print("%d programs and libraries for arm64; %d name no macOS version; newest need: macOS %s" % (
        len(found), len(unnamed), max((v for _, v in found if v), key=version, default="?")), flush=True)
    if foreign or newer:
        raise SystemExit("not for an Apple-silicon Mac with macOS %s:\n  %s" % (floor, "\n  ".join(foreign + newer)))

    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and not path.is_symlink():
            rows.append("%s\t%s\t%d" % (path.relative_to(stage).as_posix(), sha(path), path.stat().st_size))
    (stage / "RELEASE_MANIFEST.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    run(["zip", "-qry", ZIP, NAME], cwd=str(out))              # -y: links stay links (the runtime has them)
    digest = sha(out / ZIP)
    (out / (ZIP + ".sha256")).write_text("%s  %s\n" % (digest, ZIP), encoding="utf-8")
    print("MADE %s: %d files, %d MB, sha256 %s, for macOS %s and newer" % (
        out / ZIP, len(rows) + 1, (out / ZIP).stat().st_size // 1000000, digest, floor), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
