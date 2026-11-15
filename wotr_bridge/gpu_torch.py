"""The GPU build of torch for the game's Python - optional.

    python wotr_bridge\\gpu_torch.py status
    python wotr_bridge\\gpu_torch.py enable   [--wheels DIR] [--force]
    python wotr_bridge\\gpu_torch.py disable  [--wheels DIR]

(Enable_GPU.bat runs "enable" with the game's own python\\.)

The game ships with the CPU build of torch: small, and it runs on every PC.  On a PC with an NVIDIA GPU the turn
search (the AI menu's strength slider, positions 2-4) is about three times as fast with the CUDA build.
  enable    replaces the CPU build by the CUDA build OF THE SAME VERSION (about 2 GB to download, 5 GB on disk)
  disable   puts the CPU build back
Only torch is touched (pip --no-deps); the game needs no other change: its turn search decides at every start whether
the GPU can be used and falls back to the processor by itself (turn_search\\turn_search_service.log says which).
The new build is tried before this ends; when it does not import, the build that was there is installed again.
--wheels DIR: take the wheel from that folder instead of downloading it.
--force: enable although no NVIDIA driver was found.
"""
import argparse
import importlib.metadata
import os
import shutil
import subprocess
import sys

CUDA = 'cu130'                                  # the CUDA build this game is tested with
INDEX = 'https://download.pytorch.org/whl/'     # + 'cpu' or CUDA
NEEDS_GB = {'cpu': 2, CUDA: 6}                  # free disk space the download and the unpacked build need
LONGEST = 150                                   # the longest path inside a torch wheel, from site-packages
TRY = ("import torch; x = torch.arange(64, dtype=torch.float32); assert float((x @ x).item()) == 85344.0; "
       "print('TORCH', torch.__version__, torch.cuda.is_available(), "
       "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')")


def installed():
    """-> (version without the build, the build: 'cpu', 'cu130', ... or '' for a build with no name), or (None, None)."""
    try:
        version = importlib.metadata.version('torch')
    except importlib.metadata.PackageNotFoundError:
        return None, None
    base, _, build = version.partition('+')
    return base, build


def tried():
    """The torch this Python has now, tried in a process of its own: (ok, the line it printed or the error's end)."""
    env = {k: v for k, v in os.environ.items() if k != 'CUDA_VISIBLE_DEVICES'}
    r = subprocess.run([sys.executable, '-c', TRY], capture_output=True, text=True, env=env)
    line = [x for x in r.stdout.splitlines() if x.startswith('TORCH ')]
    return (r.returncode == 0 and bool(line)), (line[-1] if line else (r.stderr.strip() or r.stdout.strip())[-600:])


def others_running():
    """Other processes of this Python (a game that is open holds torch's files: pip cannot replace them)."""
    try:
        import psutil
    except ImportError:
        return []
    me, mine = os.getpid(), os.path.normcase(os.path.realpath(sys.executable))
    folder = os.path.dirname(mine)
    out = []
    for p in psutil.process_iter(['pid', 'exe']):
        exe = p.info.get('exe')
        if exe and p.info['pid'] != me and os.path.normcase(os.path.dirname(os.path.realpath(exe))) == folder:
            out.append('%s (pid %d)' % (os.path.basename(exe), p.info['pid']))
    return out


def pip_install(base, build, wheels):
    spec = 'torch==%s+%s' % (base, build)
    source = ['--no-index', '--find-links', wheels] if wheels else ['--index-url', INDEX + build]
    cmd = [sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-cache-dir', '--disable-pip-version-check',
           '--no-warn-script-location', spec] + source
    print('> ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd).returncode == 0


def switch(to, wheels, force):
    base, build = installed()
    if base is None:
        return 'this Python (%s) has no torch: nothing to switch' % sys.executable
    if build == to:
        ok, line = tried()
        return None if ok else 'torch %s+%s is installed but does not work: %s' % (base, build, line)
    if to != 'cpu' and not force and shutil.which('nvidia-smi') is None:
        return ('no NVIDIA driver found on this PC (no nvidia-smi): the GPU build would only take space, the game '
                'would still run on the processor.  Nothing changed.  (--force installs it anyway.)')
    busy = others_running()
    if busy:
        return 'close the game first - this Python is running: %s.  Nothing changed.' % ', '.join(busy)
    site = os.path.join(os.path.dirname(str(importlib.metadata.distribution('torch').locate_file('torch'))), '')
    if len(site) + LONGEST > 259:
        return ('the game\'s folder path is too long for this build (%d characters to site-packages; Windows allows 260 '
                'for a whole path): move the game to a shorter path first.  Nothing changed.' % len(site))
    free = shutil.disk_usage(site).free / 1e9
    if free < NEEDS_GB[to]:
        return 'this needs about %d GB of free disk space; the drive has %.1f GB.  Nothing changed.' % (NEEDS_GB[to], free)
    print('torch %s+%s -> torch %s+%s%s' % (base, build, base, to, '' if wheels else '   (downloading; this takes a while)'),
          flush=True)
    if pip_install(base, to, wheels):
        ok, line = tried()
        if ok:
            return None
        print('the new build does not work here: %s' % line, flush=True)
    else:
        print('pip could not install it (see above).', flush=True)
    if installed() == (base, build) and tried()[0]:
        return 'Nothing changed: torch %s+%s is still in place.' % (base, build)
    print('putting torch %s+%s back ...' % (base, build), flush=True)
    back = pip_install(base, build, wheels) and tried()[0]
    return ('torch %s+%s is back in place.' % (base, build) if back else
            'COULD NOT PUT torch %s+%s BACK.  Unzip the release again, or run:  "%s" -m pip install --no-deps '
            '--force-reinstall torch==%s+%s --index-url %s%s' % (base, build, sys.executable, base, build, INDEX, build))


def status():
    base, build = installed()
    if base is None:
        print('this Python (%s) has no torch' % sys.executable)
        return 1
    ok, line = tried()
    if not ok:
        print('torch %s+%s is installed but does not work: %s' % (base, build, line))
        return 1
    _, version, cuda, name = line.split(' ', 3)
    print('torch %s (%s build).  %s' % (version, 'CPU' if build == 'cpu' else 'GPU' if build.startswith('cu') else 'unnamed',
                                        'It sees the GPU %s: the turn search can use it.' % name if cuda == 'True' else
                                        'It sees no GPU it can use: the turn search runs on the processor.'))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('action', choices=('status', 'enable', 'disable'))
    ap.add_argument('--wheels', default='', help='a folder that holds the wheel (no download)')
    ap.add_argument('--force', action='store_true', help='enable although no NVIDIA driver was found')
    a = ap.parse_args(argv)
    if a.action == 'status':
        return status()
    problem = switch(CUDA if a.action == 'enable' else 'cpu', a.wheels, a.force)
    if problem:
        print(problem)
        return 1
    return status()


if __name__ == '__main__':
    sys.exit(main())
