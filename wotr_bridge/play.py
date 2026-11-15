"""One session of the game on a PC without cmd (macOS, Linux): what Play_vs_AI.bat does on Windows, step for step.

    python wotr_bridge/play.py [SA|FP|AUTO] [net ...]     play (Play_vs_AI.command starts this)
    python wotr_bridge/play.py --self-check               start the AI server as a session does, without the client,
                                                          and say whether it came up (Setup_Mac.command ends with this)

Every name, file list and command line is launch_head's (the one head): this module only walks the steps.
  1  this Python has the packages; Java is there and new enough; the client jar is there
  2  the two native libraries are built for this PC (their build_portable.py), built now when they are not
  3  the AI's side and nets: from the arguments, else asked (a side argument with no net takes the WIN nets)
  4  a client already running from this folder is refused; a server a session left behind is stopped and waited for
  5  the last session's files are swept: game logs archived, coordination files removed, diagnostics kept as .prev
  6  the AI server is started (its console in ai_server_console.log) and waited for until its heartbeat file is there
  7  the client runs in the foreground; when its window closes the server is stopped
On macOS the client gets Swing's own look (MAC_JAVA_OPTS): the system menu bar shows menu items only, and the AI
menu's strength slider is a component, so the menu bar must stay in the game's window.
The file that proves a live server (ai_server.alive) is never removed here: on these systems a file a process holds
can be unlinked, and a server that survived would keep its lock on a file nobody sees.
"""
import glob
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import launch_head as lh        # noqa: E402

PLAY = lh.PLAY_CLIENT
PROBE_IMPORTS = 'import torch, numpy, shapely, PIL, matplotlib, psutil'
MAC_JAVA_OPTS = ('-Dswing.defaultlaf=javax.swing.plaf.metal.MetalLookAndFeel', '-Xdock:name=AlphaWOTR')
NATIVE_BOXES = (os.path.join(lh.ROOT, 'native_backend', 'live_search_oct19'),
                os.path.join(lh.ROOT, 'turn_search', 'native_w2x', 'box'))
ALIVE_NAME = 'ai_server.alive'      # (core.transport.ALIVE_NAME; not imported: this module starts without the bridge)
STOP_WAIT_S = 20


def say(text=''):
    print(text, flush=True)


def packages():
    """-> None, or what this Python lacks."""
    r = subprocess.run([sys.executable, '-c', PROBE_IMPORTS], capture_output=True, text=True)
    return None if r.returncode == 0 else (r.stderr.strip().splitlines() or ['?'])[-1]


def java():
    """-> (the java to run, its major version), or (None, why)."""
    home = os.environ.get('JAVA_HOME')
    exe = os.path.join(home, 'bin', 'java') if home and os.path.isfile(os.path.join(home, 'bin', 'java')) else shutil.which('java')
    if not exe:
        return None, 'no java on the PATH'
    r = subprocess.run([exe, '-version'], capture_output=True, text=True)
    text = r.stderr + r.stdout
    if r.returncode:
        return None, (text.strip().splitlines() or ['java -version failed'])[0]
    import re
    m = re.search(r'version "(\d+)(?:\.(\d+))?', text)
    if not m:
        return None, 'java -version says: %s' % text.strip()[:120]
    major = int(m.group(2) or 0) if m.group(1) == '1' else int(m.group(1))
    return exe, major


def native_built():
    """Step 2.  -> None, or what went wrong.  (Windows loads the boxes' own libraries: nothing to build.)"""
    if os.name == 'nt':
        return None
    for box in NATIVE_BOXES:
        builder = os.path.join(box, 'build_portable.py')
        if subprocess.run([sys.executable, '-B', builder, '--check'], capture_output=True).returncode == 0:
            continue
        say('   Building the search library in %s for this PC (once; a few minutes) ...' % os.path.relpath(box, lh.ROOT))
        r = subprocess.run([sys.executable, '-B', builder], capture_output=True, text=True)
        if r.returncode:
            return 'the build failed:\n%s\n      On a Mac the compiler comes with:  xcode-select --install' % (
                (r.stdout + r.stderr).strip()[-1500:])
    return None


def ask(question, options):
    """options: (key, text) pairs -> the key he chose (asked again until it is one)."""
    say('   ' + question)
    for n, (key, text) in enumerate(options, 1):
        say('     [%d]  %s' % (n, text))
    while True:
        try:
            got = input('   Type %s: ' % ' or '.join(str(n) for n in range(1, len(options) + 1))).strip().upper()
        except EOFError:
            raise SystemExit('   nothing was chosen')
        for n, (key, _text) in enumerate(options, 1):
            if got in (str(n), key.upper()):
                return key


def choices(args):
    """Step 3.  -> (side, net tag, the nets' names).  Raises SystemExit on an argument that is no side or net."""
    asked = not args
    try:
        side = lh._side(args[0] if args else ask('Which side does the AI play?', (
            ('SA', 'the Shadow (you play the Free Peoples)'), ('FP', 'the Free Peoples (you play the Shadow)'),
            (lh.WATCH_SIDE, 'both - watch AI vs AI'))))
    except ValueError as error:
        raise SystemExit('   [X] %s' % error)
    fp_arg = args[1] if len(args) > 1 and side in ('FP', lh.WATCH_SIDE) else None
    sh_arg = args[1] if len(args) > 1 and side == 'SA' else args[2] if len(args) > 2 and side == lh.WATCH_SIDE else None

    def net(arg, title, nets, plays):
        if arg is not None:
            hit = [key for key, _name, _what in nets if key.upper() == arg.upper()]
            if not hit:
                raise SystemExit('   [X] Unknown %s net %r: %s' % (title, arg, ' or '.join(k.upper() for k, _, _ in nets)))
            return hit[0]
        if asked and plays:
            return ask('Which %s AI?' % title, [(key, '%-15s %s' % (name, what)) for key, name, what in nets])
        return nets[0][0]
    fp = net(fp_arg, 'Free Peoples', lh.FP_NETS, side in ('FP', lh.WATCH_SIDE))
    sh = net(sh_arg, 'Shadow', lh.SH_NETS, side in ('SA', lh.WATCH_SIDE))
    names = {'FP': dict((k, n) for k, n, _ in lh.FP_NETS)[fp], 'SA': dict((k, n) for k, n, _ in lh.SH_NETS)[sh]}
    names[lh.WATCH_SIDE] = '%s vs %s' % (names['FP'], names['SA'])
    return side, lh.NET_TAG_FORMAT % (fp, sh), names[side]


def client_running():
    """A client that already runs from this folder (its jar on its command line, this folder its own)."""
    import psutil
    for p in psutil.process_iter(['name', 'cmdline', 'cwd']):
        try:
            line = p.info.get('cmdline') or []
            if 'java' in (p.info.get('name') or '').lower() and any(lh.JAR_NAME in x for x in line) \
                    and os.path.realpath(p.info.get('cwd') or '') == os.path.realpath(PLAY):
                return p.pid
        except (psutil.Error, OSError):
            continue
    return None


def stop_server():
    """Step 4 and 7: the server a session of this folder left running is stopped, and gone before this returns."""
    lh.stop_ours(PLAY)
    deadline = time.time() + STOP_WAIT_S
    while lh.server_pid(PLAY) and time.time() < deadline:
        time.sleep(0.5)
    return lh.server_pid(PLAY) is None


def sweep():
    """Step 5.  -> None, or the coordination file that could not be removed."""
    auto, archive = os.path.join(PLAY, 'logs', 'auto'), os.path.join(PLAY, 'logs', 'archive')
    for folder in (auto, archive):
        os.makedirs(folder, exist_ok=True)
    for log in glob.glob(os.path.join(auto, '*.log')):
        os.replace(log, os.path.join(archive, os.path.basename(log)))
    for name in lh.LEFTOVER_FILES:
        path = os.path.join(PLAY, name)
        if name == ALIVE_NAME or not os.path.exists(path):
            continue
        try:
            os.remove(path)
        except OSError:
            return name
    for name in lh.ROTATE_FILES:
        path = os.path.join(PLAY, name)
        if os.path.exists(path):
            os.replace(path, path + '.prev')
    return None


def start_server(side, tag):
    """Step 6.  -> None when the heartbeat is there, else what went wrong (with the end of the server's console)."""
    console = os.path.join(PLAY, lh.SERVER_CONSOLE_LOG)
    with open(console, 'wb') as out:
        more = dict(start_new_session=True) if os.name != 'nt' else dict(creationflags=subprocess.CREATE_NO_WINDOW)
        subprocess.Popen(lh.server_cmd(PLAY, side, py=sys.executable, net_tag=tag), cwd=lh.BRIDGE_DIR, env=lh.head_env(),
                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **more)
    beat = os.path.join(PLAY, lh.HEARTBEAT_FILE)
    started = time.time()
    while not os.path.exists(beat):
        waited = time.time() - started
        if waited > lh.HEARTBEAT_WAIT_S or (waited > 30 and not lh.server_pid(PLAY)):
            with open(console, 'rb') as f:
                tail = f.read()[-1500:].decode('utf-8', 'replace')
            return ('the AI server ended before it was ready' if waited <= lh.HEARTBEAT_WAIT_S else
                    'no server heartbeat within %d seconds' % lh.HEARTBEAT_WAIT_S) + ':\n' + tail
        time.sleep(1)
    return None


def session(side, tag, client=True):
    """Steps 4 to 7.  -> the exit code."""
    if client_running():
        say('   [X] The game client is already running from this folder.  Close it first.')
        return 1
    if not stop_server():
        say('   [X] An AI server of an earlier session is still running and did not stop.')
        return 1
    stuck = sweep()
    if stuck:
        say('   [X] Could not remove the leftover file %s (is something still using it?)' % stuck)
        return 1
    say('   Starting the AI server (it loads its nets; up to a minute) ...')
    try:
        problem = start_server(side, tag)
        if problem:
            say('   [X] %s' % problem)
            return 1
        if not client:
            say('   The AI server is up (%s).' % os.path.join(PLAY, lh.SERVER_CONSOLE_LOG))
            return 0
        exe, _major = java()
        cmd = lh.java_cmd(PLAY, side, pythonw=sys.executable, net_tag=tag,
                          extra_java_opts=MAC_JAVA_OPTS if sys.platform == 'darwin' else ())
        say('   In the client: New Game -> menu AI -> "Play vs AI..." -> Start Game.')
        return subprocess.call([exe] + cmd[1:], cwd=PLAY)
    finally:
        stop_server()


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    check = args == ['--self-check']
    say()
    say(' AlphaWOTR - War of the Ring against the AI')
    say()
    lacks = packages()
    if lacks:
        say('   [X] This Python (%s) lacks a package the game needs:\n       %s' % (sys.executable, lacks))
        say('       Run Setup_Mac.command (or:  python -m pip install -r requirements.txt).')
        return 1
    if not check:
        exe, major = java()
        if exe is None or major < lh.JAVA_MIN:
            say('   [X] The game client needs Java %d or newer (17 or 21 recommended: https://adoptium.net): %s' % (
                lh.JAVA_MIN, major if exe is None else 'found Java %d' % major))
            return 1
        if not os.path.isfile(os.path.join(PLAY, lh.JAR_NAME)):
            say('   [X] The client jar is missing: %s' % os.path.join(PLAY, lh.JAR_NAME))
            return 1
    problem = native_built()
    if problem:
        say('   [X] ' + problem)
        return 1
    if check:
        return session('SA', lh.NET_TAG, client=False)
    side, tag, names = choices(args)
    say('   The AI plays %s (net tag %s).' % (names, tag))
    say()
    return session(side, tag)


if __name__ == '__main__':
    sys.exit(main())
