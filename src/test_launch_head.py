"""ONE-HEAD PINS: Play_vs_AI.bat and wotr_bridge\\launch_head.py cannot
drift, and the harness launches what the bat launches.

  * the bat on disk IS launch_head.render_bat() (generated; the module
    is the single source of the launch facts);
  * the bat's java / server lines carry exactly the tokens
    launch_head.java_cmd / server_cmd produce (same spec, cmd-quoted);
  * every HEAD_ENV pin is set by the bat, for all three modes (play,
    gui, console), and the gui/console modes run the engine entry
    points with the live net tag;
  * the play path ends in the java line + `goto cleanup` with the
    gui/console sections between (cmd resumes a running batch at a
    byte offset: the layout invariant in launch_head);
  * boot_e2e_harness.launch / start_server build their command lines
    from launch_head (no second implementation).

Each test takes a `fails` list; main() runs them all (exit 1 on any).
"""
import os
import re
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SRC)
_BRIDGE = os.path.join(_ROOT, 'wotr_bridge')
_LAB_STREAK = os.path.join(os.path.dirname(_ROOT), 'WOTR_AI_training_lab',
                           'streak')
for _p in (_BRIDGE, _LAB_STREAK):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import launch_head as LH  # noqa: E402


def _bat_text():
    with open(LH.BAT_PATH, 'rb') as f:
        return f.read().decode('ascii').replace('\r\n', '\n')


def _unquote(tok):
    return tok[1:-1] if tok.startswith('"') and tok.endswith('"') else tok


def _split_flag(tok):
    tok = _unquote(tok)
    if '="' in tok and tok.endswith('"'):
        k, v = tok.split('="', 1)
        return k + '=' + v[:-1]
    return tok


def test_bat_is_generated(fails):
    """MEASURES: the file on disk == render_bat() (launch_head --check)."""
    drift = LH.check_bat()
    if drift:
        fails.append(('bat_drift', drift[0][:300]))
    if os.path.exists(LH.BAT_PATH) and any(
            ord(c) > 127 for c in open(LH.BAT_PATH, 'rb').read().decode(
                'latin1')):
        fails.append(('bat_non_ascii', ''))


def test_java_line_is_java_cmd(fails):
    """MEASURES: the bat's java line, with its variables substituted,
    is token-for-token launch_head.java_cmd(...)."""
    txt = _bat_text()
    lines = [l for l in txt.split('\n') if l.startswith('java ')]
    if len(lines) != 1:
        fails.append(('java_line_count', len(lines)))
        return
    subs = {'!PLAY!': r'C:\cl', '%AISIDE%': 'FP', '!PYW!': r'C:\pyw.exe',
            '!BRIDGE!': r'C:\br'}
    line = lines[0]
    for k, v in subs.items():
        line = line.replace(k, v)
    toks = [_split_flag(t) for t in re.findall(r'"[^"]*"|\S+', line)]
    want = LH.java_cmd(r'C:\cl', 'FP', pythonw=r'C:\pyw.exe',
                       bridge_dir=r'C:\br')
    if toks != want:
        fails.append(('java_tokens', (toks, want)))
    if '-Dai.feed.aiside=FP' not in want or '-javaagent:AiFeed.jar=C:\\cl\\ai_feed.txt' not in want:
        fails.append(('java_blinding_flags', want))


def test_server_line_is_server_cmd(fails):
    """MEASURES: the bat's server start carries launch_head.server_cmd's
    tokens (python -m live.live_server --client-dir .. --side .. --sims
    300 --net-tag corpus_live) with cwd = wotr_bridge."""
    txt = _bat_text()
    lines = [l for l in txt.split('\n') if 'live.live_server' in l]
    if len(lines) != 1:
        fails.append(('server_line_count', len(lines)))
        return
    line = lines[0]
    for k, v in {'!PY!': r'C:\py.exe', '!PLAY!': r'C:\cl',
                 '%AISIDE%': 'SA'}.items():
        line = line.replace(k, v)
    want = LH.server_cmd(r'C:\cl', 'SA', py=r'C:\py.exe')
    inner = ' '.join(('"%s"' % t) if ('\\' in t) else t for t in want)
    if inner not in line:
        fails.append(('server_tokens', (inner, line)))
    i_srv = txt.index(lines[0])
    i_cd = txt.rfind('cd /d "!BRIDGE!"', 0, i_srv)
    if i_cd < 0:
        fails.append(('server_cwd_not_bridge', ''))
    # heartbeat wait before java
    i_java = txt.index('\njava ')
    if not (i_srv < txt.index(LH.HEARTBEAT_FILE, i_srv) < i_java):
        fails.append(('heartbeat_wait_order', ''))


def test_env_pins_every_mode(fails):
    """MEASURES: each HEAD_ENV pin is a `set` line placed BEFORE the
    play dispatch in Play_vs_AI.bat AND before the side dispatch in
    Play_Engine_Only.bat (both bats inherit every pin)."""
    txt = _bat_text()
    i_dispatch = txt.index('if /I "%~1"=="help" goto usage')
    etxt = LH.render_engine_bat()
    i_edisp = etxt.index('set "SIDE=%~1"')
    for k, v in LH.HEAD_ENV:
        line = 'set "%s=%s"' % (k, v)
        j = txt.find(line)
        if j < 0 or j > i_dispatch:
            fails.append(('env_pin_' + k, j))
        je = etxt.find(line)
        if je < 0 or je > i_edisp:
            fails.append(('engine_env_pin_' + k, je))
    for key in ('CUDA_VISIBLE_DEVICES', 'WOTR_ACT_GEN'):
        if key not in dict(LH.HEAD_ENV):
            fails.append(('env_missing_' + key, ''))
    if dict(LH.HEAD_ENV).get('WOTR_ACT_GEN') != 'v52':
        fails.append(('act_gen_not_v52', dict(LH.HEAD_ENV)))


def test_modes(fails):
    """MEASURES (USER 2026-08-23: two bats, console mode removed):
    Play_vs_AI.bat is play ONLY (no gui/console dispatch, no
    engine_play entry) and ends in the java line followed by `goto
    cleanup`; Play_Engine_Only.bat on disk == render_engine_bat(),
    runs src\\gui.py --live <side> --net corpus_live, carries the
    UNTESTED banner and the same Python probe (bundled runtime first)."""
    txt = _bat_text()
    for bad in ('goto gui', 'goto console', 'engine_play.py', '\n:gui\n',
                '\n:console\n'):
        if bad in txt:
            fails.append(('play_bat_has_' + bad.strip(':\n'), ''))
    drift = LH.check_engine_bat()
    if drift:
        fails.append(('engine_bat_drift', drift[0][:300]))
    etxt = LH.render_engine_bat()
    if '"!HERE!src\\gui.py" --live %SIDE% --net ' + LH.NET_TAG not in etxt:
        fails.append(('engine_entry', ''))
    if 'UNTESTED' not in etxt:
        fails.append(('engine_untested_banner', ''))
    if 'python\\python.exe' not in etxt or 'ALPHAWOTR_PYTHON' not in etxt:
        fails.append(('engine_probe', ''))
    if any(ord(c) > 127 for c in etxt):
        fails.append(('engine_bat_non_ascii', ''))
    i_java = txt.index('\njava ')
    i_goto = txt.index('\ngoto cleanup\n')
    i_clean = txt.index('\n:cleanup\n')
    if not (i_java < i_goto < i_clean):
        fails.append(('layout_order', (i_java, i_goto, i_clean)))
    between = txt[i_java:i_goto].split('\n')
    # java line, then exactly one comment line (PAD), then the jump
    if len(between) != 3 or not between[2].startswith('::'):
        fails.append(('java_then_pad_then_goto', between[:3]))
    if LH.goto_cleanup_offset(txt) != i_goto + 1:
        fails.append(('goto_offset_fn', (LH.goto_cleanup_offset(txt),
                                         i_goto + 1)))


def test_sweeps_scoped_to_play_dir(fails):
    """MEASURES: every process sweep in the bat is scoped to $env:PLAY
    (a harness scratch client is never in range)."""
    txt = _bat_text()
    sweeps = [l for l in txt.split('\n') if 'Win32_Process' in l]
    if len(sweeps) < 3:
        fails.append(('sweep_count', len(sweeps)))
    for l in sweeps:
        if "$env:PLAY" not in l:
            fails.append(('sweep_unscoped', l[:120]))
        if "'*play_client*'" in l:
            fails.append(('sweep_loose_match', l[:120]))


def test_harness_uses_launch_head(fails):
    """MEASURES: boot_e2e_harness builds its java / server command
    lines from launch_head (one implementation)."""
    try:
        import boot_e2e_harness as H
    except Exception as e:
        fails.append(('harness_import', repr(e)))
        return
    if getattr(H, 'LH', None) is not LH:
        fails.append(('harness_not_on_launch_head', ''))
    src = open(H.__file__, 'r', encoding='ascii', errors='replace').read()
    if 'LH.java_cmd(' not in src or 'LH.server_cmd(' not in src:
        fails.append(('harness_own_cmdline', ''))
    if "'-jar'" in src or '-Dai.pace' in src.split('def main')[0]:
        fails.append(('harness_second_flag_spec', ''))
    for name in ('build_root', 'launch', 'teardown', 'start_server',
                 'launch_bat'):
        if not callable(getattr(H, name, None)):
            fails.append(('harness_missing_' + name, ''))


def main():
    tests = [test_bat_is_generated, test_java_line_is_java_cmd,
             test_server_line_is_server_cmd, test_env_pins_every_mode,
             test_modes, test_sweeps_scoped_to_play_dir,
             test_harness_uses_launch_head]
    fails = []
    for t in tests:
        n0 = len(fails)
        try:
            t(fails)
        except Exception:
            import traceback
            fails.append((t.__name__ + ':EXC',
                          traceback.format_exc()[-600:]))
        print('%-40s %s' % (t.__name__,
                            'ok' if len(fails) == n0 else 'FAIL'))
    for name, detail in fails:
        print('  FAIL %s: %s' % (name, str(detail)[:400]))
    print('%d test(s), %d failure(s)' % (len(tests), len(fails)))
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
