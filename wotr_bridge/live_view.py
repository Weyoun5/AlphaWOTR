"""Live ENGINE board mirror for AI-vs-AI games.

The move server (ai_move_server.py --ai --gui) writes the persistent
engine GameState to <client_dir>/ai_live_state.pkl (atomic) plus a
status line to ai_live_status.json after every tick.  This viewer opens
the project's own WotrGUI on that state and repaints whenever the
snapshot changes -- so you can watch the CLONED ENGINE game side by side
with the Java client, with the verification status in the title bar.

Because the snapshot is the REAL engine state (full hidden info), the
viewer also renders an "ENGINE CARDS" side panel proving exactly that:
both players' hands (actual card names, grouped by deck), persistent
cards on the table, and remaining/discarded counts for all 4 decks --
all straight out of the unpickled GameState, live-updated per snapshot.

Usage:  python live_view.py <client_dir>
"""
import json
import os
import pickle
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJECT, 'src'))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# (panel label, GameState deck-list attribute, discard_face_up key)
_DECKS = (
    ('FP Character', 'fp_char_deck', 'fp_character'),
    ('FP Strategy', 'fp_strat_deck', 'fp_strategy'),
    ('SH Character', 'sh_char_deck', 'sh_character'),
    ('SH Strategy', 'sh_strat_deck', 'sh_strategy'),
)


def build_card_report(state, hide_side=None):
    """Plain-text dump of ALL card info in an engine GameState (hidden
    info included): both hands with real card names grouped by deck,
    persistent cards on the table, and remaining/discard counts for all
    4 event decks.  Pure function of the state so tests can assert on
    it; the GUI panel just displays the same string.

    hide_side ('SHADOW'/'FREE_PEOPLES' or a collection of them, SIDED
    mirror): those hands show a masked count instead of names - the
    human must not read the AI's hand off the mirror (same privacy
    rule as the pending jar-side hand mask; the pickle carries full
    info, the DISPLAY owns the privacy)."""
    if isinstance(hide_side, str):
        hidden = {hide_side}
    else:
        hidden = set(hide_side or ())
    import game_state as gs
    try:
        from cards import get_card
    except Exception:                                 # pragma: no cover
        def get_card(_cid):
            return None

    lines = []
    hands = getattr(state, 'hands', None) or {}

    for side, label in ((gs.Side.FREE_PEOPLES, 'FREE PEOPLES'),
                        (gs.Side.SHADOW, 'SHADOW')):
        pinfo = hands.get(side)
        hand = list(getattr(pinfo, 'hand', None) or [])
        lines.append('=== %s HAND (%d) ===' % (label, len(hand)))
        if side.name in hidden:
            lines.append('  (AI hand masked - %d card(s), '
                         'engine-side only)' % len(hand))
            lines.append('')
            continue
        if not hand:
            lines.append('  (empty)')
        by_deck = {'Character': [], 'Strategy': []}
        for cid in hand:
            card = get_card(cid)
            deck = getattr(card, 'deck', None) or '?'
            name = getattr(card, 'card_name', None) or str(cid)
            by_deck.setdefault(deck, []).append('%s  <%s>' % (name, cid))
        for deck, entries in by_deck.items():
            for entry in entries:
                lines.append('  [%s] %s' % (deck[:4], entry))
        lines.append('')

    pc = getattr(state, 'persistent_cards', None)
    fp_table = [c.card_name for c in (getattr(pc, 'fp_cards', None) or [])]
    sh_table = [c.card_name for c in (getattr(pc, 'sh_cards', None) or [])]
    lines.append('=== ON TABLE (persistent effects) ===')
    if not fp_table and not sh_table:
        lines.append('  (none)')
    for name in fp_table:
        lines.append('  FP: %s' % name)
    for name in sh_table:
        lines.append('  SH: %s' % name)
    lines.append('')

    lines.append('=== DECKS (remaining / discarded) ===')
    deck_status = getattr(state, 'decks', None)
    dfu = getattr(state, 'discard_face_up', None) or {}
    for label, deck_attr, key in _DECKS:
        deck_list = getattr(state, deck_attr, None)
        if deck_list is not None:
            remain = len(deck_list)
        elif deck_status is not None:
            remain = getattr(deck_status, key, '?')
        else:
            remain = '?'
        n_disc = len(dfu.get(key, None) or [])
        lines.append('  %-13s %3s / %s' % (label + ':', remain, n_disc))
    fd_fp = len(getattr(hands.get(gs.Side.FREE_PEOPLES),
                        'discarded_face_down', None) or [])
    fd_sh = len(getattr(hands.get(gs.Side.SHADOW),
                        'discarded_face_down', None) or [])
    lines.append('  face-down discards: FP %d / SH %d' % (fd_fp, fd_sh))
    return '\n'.join(lines)


def _extract_events(status, last):
    """New log-pane entries from a bridge-status dict (note +
    waiting_on), given the (note, waiting_on) pair last seen.  Pure so
    tests can drive it headlessly.  Returns (entries, new_last);
    each entry is (is_refusal, text) - refusals are the bridge's
    'illegal:' notes (USER ruling 2026-08-16: illegal human gestures
    are refused with the reason) and render highlighted."""
    note = str((status or {}).get('note') or '')
    waiting = str((status or {}).get('waiting_on') or '')
    l_note, l_wait = last if last else (None, None)
    entries = []
    if note and note != l_note:
        entries.append((note.startswith('illegal:'), note))
    if waiting and waiting != l_wait:
        entries.append((False, 'waiting on: ' + waiting))
    return entries, (note or l_note, waiting or l_wait)


def mask_sides(status, last_status):
    """The hands the mirror must mask.  FAIL CLOSED: names render only
    on a positive 'this side is the human' verdict - a sided status
    (current, else the last one read) names the AI side; with no
    readable status at all BOTH hands are masked."""
    for st in (status, last_status):
        if st and st.get('mode') == 'sided' and st.get('ai_side'):
            return {str(st.get('ai_side'))}
    return {'FREE_PEOPLES', 'SHADOW'}


def _parser_word(evp):
    """'OK' / 'n/a' / 'DIFF' for the engine_vs_parser field (the sided
    writer publishes 'n/a (...)': the parser IS the view)."""
    evp = str(evp or '?')
    if evp == 'OK':
        return 'OK'
    if evp.startswith('n/a'):
        return 'n/a'
    return 'DIFF'


def _make_title(status):
    """Window title from ai_live_status.json content (dict or None)."""
    if not status:
        return 'WOTR ENGINE mirror'
    div = status.get('client_divergences', 0)
    evp = status.get('engine_vs_parser', '?')
    evc = status.get('engine_vs_client', '?')
    title = ('ENGINE mirror | T%s %s %s | %s | '
             'client:%s parser:%s | vs-client:%s' % (
                 status.get('turn'), status.get('phase'),
                 status.get('active') or '',
                 (status.get('last') or '')[:50],
                 div,
                 _parser_word(evp),
                 (evc or '?')[:60]))
    if status.get('over'):
        title += ' | GAME OVER'
    return title


def _status_text(status):
    """Two-line status summary for the card panel header."""
    if not status:
        return 'no status yet'
    line1 = 'T%s %s %s' % (status.get('turn'), status.get('phase'),
                           status.get('active') or '')
    evp = status.get('engine_vs_parser', '?')
    line2 = 'client-div:%s  parser:%s  vs-client:%s' % (
        status.get('client_divergences', 0),
        _parser_word(evp),
        (status.get('engine_vs_client') or '?')[:40])
    if status.get('over'):
        line2 += '  GAME OVER'
    return line1 + '\n' + line2


def _read_bytes(path, retries=1, delay=0.05):
    """Whole-file bytes through a short open; one retry on the
    writer/reader collision (PermissionError during os.replace)."""
    for attempt in range(retries + 1):
        try:
            with open(path, 'rb') as f:
                return f.read()
        except PermissionError:
            if attempt >= retries:
                raise
            time.sleep(delay)


def divergence_marks(status):
    """Region keys to ring red: the status' structured
    'mismatch_regions' list (lower-cased, ' stronghold' suffix
    dropped); the legacy 'piece N: client@X parser@Y' census grammar
    is still parsed when present."""
    marks = []
    st = status or {}
    for tok in st.get('mismatch_regions') or []:
        tok = str(tok).strip().lower()
        if tok.endswith(' stronghold'):
            tok = tok[:-11]
        if tok and tok not in marks:
            marks.append(tok)
    for m in st.get('mismatches') or []:
        mm = re.search(r'client@(.+?) parser@(.+)$', str(m).lower())
        for tok in (list(mm.groups()) if mm else []):
            tok = tok.strip()
            if tok.endswith(' stronghold'):
                tok = tok[:-11]
            if tok and tok not in marks:
                marks.append(tok)
    return marks


class LiveView:
    """WotrGUI on the engine snapshot + a card side panel.

    Construction never blocks: if the snapshot files do not exist yet,
    the window opens on the initial placeholder state and poll() keeps
    waiting.  run() enters mainloop; tests instead call poll_once() and
    root.update() directly."""

    def __init__(self, client_dir):
        self.client_dir = client_dir
        self.pkl = os.path.join(client_dir, 'ai_live_state.pkl')
        self.stj = os.path.join(client_dir, 'ai_live_status.json')
        # bridge status (server -> panel): the log pane tails its
        # note/waiting_on stream; manual_edit is echoed here too
        self.bsj = os.path.join(client_dir, 'ai_bridge_status.json')
        self.last_mtime = 0.0
        self._log_last = (None, None)

        import board as board_mod
        import game_state as gs
        from gui import WotrGUI

        board = board_mod.load_board()
        state = gs.setup_initial_game(board)
        self.gui = WotrGUI(state, board)
        self._build_card_panel()
        self.update_card_panel(state, None)

    # ---- card panel ----------------------------------------------------
    def _build_card_panel(self):
        import tkinter as tk
        g = self.gui
        panel = tk.Frame(g.root, bg='#141420')
        # Re-pack so the panel sits to the RIGHT of the whole WotrGUI
        # layout (main_frame was already packed fill-both-expand).
        try:
            g.main_frame.pack_forget()
            panel.pack(side=tk.RIGHT, fill=tk.Y)
            g.main_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        except Exception:
            panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.card_panel = panel

        # MANUAL EDIT banner (USER ruling 2026-08-16): while gating is
        # suspended the mirror must say so as loudly as the panel does
        self.banner = tk.Label(panel, text='', font=('Arial', 11,
                                                     'bold'),
                               bg='#141420', fg='#141420')
        self.banner.pack(fill=tk.X, padx=6, pady=(4, 0))

        tk.Label(panel, text='ENGINE CARDS (full hidden info)',
                 font=('Arial', 10, 'bold'),
                 bg='#141420', fg='#f0d060').pack(fill=tk.X,
                                                  padx=6, pady=(6, 2))
        self.status_label = tk.Label(
            panel, text='waiting for snapshot...', justify=tk.LEFT,
            anchor='w', bg='#141420', fg='#a0c0ff', font=('Consolas', 9))
        self.status_label.pack(fill=tk.X, padx=6, pady=(0, 4))

        body = tk.Frame(panel, bg='#141420')
        body.pack(fill=tk.BOTH, expand=True, padx=(6, 0), pady=(0, 6))
        self.card_text = tk.Text(
            body, width=52, bg='#101018', fg='#e8e8e8',
            font=('Consolas', 9), wrap='none', state=tk.DISABLED,
            borderwidth=0, highlightthickness=0)
        sb = tk.Scrollbar(body, command=self.card_text.yview)
        self.card_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.card_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # LOG PANE (N3): rolling tail of the bridge's own state-machine
        # words (note + waiting_on stream) with refusals highlighted -
        # what the engine last said, readable without the server log
        tk.Label(panel, text='ENGINE LOG (bridge notes)',
                 font=('Arial', 10, 'bold'),
                 bg='#141420', fg='#f0d060').pack(fill=tk.X,
                                                  padx=6, pady=(2, 2))
        lbody = tk.Frame(panel, bg='#141420')
        lbody.pack(fill=tk.X, padx=(6, 0), pady=(0, 6))
        self.log_text = tk.Text(
            lbody, width=52, height=10, bg='#101018', fg='#c8c8c8',
            font=('Consolas', 8), wrap='word', state=tk.DISABLED,
            borderwidth=0, highlightthickness=0)
        lsb = tk.Scrollbar(lbody, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=lsb.set)
        self.log_text.tag_configure('illegal', foreground='#ff5050',
                                    font=('Consolas', 8, 'bold'))
        lsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def update_card_panel(self, state, status):
        import tkinter as tk
        hide = mask_sides(status, getattr(self, '_last_status', None))
        report = build_card_report(state, hide_side=hide)
        # DIVERGENCES first (USER 2026-08-16: untracked desyncs must
        # be visible): the engine-vs-client census, named per piece
        mism = (status or {}).get('mismatches') or []
        if mism:
            report = ('!!! DIVERGENCES (%d) - client board vs ENGINE '
                      'view !!!\n%s\n\n%s'
                      % (len(mism), '\n'.join('  ' + m for m in mism),
                         report))
        elif status and status.get('mode') == 'sided':
            report = ('engine-vs-client census: CLEAN\n\n' + report)
        t = self.card_text
        t.configure(state=tk.NORMAL)
        t.delete('1.0', tk.END)
        t.insert('1.0', report)
        t.configure(state=tk.DISABLED)
        self.status_label.configure(text=_status_text(status))

    # ---- bridge-status tail (log pane + manual-edit banner) ------------
    def poll_status_notes(self):
        """Tail ai_bridge_status.json into the log pane (every loop
        tick - notes change without a new state snapshot) and drive
        the MANUAL EDIT banner off the bridge's echo."""
        import tkinter as tk
        try:
            with open(self.bsj, encoding='utf-8') as f:
                bst = json.load(f)
        except Exception:
            return
        if bst.get('manual_edit'):
            self.banner.config(text='MANUAL EDIT MODE - gating '
                                    'suspended', bg='#c22',
                               fg='white')
        else:
            self.banner.config(text='', bg='#141420', fg='#141420')
        entries, self._log_last = _extract_events(bst, self._log_last)
        if not entries:
            return
        t = self.log_text
        t.configure(state=tk.NORMAL)
        stamp = time.strftime('%H:%M:%S')
        for is_refusal, text in entries:
            t.insert(tk.END, '[%s] %s\n' % (stamp, text),
                     ('illegal',) if is_refusal else ())
        # bounded buffer: keep the last ~400 lines
        try:
            n = int(float(t.index('end-1c').split('.')[0]))
            if n > 400:
                t.delete('1.0', '%d.0' % (n - 400))
        except Exception:
            pass
        t.see(tk.END)
        t.configure(state=tk.DISABLED)

    # ---- snapshot polling ----------------------------------------------
    def poll_once(self):
        """Check the snapshot once; render it if new.  Returns True when
        a fresh state was rendered, False otherwise (missing pkl counts
        as 'not yet' -- the caller keeps waiting)."""
        try:
            mtime = os.path.getmtime(self.pkl)
        except OSError:
            return False
        if mtime == self.last_mtime:
            return False
        # short open, unpickle from memory: the writer's os.replace
        # must never find this reader holding the file, and a failed
        # load leaves last_mtime alone so the snapshot is retried
        new_state = pickle.loads(_read_bytes(self.pkl))
        status = None
        try:
            with open(self.stj, encoding='utf-8') as f:
                status = json.load(f)
        except Exception:
            pass
        if status is not None:
            self._last_status = status
        self.gui.mirror_marks = divergence_marks(status)
        self.gui.state = new_state
        self.gui.refresh()
        self.update_card_panel(new_state, status)
        self.last_mtime = mtime
        try:
            self.gui.root.title(_make_title(status))
        except Exception:
            pass
        return True

    def _poll_loop(self):
        try:
            self.poll_status_notes()
        except Exception as e:
            print('[live_view] status-note poll error: %r' % (e,))
        try:
            if not self.poll_once():
                # STALENESS must look stale (the frozen-mirror lesson):
                # no fresh snapshot - stamp the age into the title
                st = getattr(self, '_last_status', None)
                ts = float((st or {}).get('ts', 0) or 0)
                if ts:
                    age = time.time() - ts
                    if age > 10:
                        self.gui.root.title(
                            _make_title(st) + ' | STALE %ds' % int(age))
        except Exception as e:
            print('[live_view] poll error: %r' % (e,))
        self.gui.root.after(1000, self._poll_loop)

    def run(self):
        self.gui.root.after(500, self._poll_loop)
        self.gui.run()


def main():
    client_dir = sys.argv[1] if len(sys.argv) > 1 else '.'
    view = LiveView(client_dir)
    print('[live_view] waiting for engine snapshots in %s' % view.pkl)
    view.run()


if __name__ == '__main__':
    main()
