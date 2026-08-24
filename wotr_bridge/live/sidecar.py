# -*- coding: ascii -*-
"""CLIENT-WAVE sidecar consumer (WAVE-J adoption): the AiFeed agent
writes `<feed>.meta.jsonl` - ONE JSON record per feed line (seq =
1-based feed line number), carrying the interpreter's own raw facts:
nick, tags (the <~..~> tokens in raw order), pids (or the string
'redacted' under blinding), segs (numeric tokens PER inter-tag gap -
die pids and face values stay separate channels, the H7 lesson made
structural), side ('FP'/'SA' only when zone-provable), ts.

WHERE THE SIDECAR ANSWERS, THE HEURISTIC DIES (defect-9): with a
present sidecar the driver's raw import #2 (RAW_ACTION_ROLL_RE
pid-binding) and the draw-line side inference retire; the corpus
spectator has no sidecar and keeps the heuristic lanes - one answerer
PER CHANNEL AVAILABILITY, never two for one quantity.

BLINDING INVARIANT: a record whose feed line was redacted carries no
identities ('redacted' pids) - this reader never reconstructs them
(no bypass around the feed's blinding)."""
import json
import os


class SidecarReader(object):
    def __init__(self, path):
        self.path = path
        self.by_seq = {}
        self.present = False     # a meta file has been seen
        self._pos = 0
        self.n_records = 0
        # the agent's attach generation: gen 1 = the normal first
        # attach (its replay=true paragraphs are history to process);
        # a HIGHER gen = a re-attach to a NEW chat document whose
        # paragraphs were re-emitted onto the same feed.  gen_change =
        # (first seq of the new gen, old gen, new gen) or None.
        self.gen = None
        self.gen_change = None

    def poll(self):
        """Read new complete jsonl lines; lockstep-truncation aware
        (a shrink re-reads from the start; seq keys are idempotent).
        Returns the number of new records."""
        try:
            sz = os.path.getsize(self.path)
        except OSError:
            return 0
        self.present = True
        if sz < self._pos:
            self._pos = 0
        try:
            with open(self.path, 'rb') as f:
                f.seek(self._pos)
                data = f.read()
        except OSError:
            return 0
        nl = data.rfind(b'\n')
        if nl < 0:
            return 0
        chunk = data[:nl + 1]
        self._pos += len(chunk)
        n = 0
        for ln in chunk.decode('utf-8', 'replace').split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            s = r.get('seq')
            if isinstance(s, int):
                self.by_seq[s] = r
                n += 1
                g = r.get('gen')
                # the premain attestation record (attest=true) is written
                # BEFORE the agent attaches to any chat document: it
                # carries no attach generation and sets no baseline
                if isinstance(g, int) and not r.get('attest'):
                    if self.gen is None:
                        self.gen = g
                    elif g != self.gen and self.gen_change is None:
                        self.gen_change = (s, self.gen, g)
                        self.gen = g
        self.n_records += n
        return n

    def rec(self, seq):
        try:
            return self.by_seq.get(int(seq))
        except (TypeError, ValueError):
            return None

    def replay(self, seq):
        """True when the agent marked this feed line as a paragraph
        that pre-existed its attach (replay=true); False otherwise or
        when unknown."""
        r = self.rec(seq)
        return bool(r.get('replay')) if isinstance(r, dict) else False


def meta_side(meta):
    """The record's zone-proven acting side as a game_state.Side, or
    None (the '<game>' channel and ambiguous lines carry none)."""
    s = (meta or {}).get('side')
    if s not in ('FP', 'SA'):
        return None
    import game_state as gs
    return (gs.Side.FREE_PEOPLES if s == 'FP' else gs.Side.SHADOW)


def meta_pids(meta):
    """The record's exact piece ids (list of str), or None when absent
    OR REDACTED (blinding fail-closed: redaction is never a value)."""
    p = (meta or {}).get('pids')
    if isinstance(p, list):
        return [str(x) for x in p]
    return None


def roll_pids(meta):
    """The die-pid channel of an action-roll record (segs[1] per the
    contract - pids and face values are separate channels), or None."""
    segs = (meta or {}).get('segs')
    if isinstance(segs, list) and len(segs) > 1 \
            and isinstance(segs[1], list) and segs[1]:
        return [str(x) for x in segs[1]]
    return None
