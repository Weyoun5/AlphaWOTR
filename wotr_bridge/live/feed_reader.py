"""Tail the AiFeed rendered-feed file.

MEASURED CONTRACT (AiFeed.java, verified 2026-08-19 / ARCH_AUDIT A14):
  * one rendered chat-pane paragraph per line, UTF-8, flushed per drain;
  * the file is TRUNCATED at agent premain (client restart);
  * a swapped chat document re-emits from its start (watch thread
    re-attach resets the agent's emitted counter) — the consumer must
    recognize an identical prefix and resume past it (client fact 21);
  * live prefixes: '$Nick ...' / '<auto> ...' / '<game> ...';
    replay prefixes: '%Nick ...' / '%auto> ...' (A15).

Design rule 12: this module never drops content silently.  Anything
that is not a clean continuation is surfaced as an explicit event.
"""
import io
import os
import re

# the client's own new-game announcement (the first paragraph of every
# game's chat document); a second one IN-STREAM is a new document
# re-emitted onto the same feed (agent re-attach: the file only grows)
_NEW_GAME_RE = re.compile(
    r'^<game>\s+\S+\s+has started a new War of the Ring game')
# bytes of the consumed head re-verified on every poll: a premain
# truncation followed by a LONGER re-emission never shrinks the file,
# so the head is the only witness that the document changed
_HEAD_CHECK = 4096


class FeedRestart(Exception):
    """The feed was truncated and its new content is NOT a prefix-match
    of what was already consumed: a different document (client restart
    or new game).  The driver decides what that means; the reader never
    guesses."""


class FeedReader(object):
    """Incremental line reader over the AiFeed output file.

    poll() returns the list of NEW complete lines (str, no terminator)
    since the last call.  Truncation handling: on shrink, re-read from
    the start; if the re-emitted stream prefix-matches every line
    already consumed, silently resume past the prefix (document
    re-attach); otherwise raise FeedRestart.

    The full consumed history is kept in memory (a full game is a few
    thousand lines) — it is both the prefix-match reference and the
    driver's replay source after a crash."""

    def __init__(self, path):
        self.path = path
        self._offset = 0          # byte offset of consumed content
        self._buf = b''           # unterminated tail bytes
        self.lines = []           # every line ever consumed, in order
        self._head = b''          # consumed head bytes (<= _HEAD_CHECK)
        self._seen_new_game = False

    def poll(self):
        """Return new complete lines (possibly an empty list).
        Invariants (rule 12): a shrunken file re-syncs or raises; a
        file whose consumed HEAD changed (truncate-then-longer
        re-emission) raises; a second new-game announcement in-stream
        (append re-attach) raises - never a silent continuation."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return []             # not created yet
        if size < self._offset:
            self._resync_after_truncation()
        with io.open(self.path, 'rb') as f:
            if self._head:
                head = f.read(len(self._head))
                if head != self._head:
                    raise FeedRestart(
                        'feed head changed under %d consumed byte(s) - '
                        'the document was re-emitted from the start '
                        '(client restart or new game)'
                        % self._offset)
            f.seek(self._offset)
            chunk = f.read()
        self._offset += len(chunk)
        out = self._split(chunk)
        self._note_head(chunk)
        for ln in out:
            if _NEW_GAME_RE.match(ln):
                if self._seen_new_game:
                    raise FeedRestart(
                        'a second new-game announcement arrived in-'
                        'stream at line %d: a new chat document was '
                        're-emitted onto this feed (re-attach)'
                        % len(self.lines))
                self._seen_new_game = True
        return out

    def _note_head(self, chunk):
        room = _HEAD_CHECK - len(self._head)
        if room > 0 and chunk:
            self._head += chunk[:room]

    @staticmethod
    def _decode(raw):
        return raw.rstrip(b'\r').decode('utf-8', 'replace')

    def _split(self, chunk):
        data = self._buf + chunk
        out = []
        while True:
            i = data.find(b'\n')
            if i < 0:
                break
            out.append(self._decode(data[:i]))
            data = data[i + 1:]
        self.lines.extend(out)
        self._buf = data
        return out

    def _resync_after_truncation(self):
        """The file shrank.  Re-read from 0; a consumed-history prefix
        match means a document re-attach re-emitted history — resume
        past it.  Anything else raises FeedRestart (rule 12)."""
        with io.open(self.path, 'rb') as f:
            whole = f.read()
        segments = whole.split(b'\n')[:-1]     # terminated lines only
        n = len(self.lines)
        replayed = [self._decode(s) for s in segments[:n]]
        first_bad = next((i for i in range(len(replayed))
                          if replayed[i] != self.lines[i]), None)
        if first_bad is not None:
            raise FeedRestart(
                'feed truncated; re-emitted content diverges at line %d '
                'of %d consumed' % (first_bad, n))
        if len(replayed) < n:
            # shorter than history and still a prefix: the new document
            # has LOST consumed lines — that is a restart, not a resume
            raise FeedRestart(
                'feed truncated to %d lines, %d were consumed'
                % (len(replayed), n))
        # exact prefix: resume at the byte just past segment n-1
        self._offset = sum(len(s) + 1 for s in segments[:n])
        self._buf = b''
        self._head = whole[:min(self._offset, _HEAD_CHECK)]
