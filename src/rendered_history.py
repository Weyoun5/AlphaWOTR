"""Prefix-only persistent history and explicit corpus decision chronology."""
from bisect import bisect_left


class HistoryUnproven(ValueError):
    pass


class DeckHistory:
    """Remaining card multisets before a rendered line, never a future hand.

    Catalogue order is only a deterministic storage order, not a claim about
    the hidden shuffled deck order. Undecoded/contradictory changes make only
    the affected prefix unproven; an exact undo restores the previous state.
    """
    def __init__(self, catalogues):
        self.catalogues = {code: tuple(cards) for code, cards in catalogues.items()}
        self.remaining = {code: list(cards) for code, cards in self.catalogues.items()}
        self.unknown = {code: 0 for code in self.catalogues}
        self.lines, self.snapshots = [], []

    def _record(self, line):
        if self.lines and line < self.lines[-1]:
            raise ValueError('Deck history must be recorded chronologically')
        self.lines.append(line)
        self.snapshots.append(({code: tuple(cards) for code, cards in self.remaining.items()}, dict(self.unknown)))

    def draw(self, code, card, line):
        known = card is not None and card in self.remaining[code]
        if known:
            self.remaining[code].remove(card)
        else:
            self.unknown[code] += 1
        self._record(line)
        return code, card, known

    def undo_draw(self, token, line):
        code, card, known = token
        if known:
            self.remaining[code].append(card)
        else:
            self.unknown[code] -= 1
        self._record(line)

    def return_card(self, code, card, line):
        if card not in self.catalogues[code] or card in self.remaining[code]:
            self.unknown[code] += 1
        else:
            self.remaining[code].append(card)
        self._record(line)

    def before(self, line):
        if not isinstance(line, (int, float)) or line <= 0:
            raise HistoryUnproven('deck_prefix_boundary_unproven')
        index = bisect_left(self.lines, line) - 1
        cards, unknown = (self.snapshots[index] if index >= 0 else (self.catalogues, {}))
        if any(unknown.values()):
            raise HistoryUnproven('deck_prefix_identity_unproven')
        return {code: list(values) for code, values in cards.items()}


CHRONOLOGY_CONTRACT = 'rendered_decision_boundary_v1'


def decision_boundary(line):
    """A line means before its gesture; the supported half-line means after it.

    Unsupported fractional heuristics do not acquire an invented tie-break.
    """
    if not isinstance(line, (int, float)) or line < 1:
        return None
    base = int(line)
    if line == base:
        return base, 0
    if line == base + 0.5:
        return base, 1
    return None


def chronological_order(times, *, turns=None):
    """Sort exact (line, subdecision) keys; emitter order cannot break ties."""
    import numpy as np
    times = np.asarray(times)
    if times.ndim != 2 or times.shape[1] != 2 or not np.issubdtype(times.dtype, np.integer):
        raise ValueError('TD requires explicit integer decision boundaries; re-extract unordered data')
    if np.any(times[:, 0] < 1) or np.any(times[:, 1] < 0):
        raise ValueError('TD decision chronology is unproven; data remains policy-only')
    if len({tuple(row) for row in times.tolist()}) != len(times):
        raise ValueError('TD decision boundaries are ambiguous; emitter order is not chronology')
    order = np.lexsort((times[:, 1], times[:, 0]))
    if turns is not None and np.any(np.diff(np.asarray(turns)[order]) < 0):
        raise ValueError('TD decision boundaries contradict turn order')
    return order
