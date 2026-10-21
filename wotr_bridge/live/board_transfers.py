"""Render a count-conserving endpoint for one fungible figure type."""


def balanced_transfers(minus, plus):
    """Return deterministic physical transfers, or None if counts differ.

    Call only after the engine has committed the decision and ruled out
    casualties of this type. This chooses physical piece IDs, not game moves.
    Named characters and different nations/unit types never share a call.
    """
    from collections import Counter
    from board import sanitize
    delta = Counter()
    for rows, sign in ((minus, -1), (plus, 1)):
        for region, count in rows:
            if int(count) != count or count < 0:
                return None
            delta[sanitize(str(region))] += sign * int(count)
    if sum(delta.values()):
        return None
    destinations = iter(sorted((region, n) for region, n in delta.items() if n > 0))
    target, remaining = next(destinations, (None, 0))
    transfers = []
    for source, count in sorted((region, -n) for region, n in delta.items() if n < 0):
        while count:
            take = min(count, remaining)
            transfers.append((source, target, take))
            count -= take
            remaining -= take
            if not remaining:
                target, remaining = next(destinations, (None, 0))
    return transfers


def physical_move_ops(view, nick, pids, destination):
    """The native command source is the actual zone of each selected PID."""
    import emitter
    groups = {}
    for pid in pids:
        source = emitter._piece_zone_code(view, pid)
        groups.setdefault(source, []).append(pid)
    return [('move', {'nick': nick, 'pieces': pieces, 'frm': source, 'to': destination})
            for source, pieces in groups.items()]
