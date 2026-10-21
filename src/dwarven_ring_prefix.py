"""Use the actual public board prefix at a Dwarven Ring gesture."""
import copy
import re


def remove_marker_hesitations(life, raw):
    """Remove only exact, effect-free counter stage/return pairs from a lifecycle."""
    kept = list(life[:1])
    cancelled = set()
    index = 1
    while index < len(life):
        first = life[index]
        second = life[index + 1] if index + 1 < len(life) else None
        if (second is not None and first['op'] == second['op'] == 'move'
                and first['ids'] == second['ids'] and len(first['ids']) == 1
                and first['from'] == second['to'] == 'FP Dice'
                and str(first['to']).startswith('Current') and second['from'] == first['to']
                and first['player'] == second['player']
                and all(not text.strip() or re.fullmatch(r'<[^<>~]+> [^<]*', text)
                        for text in raw[first['raw_line']:second['raw_line'] - 1])):
            cancelled.update((first['raw_line'], second['raw_line']))
            index += 2
        else:
            kept.append(first)
            index += 1
    return kept, cancelled


def reroll_complete(proof):
    """A different face class ends this reroll; counter cleanup can follow later."""
    import game_logic
    from game_state import ActionDieFace
    letters = {'W': 'WILL_OF_WEST', 'C': 'CHARACTER', 'P': 'EVENT',
               'M': 'MUSTER', 'H': 'MUSTER_ARMY', 'A': 'ARMY', 'E': 'EYE'}
    try:
        old = ActionDieFace[letters[proof['old_face']]]
        results = [ActionDieFace[letters[row['face']]] for row in proof['results']]
        classes = game_logic._DWARVEN_FACE_CLASSES
        return bool(results) and all(classes[face] & classes[old] for face in results[:-1]) and not (classes[results[-1]] & classes[old])
    except (KeyError, TypeError):
        return False


def literal_snapshot(record):
    snapshot = record.get('decision_truth')
    if not isinstance(snapshot, dict):
        return None
    # This is the renderer's pre-gesture snapshot. The raw physical reroll
    # validator separately certifies ownership, original face and timing.
    # A later batch boundary can include administrative board changes; it
    # cannot replace an available literal pre-gesture observation.
    if not isinstance(snapshot.get('armies'), dict) or not isinstance(snapshot.get('fel'), dict):
        return None
    return copy.deepcopy(snapshot)
