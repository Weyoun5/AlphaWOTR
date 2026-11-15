"""Public staged card cargo: current category and already selected pieces."""
import numbers
import numpy as np

CATEGORIES = tuple(('unit', nation, kind)
                   for nation in ('ISENGARD', 'SAURON', 'SOUTHRON_EASTERLINGS')
                   for kind in ('REGULAR', 'ELITE')) + (
    ('nazgul', 'SAURON', ''), ('named', 'Witch King', ''),
    ('named', 'Mouth of Sauron', ''),
) + tuple(('leader', nation, '')
          for nation in ('ISENGARD', 'SAURON', 'SOUTHRON_EASTERLINGS'))
TAIL_DIM = 2 * len(CATEGORIES)
OFFSET = 133


def encode_tail(action):
    result = np.zeros(TAIL_DIM, dtype=np.float32)
    if action.action_type != 'card_option':
        return result
    choices = action.params.get('choices') or {}
    key = choices.get('_cargo_key')
    if key is not None:
        result[CATEGORIES.index(tuple(key))] = 1.
    seen = set()
    for row in choices.get('cargo', ()):
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise ValueError('Malformed card cargo row')
        category, count = tuple(row[:3]), row[3]
        if (category in seen or isinstance(count, bool)
                or not isinstance(count, numbers.Integral) or not 1 <= count <= 10):
            raise ValueError('Duplicate or invalid card cargo count')
        seen.add(category)
        result[len(CATEGORIES) + CATEGORIES.index(category)] = count / 10.
    return result
