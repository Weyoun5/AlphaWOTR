"""Public leadership costs. Fixed outcomes are separate from Dread choices."""
import re
import unicodedata

_FORFEIT = re.compile(r'(?:ff\s*(\d+)|dread and despair:\s*forfeit\s+(\d+)\s+nazgul leadership)\s*[.!]?', re.I)
_FIXED_FORFEIT = re.compile(
    r'(they are terrible|cruel as death):\s*forfeit\s+(\d+)\s+nazgul leadership\s*[.!]?', re.I)


def _normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', str(text or ''))
                   if not unicodedata.combining(c)).strip()

def dread_forfeit_count(text):
    normalized = _normalize(text)
    match=_FORFEIT.fullmatch(normalized)
    return int(match.group(1) or match.group(2)) if match else None


def fixed_forfeit_report(text):
    """Return (effect_slug, fixed_count), never a learned choice or legal action.

    A consumer must still bind actor, current revealed card, and timing. Exact
    fixed costs alone are recognized; malformed quantities are not corrected.
    """
    match = _FIXED_FORFEIT.fullmatch(_normalize(text))
    if not match:
        return None
    name = match.group(1).casefold()
    count = int(match.group(2))
    if count != {'they are terrible': 1, 'cruel as death': 2}[name]:
        return None
    return name.replace(' ', '_'), count
