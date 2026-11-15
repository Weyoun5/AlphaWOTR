"""LIVE1104 W13 - one judge for "the human's reply was refused in one clear sentence".

Why this exists (wave 2, RED_LIST W13 integration reds): many negative controls
pinned the PREFIX of one refusal family ("That move is not legal").  F13/C13
(OVERVIEW D2; chat.refusal_reply_unmatched) deliberately answer an unidentified
chat reply with a plainer family, "That reply does not answer your <decision>.
<what to do>", and W11 renames some decision labels again.  A prefix pin then
fails on BETTER text, so the negative controls went red for wording, not for
behaviour.  The judge below asserts the SUBSTANCE instead:

  * at least one chat in the reply window is a refusal of one of the live
    bridge's refusal families (REFUSAL_PREFIXES);
  * the refusal is a full sentence (>= 20 chars, ends in '.', '!' or '?');
  * it names the open decision or what to do (any of `names`, case-blind);
  * no chat in the window claims the reply was taken (forbid substrings).

The caller still asserts that NOTHING was committed and the decision is still
open - that is the behaviour the negative control exists for.  The judge
never accepts silence, a stall sentence ("I cannot continue", "matches none of
its N legal options") or a self-repair ask as a refusal.
"""

REFUSAL_PREFIXES = (
    'That move is not legal',             # chat.refusal_illegal (R10)
    'That reply does not answer your ',   # chat.refusal_reply_unmatched (F13/C13)
    'That is not a command here',         # chat.not_a_command_* family
)

# never a refusal: these mean the server stalled / asked for a repair
STALL_MARKS = ('i cannot continue', 'match none of', 'matches none of',
               'self-repair', 'held silently')


def refusals(chats):
    """The chats of `chats` that open with a refusal family's prefix."""
    return [str(c) for c in chats or () if str(c).startswith(REFUSAL_PREFIXES)]


def judge_refusal(chats, names=(), forbid=()):
    """[] when `chats` holds one clear refusal sentence; else the problems.

    names  - words of which at least one must appear in the refusal (the
             decision's label or the words that answer it), case-blind.
    forbid - substrings no chat of the window may contain (a commit voice,
             e.g. chat.PICK_COMMITTED_PREFIX)."""
    probs = []
    chats = [str(c) for c in chats or ()]
    refs = refusals(chats)
    if not refs:
        probs.append('no refusal sentence among %r' % (chats[:3],))
        return probs
    ref = refs[0]
    low = ref.lower()
    if len(ref) < 20 or not ref.rstrip().endswith(('.', '!', '?')):
        probs.append('refusal is not a full sentence: %r' % ref)
    if any(m in low for m in STALL_MARKS):
        probs.append('refusal is a stall / repair sentence: %r' % ref)
    if names and not any(str(n).lower() in low for n in names):
        probs.append('refusal names none of %r: %r' % (tuple(names), ref))
    for c in chats:
        for f in forbid or ():
            if f and f in c:
                probs.append('a chat claims the reply was taken (%r): %r'
                             % (f, c))
    return probs


def self_test():
    """In-file sensitivity of the judge itself (run by the W13 test)."""
    ok = judge_refusal(['That reply does not answer your Fellowship Guide. '
                        'Place your Guide in the Guide box, then type "done".'],
                       names=('guide',))
    assert ok == [], ok
    ok = judge_refusal(['That move is not legal: that is not a published row.'],
                       names=('row',))
    assert ok == [], ok
    # silence, a stall, a too-short text, a missing name, a commit voice
    assert judge_refusal([]), 'silence accepted'
    assert judge_refusal(['I cannot continue on the hunt decision: your last '
                          'board line matches none of its 2 legal options.'])
    assert judge_refusal(['That move is not legal']), 'fragment accepted'
    assert judge_refusal(['That reply does not answer your Hunt resolution. '
                          'Type "corruption".'], names=('guide',))
    assert judge_refusal(['That move is not legal: no.', 'Committed: x'],
                         forbid=('Committed:',))
    return True
