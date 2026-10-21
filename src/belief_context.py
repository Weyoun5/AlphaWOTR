"""Confidence for hypothetical opponent replies must not inherit root secrets."""
from contextlib import contextmanager

from reply_evaluation import validate_root_context
from evaluation_context import preserved_evaluation_context


@contextmanager
def belief_confidence(ai, pre_action_state, root_side):
    """Use the hypothetical pre-action root view as this world's reference.

    Call after constructing an actor-conditioned world, around its successor
    evaluation. This keeps enabled confidence semantics; it does not disable
    flags or consult another player's network. Peak/mean/root state is local
    to the hypothetical world, and the outer search is restored on every exit.
    """
    validate_root_context(ai,root_side,require_model=False)
    with preserved_evaluation_context(ai):
        active=(ai._CONF_LEAF or ai._CONF_TRACK) and ai._CONF_FN is not None
        vector=ai.encode_state(pre_action_state,root_side) if active else None
        ai._conf_search_reset(vector,root_side)
        yield
