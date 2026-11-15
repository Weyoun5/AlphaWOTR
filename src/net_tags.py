"""Which checkpoint pair a net tag of 'auto' means for an engine generation.

A checkpoint loads only under its own encoding generation (WOTR_ACT_GEN), so
'auto' tries the deployed pair of the running generation first, then the
legacy chain.  wotr_bridge/launch_head.py names the head's pair explicitly
(NET_TAG at NET_GEN); the head's tag is first for its generation here.
Used by engine_play._resolve_net_tag and wotr_bridge/live/own_model.py.
"""

AUTO_TAGS_BY_GENERATION = {
    'v53': ('d6_r1sp_r2sp',),
    'v52': ('corpus_live',),
}
LEGACY_AUTO_TAGS = ('corpus_live', 'corpus_v41jt', 'corpus_v41td', 'corpus_v41',
                    'corpus_v37jt', 'corpus_v37td', 'corpus_v37')


def auto_tags(generation):
    """The 'auto' tag order for an engine generation (WOTR_ACT_GEN value)."""
    first = AUTO_TAGS_BY_GENERATION.get(str(generation or '').strip().lower(), ())
    return tuple(first) + tuple(tag for tag in LEGACY_AUTO_TAGS if tag not in first)
