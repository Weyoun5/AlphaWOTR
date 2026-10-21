"""Selected action semantics and shape; historical generations stay explicit."""
import os

CARD_CHOICE_GENERATIONS = ('v45', 'v46', 'v47', 'v48', 'v49', 'v50', 'v51', 'v52')
CARD_CHOICE_CONTRACT = 'casualty_card_choices_v3'
# v52 v6 distinguishes Rage/Lidless compositions and Axe/Horn payments;
# v7 appends exact cargo category/composition lanes (133 -> 157).
# v8 identifies Help Unlooked For's source army in existing region lane 14.
# Historical generations retain their historical identity for offline reads.
PREVIOUS_V52_ACTION_CONTRACT = 'observed_combat_choices_v6'
LEGACY_CONTRACT = 'casualty_operations_v2'
CARD_CHOICE_TAIL_DIM = 12
ACTION_GENERATION = os.environ.get('WOTR_ACT_GEN', '').strip().lower()
ACTION_PREFIX_DIM = 93 if ACTION_GENERATION in ('v41', *CARD_CHOICE_GENERATIONS) else 92
CARD_CHOICE_TAIL_ENABLED = ACTION_GENERATION in CARD_CHOICE_GENERATIONS
ACTION_ENCODING_DIM = ACTION_PREFIX_DIM + (CARD_CHOICE_TAIL_DIM if CARD_CHOICE_TAIL_ENABLED else 0)
ACTION_ENCODING_CONTRACT = CARD_CHOICE_CONTRACT if CARD_CHOICE_TAIL_ENABLED else LEGACY_CONTRACT
if ACTION_GENERATION == 'v52':
    from combat_round_observation import ACTION_TAIL_DIM, ACTION_CONTRACT
    ACTION_ENCODING_DIM += ACTION_TAIL_DIM
    from card_cargo_observation import TAIL_DIM as CARGO_TAIL_DIM
    ACTION_ENCODING_DIM += CARGO_TAIL_DIM
    ACTION_ENCODING_CONTRACT = ACTION_CONTRACT

def validate_active_width(width):
    if width != ACTION_ENCODING_DIM:
        raise ValueError('Action generation %r requires width%d, got%d' %
                         (ACTION_GENERATION, ACTION_ENCODING_DIM, width))
