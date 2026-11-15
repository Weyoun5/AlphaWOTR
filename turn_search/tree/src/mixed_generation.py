"""Mixed-generation play: a net of an OLDER encoding generation inside a NEWER engine (D6P-MIXED-GEN-ADAPTER;
D6 plan S1(e) and section 6, 2026-10-23).

USER D6-GO (verbatim): "once it's all working and good, rerun training. don't gen more games, just rerun training.
then do evals to determine strength change".  The evaluations play retrained v53 nets in the same game as the v52 nets
deployed at the time, so the engine runs at v53 (WOTR_ACT_GEN=v53) and every v52 net must be fed its own v52 inputs.
(The D7 release of 2026-10-26 has since deployed a v53 pair: wotr_bridge/launch_head.py NET_TAG / NET_GEN.)

THE VIEW (the documented slice of D6 plan S1(e)).  v53 = v52 + APPENDED tails (encoding_v53), so a v52 net reads

    observation = v53_observation[:4647]        candidate rows = v53_candidates[:, :157]

(regions, chance rows and action kinds do not depend on the generation; the AI menu projection is the same code under
both generations, only its contract stamp differs).  This is bit-identical to the v52 encoding with ONE documented
exception, inherited from encoding_v53 (R3-V53-FOLLOWONS, rescan v040): the prefix's overstack-excess lane
(OVERSTACK_EXCESS_LANE) counts the excess over the REAL stacking cap under v53 (event_effects._stack_cap) and over the
pre-v040 flag-only cap under v52 (event_effects._stack_cap_flag_only).  The two caps differ only while a besieged-
stronghold flag is stale (the entry holds no besieger), which cleanup_sieges prevents in play; in that state a v52 net
would read the corrected excess.  Pinned on a constructed stale-flag state by src/test_d6prep_D6P_MIXED_GEN_ADAPTER.py;
selfplay_actor_critic/tests/test_d6prep_D6P_MIXED_GEN_ADAPTER_sac.py shows no difference at all on real games (raw
policy and Python search, bit for bit).

OPT-IN, NEVER SILENT.  WOTR_ADMIT_GENERATIONS lists the older generations a run admits (e.g. "v52").  Unset (the
default), checkpoint_view() returns None before reading anything and every loader keeps refusing a checkpoint of
another generation exactly as before.  A checkpoint of the engine's OWN generation never gets a view either, so a v52
pair (the corpus_live pair) under WOTR_ACT_GEN=v52 and the deployed v53 pair under WOTR_ACT_GEN=v53 load on the
unchanged path, bit for bit, opt-in or not.

EACH NET ITS OWN GENERATION, read from its checkpoint contract stamp (observation_contract, action_encoding_contract,
ai_policy_contract, input_dim, action_dim - every field must match one generation exactly).  The view lives ON THE
NET (GenerationViewNet, a WOTRValueNet subclass built at the net's own widths), so it survives every re-wrap of the
model (the PolicyActor re-identities of offline_eval / paired_eval) and serves both inference paths:
  * raw policy (selfplay_actor_critic.policy.forward_ragged): view_actor_input() turns each engine ActorInput into the
    v52 one before ANY module reads it (backbone, scorer, policy delta, chance head, credit head, kind biases);
  * Python search (ai.load_value_net -> _get_priors_and_value, _pv_forward_many, _eval_forward_many, mcts_search,
    arena_scale's base-arch local nets): _process_input / _embed_action_encs read the prefix of whatever the engine
    hands them.
Confidence providers (checkpoint conf_* stamps) read engine-width vectors and are refused in a view.

NATIVE SEARCH (R17-DOCSTRINGS, 2026-10-27; this replaces the D6 plan 4d note that the native route speaks only v52).
Since R3-V53-FOLLOWONS the native search controller encodes the engine's own generation, v52 or v53
(native_search_core/inference controller_binding.encoding_widths and numeric_bridge.observation_width; the live box's
encode_state_v53 / af_encode_v53), and the shipped v53 head searches on it (wotr_bridge/launch_head.py HEAD_ENV:
WOTR_ACT_GEN=NET_GEN with WOTR_LIVE_SEARCH_BACKEND=native).  An evaluation of nets of the engine's own generation may
therefore use either search route.  The view is not served on the native route: no native-route caller loads a view
net (the live head loads its own generation only; HEAD_ENV sets no WOTR_ADMIT_GENERATIONS).  The controller would hand
a view net engine-width vectors through the same Python inference calls (numeric_bridge: forward_backbone /
score_actions_spatial, _pv_forward_many, _eval_forward_many), which read the prefix through _process_input /
_embed_action_encs as on the Python route, but that combination is not validated or pinned, so a mixed-generation
evaluation uses the raw policy or the Python search route.
"""
from dataclasses import dataclass, replace
import os

import numpy as np

ENV = 'WOTR_ADMIT_GENERATIONS'
# v52 lane counting the units in excess of the stacking cap in the overstack regions (see the module docstring):
# the only v52-prefix lane whose v53 value can differ (stale besieged flag only; pinned by the src test).
OVERSTACK_EXCESS_LANE = 3725


class MixedGenerationError(ValueError):
    """A mixed-generation request this engine cannot serve exactly; never answered by a default or a pad."""


@dataclass(frozen=True)
class GenerationSpec:
    name: str
    observation: str
    observation_width: int
    actions: str
    action_width: int
    ai_policy: str

    def stamp(self):
        """The checkpoint metadata that identifies this generation (compatibility.assess_legacy_artifact's keys)."""
        return {'observation_contract': self.observation, 'action_encoding_contract': self.actions,
                'ai_policy_contract': self.ai_policy, 'input_dim': self.observation_width,
                'action_dim': self.action_width}


# The self-play generations (selfplay_actor_critic.contracts.Contract knows v52 and v53).  engine_spec() checks the
# running engine's own entry against its live values, so a stale row here is refused loudly, never served.
SPECS = {
    'v52': GenerationSpec('v52', 'v52_observed_combat_round_v3', 4647, 'observed_combat_choices_v8', 157,
                          'elite_bonus_draw_combat_victim_mighty_v3'),
    'v53': GenerationSpec('v53', 'v53_hand_threat_context_v1', 4745, 'v53_identity_route_choices_v1', 199,
                          'elite_bonus_draw_combat_victim_mighty_ring_twin_v4'),
}
# engine generation -> the older generations whose observation and action vectors are its exact prefix
PREFIX_GENERATIONS = {'v53': ('v52',)}
_STAMP_KEYS = tuple(SPECS['v52'].stamp())


def admitted_generations():
    """The generations WOTR_ADMIT_GENERATIONS admits (a tuple; empty when unset).  Unknown names are refused."""
    raw = os.environ.get(ENV, '')
    names = tuple(dict.fromkeys(part.strip().lower() for part in raw.replace(';', ',').split(',') if part.strip()))
    unknown = [n for n in names if n not in SPECS]
    if unknown:
        raise MixedGenerationError('%s=%r names unknown generation(s) %s (known: %s)'
                                   % (ENV, raw, unknown, ', '.join(SPECS)))
    return names


def engine_spec(ai):
    """The running engine's generation spec, checked against the engine's live contract values."""
    generation = getattr(ai, 'ACT_GEN', None)
    if generation != getattr(ai, '_OBSERVATION_GENERATION', None) or generation not in SPECS:
        raise MixedGenerationError('mixed-generation play needs an engine of a self-play generation (%s); this one '
                                   'runs ACT_GEN=%r / observation %r'
                                   % (', '.join(SPECS), generation, getattr(ai, '_OBSERVATION_GENERATION', None)))
    spec = SPECS[generation]
    import ai_policies
    import trace_decisions
    live = {'observation_contract': ai.OBSERVATION_CONTRACT,
            'action_encoding_contract': trace_decisions.ACTION_ENCODING_CONTRACT,
            'ai_policy_contract': ai_policies.POLICY_VERSION,
            'input_dim': int(ai.get_feature_dim()), 'action_dim': int(ai.ACTION_ENCODING_DIM)}
    if live != spec.stamp():
        raise MixedGenerationError('the %s engine reports %r but mixed_generation.SPECS says %r; update SPECS'
                                   % (generation, live, spec.stamp()))
    if generation == 'v53' and getattr(ai, '_V53_ACTION_TAIL_START', None) != SPECS['v52'].action_width:
        raise MixedGenerationError('the v53 action tail does not start at the v52 width %d'
                                   % SPECS['v52'].action_width)
    return spec


def stamp_generation(metadata):
    """The generation whose contract stamp `metadata` carries EXACTLY (types included), else None."""
    if not hasattr(metadata, 'get'):
        return None
    stamp = {key: metadata.get(key) for key in _STAMP_KEYS}
    for spec in SPECS.values():
        expected = spec.stamp()
        if all(type(stamp[k]) is type(v) and stamp[k] == v for k, v in expected.items()):
            return spec.name
    return None


def checkpoint_view(ai, metadata, where=''):
    """The generation spec a checkpoint must be VIEWED through in this engine, or None.

    None (the caller takes its unchanged path, which loads or refuses exactly as before) when WOTR_ADMIT_GENERATIONS
    is unset, when the checkpoint is of the engine's own generation, and when its stamp is not one admitted
    generation.  A spec when the stamp is an admitted generation whose vectors are the exact prefix of the engine's.
    An admitted generation that is NOT such a prefix (a newer net in an older engine) is refused loudly."""
    if not os.environ.get(ENV, '').strip():
        return None
    admitted = admitted_generations()
    found = stamp_generation(metadata)
    engine = engine_spec(ai)
    if found is None or found == engine.name or found not in admitted:
        return None
    if found not in PREFIX_GENERATIONS.get(engine.name, ()):
        raise MixedGenerationError('%s: a %s net cannot play in a %s engine: its inputs are not a prefix of the '
                                   'engine\'s (%s=%s)' % (where, found, engine.name, ENV, ','.join(admitted)))
    return SPECS[found]


def _prefix(value, width, engine_width, what):
    """The first `width` columns of the last axis: as is at the net's own width, sliced at the engine's width."""
    have = int(value.shape[-1])
    if have == width:
        return value
    if have != engine_width:
        raise MixedGenerationError('%s has %d columns; this net reads %d (its own generation) or the first %d of the '
                                   'engine\'s %d' % (what, have, width, width, engine_width))
    return value[..., :width]


_VIEW_CLASSES = {}


def view_net_class(ai):
    """GenerationViewNet for this engine module's WOTRValueNet (a subclass, so every isinstance check holds)."""
    base = ai.WOTRValueNet
    cls = _VIEW_CLASSES.get(base)
    if cls is not None:
        return cls

    class GenerationViewNet(base):
        """A WOTRValueNet of an older generation inside a newer engine.  Built at the net's OWN widths (the same
        modules, parameters and construction order as in its own engine, so the same RNG draws); reads the exact
        prefix of every engine vector.  No parameter or buffer is added: the state dict is the checkpoint's."""

        def __init__(self, view, engine, side=None):
            super().__init__(view.observation_width, side=side, action_dim=view.action_width)
            self.generation = view.name
            self.engine_generation = engine.name
            self.view_widths = (view.observation_width, view.action_width)
            self.engine_widths = (engine.observation_width, engine.action_width)

        def _process_input(self, x):
            x = _prefix(x, self.view_widths[0], self.engine_widths[0], 'observation')
            return super()._process_input(x.contiguous())

        def _embed_action_encs(self, action_encodings):
            action_encodings = _prefix(action_encodings, self.view_widths[1], self.engine_widths[1], 'action rows')
            return super()._embed_action_encs(action_encodings.contiguous())

        def view_actor_input(self, item):
            """The ActorInput (selfplay_actor_critic.policy) this net reads: observation and candidate rows cut to its
            own widths; regions, chance rows and kinds unchanged."""
            observation = _prefix(item.observation, self.view_widths[0], self.engine_widths[0], 'observation')
            candidates = _prefix(item.candidates, self.view_widths[1], self.engine_widths[1], 'action rows')
            if observation is item.observation and candidates is item.candidates:
                return item
            return replace(item, observation=np.ascontiguousarray(observation),
                           candidates=np.ascontiguousarray(candidates))

    _VIEW_CLASSES[base] = GenerationViewNet
    return GenerationViewNet


def build_view_net(ai, view, side=None):
    """A fresh GenerationViewNet of generation `view` for this engine (weights as constructed; callers load)."""
    return view_net_class(ai)(view, engine_spec(ai), side=side)
