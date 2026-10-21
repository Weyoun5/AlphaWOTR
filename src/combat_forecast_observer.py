"""Read-only, pre-outcome combat forecasts. No network calls or random draws."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from combat_forecast_scoring import score_forecast


def frozen(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str, allow_nan=False))


def action_key(action):
    if action is None:
        return None
    return json.dumps([action.action_type, action.die_idx, action.params], sort_keys=True, default=str)


def card_action(action):
    if action.action_type == 'combat_card_none':
        return 'no_card'
    if action.action_type == 'combat_card_play':
        return str(action.params['card_id'])
    return None


class CombatForecastObserver:
    def __init__(self, ai):
        self.ai = ai
        self.path = None
        self.pending = []
        self.ready = None
        self.active = False
        self.in_real_action = False
        self.sequence = 0
        self.search_number = 0
        self.metadata = {}

    def start_episode(self, config):
        trace = config.get('trace_path')
        if not trace:
            raise ValueError('Combat evaluation requires an episode trace path')
        path = Path(str(trace) + '.combat_eval.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh served episode cannot silently append to a previous attempt.
        with path.open('x', encoding='utf-8'):
            pass
        self.path = path
        self.pending, self.ready, self.active = [], None, False
        self.in_real_action = False
        self.sequence = self.search_number = 0
        self.round_contexts = {}
        self.metadata = frozen({'checkpoint_ids': config.get('checkpoint_ids', {}),
                                'configured_sims': config.get('mcts_sims', config.get('sims')), 'trace_path': str(trace)})
        self.emit({'kind': 'episode_start', 'schema': 'combat_forecast_eval_v1',
                   'evaluation_only': True, **self.metadata})

    def emit(self, row):
        if self.path is not None:
            with self.path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')

    def final_trace(self, trace_path):
        trace = Path(trace_path).resolve()
        if (self.path is None or trace.parent != self.path.resolve().parent
                or not trace.name.endswith('.trace.gz') or not trace.is_file()):
            raise ValueError('Final trace must exist beside its episode sidecar')
        destination = trace.with_name(trace.name + '.combat_eval.jsonl')
        if destination != self.path.resolve():
            # Exclusive hard-link creation cannot overwrite another attempt.
            # An interruption can leave both names, but never destroys either.
            os.link(self.path, destination)
            self.path.unlink()
            self.path = destination
        self.emit({'kind': 'final_trace', 'trace_path': str(trace)})

    def battle(self, state, pc=None):
        pc = pc if pc is not None else (getattr(state, 'pending_combat', None) or {})
        if not all(k in pc for k in ('attacker_region', 'defender_region', 'attacker_side', 'defender_side')):
            return None
        return [getattr(state, 'turn', getattr(state, 'turn_number', None)),
                self.ai.sanitize(pc['attacker_region']), self.ai.sanitize(pc['defender_region']),
                str(pc['attacker_side']), str(pc['defender_side']), int(pc.get('rounds_fought', 0))]

    def begin_search(self, root_state, root_side):
        self.search_number += 1
        self.ready = None
        self.root_battle = self.battle(root_state)
        pc = getattr(root_state, 'pending_combat', None) or {}
        self.active = self.root_battle is not None and pc.get('phase') in (
            'attacker_combat_card', 'defender_combat_card', 'combat_effect_choice')
        self.root_side = str(root_side)
        self.root_phase = pc.get('phase')
        self.root_actor = str(pc.get('deciding_side'))
        self.root_role = 'attacker' if root_side == pc.get('attacker_side') else 'defender'
        self.root_search = self.search_number
        self.card_sums = {}
        self.exact = {}
        self.hidden = None

    @staticmethod
    def root_branch(node):
        current = node
        while current.parent is not None and current.parent.parent is not None:
            current = current.parent
        return action_key(current.action) if current.parent is not None else None

    def card_policy(self, node, actions, probabilities):
        if not self.active or str(node.side) == self.root_side or self.battle(node.state) != self.root_battle:
            return
        pc = node.state.pending_combat or {}
        if pc.get('phase') not in ('attacker_combat_card', 'defender_combat_card'):
            return
        labels = [card_action(a) for a in actions]
        if not labels or any(label is None for label in labels):
            return
        key = self.root_branch(node)
        if key is None:
            return
        item = self.card_sums.setdefault(key, {'sums': defaultdict(float), 'count': 0,
                                               'target_side': str(node.side)})
        for label, probability in zip(labels, probabilities):
            item['sums'][label] += float(probability)
        item['count'] += 1

    def hidden_commit(self, root_state, root_side, mixture):
        if not self.active or self.root_role != 'defender' or self.root_phase != 'defender_combat_card':
            return
        total = sum(float(weight) for _, weight in mixture)
        probabilities = defaultdict(float)
        for card, weight in mixture:
            probabilities[str(card)] += float(weight) / total
        value = dict(probabilities)
        if self.hidden is not None and self.hidden != value:
            raise ValueError('Hidden-card forecast changed within the same information set')
        self.hidden = value

    def exact_round(self, node, ctx, dist):
        if not self.active or self.battle(node.parent.state) != self.root_battle:
            return
        key = self.root_branch(node)
        if key is None:
            return
        # This target is ordinary round hits presented for allocation, excluding
        # later Onslaught/Great Host effects. Early-ending/pre-hit-only outcomes
        # do not share this target and must not be relabeled as zero hits.
        if any(not outcome[16] for outcome in dist.outcomes):
            return
        probabilities = defaultdict(float)
        for outcome, mass in dist.outcomes.items():
            label = self.hit_label(outcome[17], outcome[18])
            probabilities[label] += float(mass)
        identity = hashlib.sha256(repr(ctx).encode()).hexdigest()
        self.exact.setdefault(key, {})[identity] = {
            'probabilities': dict(probabilities),
            'cards': [ctx.attacker_card, ctx.defender_card], 'context_sha256': identity}

    @staticmethod
    def hit_label(on_defender, on_attacker):
        return 'hits_on_defender=%d,hits_on_attacker=%d' % (int(on_defender), int(on_attacker))

    def finish_search(self, chosen_action, visit_info=None):
        if not self.active or chosen_action is None:
            self.active = False
            return None
        key = action_key(chosen_action)
        forecasts = []
        sampled = self.card_sums.get(key)
        if sampled:
            count = sampled['count']
            forecasts.append({'forecast_kind': 'opponent_card',
                              'probabilities': {card: p / count for card, p in sampled['sums'].items()},
                              'complete': False, 'samples': count, 'target_side': sampled['target_side'],
                              'method': 'mean_policy_over_visited_sampled_hands',
                              'limitation': 'Search-conditioned hand coverage; unvisited hands are not proven impossible.'})
        if self.hidden:
            forecasts.append({'forecast_kind': 'hidden_committed_card', 'probabilities': self.hidden,
                              'complete': True, 'samples': None, 'method': 'copy_weighted_public_candidate_pool'})
        contexts = list(self.exact.get(key, {}).values())
        # Different effect decisions under the same card pair must not be
        # silently averaged with equal weights. Log only uniquely defined pairs.
        by_pair = defaultdict(list)
        for item in contexts:
            by_pair[tuple(item['cards'])].append(item)
        for pair, items in by_pair.items():
            if len(items) == 1:
                forecasts.append({'forecast_kind': 'conditional_round_hits', **items[0],
                                  'complete': True, 'samples': None, 'method': 'full_exact_kernel_before_pruning'})
        visits = {}
        if isinstance(visit_info, dict) and 'visit_counts' in visit_info:
            counts = visit_info['visit_counts']
            played = int(visit_info['played'])
            visits = {'root_visit_total': sum(float(value) for value in counts),
                      'played_action_visits': float(counts[played]),
                      'rootmenu_count': len(counts)}
        self.ready = frozen({'battle': self.root_battle, 'root_side': self.root_side,
                             'root_role': self.root_role, 'root_phase': self.root_phase,
                             'search_id': self.root_search, 'action': key, 'forecasts': forecasts,
                             'search_visits': visits})
        self.active = False
        # Keep training's visit payload small; probabilities live only in sidecar.
        return {'schema': 'combat_forecast_eval_v1', 'search_id': self.root_search,
                'forecasts_available': len(forecasts), 'evaluation_only': True}

    def before_action(self, state, action, tracer):
        self.in_real_action = True
        self.action_round_results = []
        self.action_battle = self.battle(state)
        pc = getattr(state, 'pending_combat', None) or {}
        self.action_actor = str(pc.get('deciding_side'))
        self.action_phase = pc.get('phase')
        self.action_label = card_action(action)
        ready, self.ready = self.ready, None
        if ready and ready['battle'] == self.action_battle and ready['action'] == action_key(action):
            if not ready['forecasts']:
                self.emit({'kind': 'skip', 'forecast_kind': 'combat', 'reason': 'no_supported_forecast_in_search',
                           'battle': self.action_battle, 'search_id': ready['search_id']})
            for prediction in ready['forecasts']:
                self.sequence += 1
                forecast = dict(prediction, forecast_id=self.sequence, battle=ready['battle'],
                                root_side=ready['root_side'], root_role=ready['root_role'],
                                root_phase=ready['root_phase'], search_id=ready['search_id'],
                                **ready['search_visits'])
                preview = score_forecast(forecast['probabilities'], '__not_observed__',
                                         complete=forecast['complete'], samples=forecast['samples'])
                self.emit({'kind': 'forecast', **forecast, 'top': preview['top'],
                           'top_k_probability': preview['top_k_probability'],
                           'remainder_probability': preview['remainder_probability'],
                           'recorded_before_actual_action': True})
                self.pending.append(forecast)
        # Private actual card identity is read only after freezing forecasts and
        # is never supplied to their collectors or policy networks.
        self.action_attacker_card = pc.get('atk_combat_card')

    def before_round(self, state, info):
        if not self.in_real_action:
            return
        battle = self.battle(state)
        if battle is None:
            return
        key = tuple(battle)
        self.round_contexts.pop(key, None)
        try:
            ctx = self.ai._kernel_ctx(state, info, info.get('atk_card'),
                                      info.get('def_card'), defer=True)
        except ValueError as exc:
            if str(exc) != 'empty battle side':
                raise
            self.emit({'kind': 'skip', 'forecast_kind': 'conditional_round_hits',
                       'battle': battle, 'reason': 'empty_battle_side_no_kernel_context'})
            return
        self.round_contexts[key] = hashlib.sha256(repr(ctx).encode()).hexdigest()

    def round_result(self, state, pc, result, atk_card_id, def_card_id):
        if not self.in_real_action or getattr(result, 'round_paused', False):
            return
        from cards import get_card
        names = [get_card(card).combat_name if card else None for card in (atk_card_id, def_card_id)]
        battle = self.battle(state, pc)
        self.action_round_results.append({'battle': battle, 'cards': names,
            'context_sha256': self.round_contexts.get(tuple(battle)) if battle is not None else None,
            'attacker_card_id': str(atk_card_id) if atk_card_id else 'no_card',
            'defender_card_id': str(def_card_id) if def_card_id else 'no_card',
            'deferred': bool(getattr(result, 'casualties_deferred', False)),
            'hits': self.hit_label(getattr(result, 'deferred_hits_on_defender', 0),
                                   getattr(result, 'deferred_hits_on_attacker', 0))})

    def score(self, forecast, observed):
        value = score_forecast(forecast['probabilities'], observed, complete=forecast['complete'],
                               samples=forecast['samples'])
        self.emit({'kind': 'actual', 'forecast_kind': forecast['forecast_kind'],
                   'forecast_id': forecast['forecast_id'], 'battle': forecast['battle'],
                   'root_phase': forecast['root_phase'], 'score': value})

    def after_action(self, state, action, tracer):
        self.in_real_action = False
        remaining = []
        for forecast in self.pending:
            observed = None
            matching = forecast['battle'] == self.action_battle
            if matching and forecast['forecast_kind'] == 'opponent_card' and self.action_label is not None:
                if forecast['target_side'] == self.action_actor:
                    observed = self.action_label
            if matching and forecast['forecast_kind'] == 'hidden_committed_card':
                if self.action_phase == 'defender_combat_card' and self.action_label is not None:
                    observed = str(self.action_attacker_card) if self.action_attacker_card else 'no_card'
            finished_round = False
            for actual in self.action_round_results:
                if actual['battle'] != forecast['battle']:
                    continue
                finished_round = True
                if (forecast['forecast_kind'] == 'conditional_round_hits'
                        and actual['cards'] == forecast['cards']
                        and actual.get('context_sha256') == forecast['context_sha256']):
                    if actual['deferred']:
                        observed = actual['hits']
            if observed is not None:
                self.score(forecast, observed)
            elif finished_round:
                self.emit({'kind': 'skip', 'forecast_kind': forecast['forecast_kind'],
                           'forecast_id': forecast['forecast_id'], 'reason': 'condition_not_realized_or_target_unavailable'})
            elif forecast['battle'] != self.battle(state):
                self.emit({'kind': 'skip', 'forecast_kind': forecast['forecast_kind'],
                           'forecast_id': forecast['forecast_id'], 'reason': 'left_battle_or_round_before_target'})
            else:
                remaining.append(forecast)
        self.pending = remaining
        current = self.battle(state)
        self.round_contexts = {key: value for key, value in self.round_contexts.items()
                               if current is not None and key == tuple(current)}
        self.action_round_results = []

    def action_failed(self, exception):
        self.in_real_action = False
        self.action_round_results = []
        self.emit({'kind': 'execution_error', 'error_type': type(exception).__name__})

    def end_episode(self, success):
        for forecast in self.pending:
            self.emit({'kind': 'skip', 'forecast_kind': forecast['forecast_kind'],
                       'forecast_id': forecast['forecast_id'], 'reason': 'episode_ended_before_target'})
        self.pending = []
        self.ready = None
        self.in_real_action = False
        self.emit({'kind': 'episode_end', 'success': bool(success), 'forecasts': self.sequence})
        self.path = None
