"""Profile outcome counts, separated by opponent and reward-contract cohort.

This module reports outcomes; it never assigns training rewards or decides
trace admission. Callers should attach admission reports to game-log records.
Unreviewed historical results remain explicitly counted as unreviewed.
"""
from collections.abc import Mapping
import json


PROFILES = {'fp': frozenset({'mixed', 'ring', 'military'}),
            'sh': frozenset({'mixed', 'corruption', 'military'})}
TARGET_ROUTE = {('fp', 'ring'): 'ring_destroyed',
                ('sh', 'corruption'): 'ring_corruption',
                ('fp', 'military'): 'military',
                ('sh', 'military'): 'military'}
WIN_ROUTES = {'fp': frozenset({'military', 'ring_destroyed'}),
              'sh': frozenset({'military', 'ring_corruption'})}


def _rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'value': numerator / denominator if denominator else None}


def summarize_profile_results(records, *, side):
    """Return JSON-compatible counts/rates for each comparable cohort.

    Input records use games.jsonl fields: profiles={fp:..., sh:...}, winner
    (fp/sh/timeout), victory_type, and optional reward_contract. Opponent
    identity is explicit opponent_id, otherwise checkpoint_ids[other_side].
    Supply immutable checkpoint IDs, not a path overwritten between games.
    Missing identity/contract stays None and is marked as incomplete.

    Optional status is completed/accepted/rejected/error. An attached
    admission report overrides a completed result: an integrity rejection
    counts as rejected; a valid timeout excluded solely as truncated still
    counts as timeout. Error/rejected records need no terminal labels.
    Completed records with malformed labels count as errors, never losses.
    Malformed is a subset of errors. Unreviewed is a subset of decided plus
    timeouts. One input record represents one attempted game; no deduplication
    or missing worker-attempt inference is performed.

    Each cohort has attempted = decided + timeouts + rejected + errors and
    decided = wins + losses. All rates carry their exact denominator. Mixed
    profiles have no on/off-target classification (both counts/rates None).
    No rate across different cohorts is emitted: changing opponents is not
    evidence that the learner improved.
    """
    if side not in PROFILES:
        raise ValueError('side must be fp or sh')
    if isinstance(records, (Mapping, str, bytes)):
        raise ValueError('records must be an iterable of game records')
    opponent = 'sh' if side == 'fp' else 'fp'
    cohorts, diagnostics = {}, []
    attempted = 0

    for index, original in enumerate(records):
        attempted += 1
        problems = []

        def problem(code, detail):
            problems.append({'index': index, 'code': code, 'detail': detail})

        if not isinstance(original, Mapping):
            problem('record_schema', 'game record must be a mapping')
            record = {}
        else:
            record = original
        profiles = record.get('profiles')
        if not isinstance(profiles, Mapping):
            problem('profiles_schema', 'profiles must identify both player profiles')
            profiles = {}
        own_profile, opponent_profile = profiles.get(side), profiles.get(opponent)
        for player, value in ((side, own_profile), (opponent, opponent_profile)):
            if not isinstance(value, str) or value not in PROFILES[player]:
                problem('profile_label', '%s profile must be one of %s' %
                        (player, sorted(PROFILES[player])))
        # Invalid labels stay unknown; no alias or mission is inferred.
        if not isinstance(own_profile, str) or own_profile not in PROFILES[side]:
            own_profile = None
        if not isinstance(opponent_profile, str) or opponent_profile not in PROFILES[opponent]:
            opponent_profile = None
        identities = record.get('checkpoint_ids', {})
        if not isinstance(identities, Mapping):
            problem('checkpoint_ids_schema', 'checkpoint_ids must be a player-to-ID mapping')
            identities = {}
        opponent_id = record.get('opponent_id')
        if opponent_id is None:
            opponent_id = identities.get(opponent)
        contract = record.get('reward_contract')
        for label, value in (('opponent_id', opponent_id), ('reward_contract', contract)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                problem(label + '_schema', label + ' must be a nonempty string or None')
                if label == 'opponent_id':
                    opponent_id = None
                else:
                    contract = None
        key = (own_profile, opponent_profile, opponent_id, contract)
        if key not in cohorts:
            target = TARGET_ROUTE.get((side, own_profile))
            cohorts[key] = {
                'cohort': {'own_profile': own_profile, 'opponent_profile': opponent_profile,
                           'opponent_id': opponent_id, 'reward_contract': contract},
                'opponent_identity_known': opponent_id is not None,
                'cohort_complete': all(value is not None for value in key),
                'counts': {'attempted': 0, 'decided': 0, 'wins': 0, 'losses': 0,
                           'timeouts': 0, 'rejected': 0, 'errors': 0, 'malformed': 0,
                           'unreviewed': 0,
                           'on_target_wins': 0 if target else None,
                           'off_target_wins': 0 if target else None}}
        group = cohorts[key]
        counts = group['counts']
        counts['attempted'] += 1
        status = record.get('status', 'completed')
        if status not in ('completed', 'accepted', 'rejected', 'error'):
            problem('status_label', 'status must be completed, accepted, rejected, or error')
        if record.get('error') and status not in ('rejected', 'error'):
            status = 'error'
        admission = record.get('admission')
        reviewed = status in ('accepted', 'rejected', 'error')
        if admission is not None:
            if (not isinstance(admission, Mapping)
                    or type(admission.get('accepted')) is not bool):
                problem('admission_schema', 'admission must contain a boolean accepted field')
            elif status not in ('rejected', 'error'):
                reviewed = True
                error_records = admission.get('errors', [])
                if (not isinstance(error_records, (list, tuple))
                        or any(not isinstance(item, Mapping) for item in error_records)):
                    problem('admission_schema', 'admission errors must be a sequence of mappings')
                elif admission['accepted']:
                    if error_records or admission.get('valid') is False:
                        problem('admission_conflict', 'accepted admission contains integrity errors')
                elif (admission.get('valid') is True and error_records
                      and all(item.get('code') == 'truncated' for item in error_records)):
                    if (record.get('winner'), record.get('victory_type')) != ('timeout', 'timeout'):
                        problem('admission_conflict', 'truncated admission requires timeout labels')
                else:
                    status = 'rejected'
        if not problems and status not in ('rejected', 'error'):
            winner, route = record.get('winner'), record.get('victory_type')
            if winner == 'timeout':
                if route != 'timeout':
                    problem('termination_label', 'timeout winner requires timeout victory_type')
            elif not isinstance(winner, str) or winner not in WIN_ROUTES:
                problem('winner_label', 'winner must be fp, sh, or timeout')
            elif not isinstance(route, str) or route not in WIN_ROUTES[winner]:
                problem('victory_type_label', 'victory_type must be a legal route for the winner')
        if problems:
            counts['errors'] += 1
            counts['malformed'] += 1
            diagnostics.extend(problems)
        elif status in ('rejected', 'error'):
            counts['rejected' if status == 'rejected' else 'errors'] += 1
        elif record['winner'] == 'timeout':
            counts['timeouts'] += 1
            counts['unreviewed'] += int(not reviewed)
        else:
            counts['decided'] += 1
            counts['unreviewed'] += int(not reviewed)
            if record['winner'] != side:
                counts['losses'] += 1
            else:
                counts['wins'] += 1
                target = TARGET_ROUTE.get((side, own_profile))
                if target:
                    field = 'on_target_wins' if record['victory_type'] == target else 'off_target_wins'
                    counts[field] += 1

    for group in cohorts.values():
        counts = group['counts']
        target_applies = counts['on_target_wins'] is not None
        group['rates'] = {
            'win_rate_decided': _rate(counts['wins'], counts['decided']),
            'on_target_win_rate_decided': (
                _rate(counts['on_target_wins'], counts['decided']) if target_applies else None),
            'on_target_win_rate_attempted': (
                _rate(counts['on_target_wins'], counts['attempted']) if target_applies else None),
            'off_target_win_rate_wins': (
                _rate(counts['off_target_wins'], counts['wins']) if target_applies else None)}
    return {'side': side, 'attempted': attempted,
            'cohorts': sorted(cohorts.values(),
                              key=lambda group: json.dumps(group['cohort'], sort_keys=True)),
            'diagnostics': diagnostics}
