"""Opt-in evaluation hooks; observer owns sidecar IO, forecasts and scores."""
from functools import wraps
from combat_forecast_observer import CombatForecastObserver


def install(ai):
    """Install once in a game worker. No inference, RNG or game-state changes."""
    existing = getattr(ai, '_combat_forecast_eval_adapter', None)
    if existing is not None:
        return existing
    if not hasattr(ai, '_COMBAT_FORECAST_OBSERVER'):
        raise RuntimeError('Combat forecast search hooks are not installed')
    if ai._COMBAT_FORECAST_OBSERVER is not None:
        raise RuntimeError('A different combat forecast observer is installed')
    observer = CombatForecastObserver(ai)
    original_episode = ai._run_single_episode
    original_execute = ai._traced_execute
    original_accept = ai._accept_combat_round_result
    original_battle = ai.game_logic.battle

    @wraps(original_episode)
    def episode(fp_net, sh_net, config):
        success = False
        try:
            observer.start_episode(config)
            result = original_episode(fp_net, sh_net, config)
            trace = (result.get('game_stats') or {}).get('trace_path') if isinstance(result, dict) else None
            if not trace:
                raise RuntimeError('Successful episode is missing final trace identity')
            observer.final_trace(trace)
            success = True
            return result
        finally:
            observer.end_episode(success)

    @wraps(original_execute)
    def execute(state, action, tracer):
        try:
            observer.before_action(state, action, tracer)
            result = original_execute(state, action, tracer)
            observer.after_action(state, action, tracer)
            return result
        except BaseException as exc:
            observer.action_failed(exc)
            raise

    @wraps(original_accept)
    def accept(state, pc, result, atk_card_id, def_card_id):
        # Observer checks in_real_action: search applies use this engine
        # function too, but must never publish imagined outcomes as actuals.
        observer.round_result(state, pc, result, atk_card_id, def_card_id)
        return original_accept(state, pc, result, atk_card_id, def_card_id)

    @wraps(original_battle)
    def battle(state, *args, **kwargs):
        info = getattr(state, '_last_round_call', None)
        if (observer.in_real_action and kwargs.get('max_rounds') == 1
                and isinstance(info, dict) and len(args) >= 2
                and args[0] == info.get('atk_region') and args[1] == info.get('def_region')
                and kwargs.get('attacker_card_id') == info.get('atk_card')
                and kwargs.get('defender_card_id') == info.get('def_card')
                and kwargs.get('choices') == info.get('choices')):
            observer.before_round(state, info)
        return original_battle(state, *args, **kwargs)

    ai._run_single_episode = episode
    ai._traced_execute = execute
    ai._accept_combat_round_result = accept
    ai.game_logic.battle = battle
    ai._COMBAT_FORECAST_OBSERVER = observer
    ai._combat_forecast_eval_adapter = observer
    return observer
