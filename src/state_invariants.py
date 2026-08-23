"""Process-wide hard-invariant violation sink.

The iteration 33-34 per-action state checker that lived here
(check_state + five structural invariants) was deleted; what remains
is the violation CHANNEL: ai.py hard-failure paths append record dicts
({'code': ..., 'ctx': ..., ...}) to VIOLATIONS from inside exception
handlers.  Writer codes today: combat_card_menu_failed,
action_exec_failed, eye_phase_hijacked.

Every append also PRINTS the record (loud-failure doctrine): the
writers are try/except-wrapped diagnostics deep in self-play, and a
silent list would discard every violation at process exit.  The list
itself stays inspectable for showcases/batch tooling.

This module must stay import-light: the writers import it inside
'except Exception: pass' blocks, so any import failure here silently
kills the whole channel.
"""


class _LoudList(list):
    def append(self, item):
        try:
            txt = repr(item).encode('ascii', 'backslashreplace').decode()
        except Exception:
            txt = '<unprintable violation record>'
        print('[STATE-INVARIANT] ' + txt, flush=True)
        list.append(self, item)


VIOLATIONS = _LoudList()
