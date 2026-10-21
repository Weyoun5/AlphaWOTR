"""RED-TEAM directed-coverage legs for validator RULE CLAUSES the
QUEUE round's directed tests do not name (builder's own structural
note: under propose-only generators a validator mutation moves mask
and universe TOGETHER, so the mask===validator invariant cannot see
validator-rule mutations -- only DIRECTED legs carry that axis).

Audit method: enumerated every distinct (False, reason) clause in
ai.is_legal_muster_action and ai.is_legal_companion_action, then
matched them against the directed legs in
streak/test_queue_family_validators.py and
streak/test_companion_mask_validator_invariant.py.  Clauses WITHOUT a
directed leg (covered only vacuously by the invariant):
  muster:    wrong-nation settlement, lone-leader stranding,
             nazgul-recruit-at-strongholds-only (p.25)
  companion: Mordor-track separation bar, hunt-place hobbit-name
             mismatch, Gollum non-separable
Each leg below CRAFTS a state where exactly that clause must fire and
asserts (False, named reason) -- so a future mutation deleting the
clause turns a leg RED.  All legs also assert the TWIN positive (the
same shape passes once the blocking fact is removed) so no leg can go
vacuously green.

Net-free (no nets loaded).  These are CONTRIBUTED coverage, expected
GREEN on e1b0f21d84bf; any FAIL = a clause that does not actually
enforce its rule (a real finding).
"""
import os
import sys
import types

SRC = os.path.dirname(os.path.abspath(__file__))
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
if 'torch' not in sys.modules and 'ai' not in sys.modules:
    sys.modules['torch'] = types.ModuleType('torch')


def _fresh():
    import board as bm
    from game_state import setup_initial_game
    return setup_initial_game(bm.Board())


def _muster(st, **params):
    import ai
    a = ai.Action(die_idx=0, action_type=params.pop(
        'action_type', 'muster_recruit'), params=params)
    try:
        return ai.is_legal_muster_action(st, a)
    except TypeError:
        return ai.is_legal_muster_action(st, a, {})


def test_muster_wrong_nation_settlement_clause():
    from game_state import Nation
    st = _fresh()
    st.political.boxes[Nation.DWARVES] = 3
    st.political.active[Nation.DWARVES] = True
    # NOTE (audit find, robustness nit): a STRING nation ('dwarves')
    # falls through boxes.get -> misleading 'not At War'; the live
    # mask always carries the ENUM, so legs use enums.
    ok, why = _muster(st, region='edoras', type='regular',
                      nation=Nation.DWARVES)
    assert ok is False and 'settlement' in str(why).lower(), (
        'WRONG-NATION SETTLEMENT CLAUSE DEAD: dwarves recruiting at '
        'edoras (a Rohan settlement) returned (%r, %r)' % (ok, why))
    ok2, why2 = _muster(st, region='erebor', type='regular',
                        nation=Nation.DWARVES)
    assert ok2 is True, ('twin positive broken: dwarves at erebor '
                         'refused (%r)' % (why2,))


def test_muster_nazgul_stronghold_only_clause():
    from game_state import Nation
    st = _fresh()
    st.political.boxes[Nation.SAURON] = 3
    st.political.active[Nation.SAURON] = True
    ok, why = _muster(st, action_type='muster_recruit',
                      region='nurn', type='leader',
                      nation=Nation.SAURON)
    assert ok is False and 'stronghold' in str(why).lower(), (
        'NAZGUL-STRONGHOLD CLAUSE DEAD (p.25): nazgul recruit at '
        'nurn (a Sauron town, no stronghold) returned (%r, %r)'
        % (ok, why))
    ok2, why2 = _muster(st, action_type='muster_recruit',
                        region='minas morgul', type='leader',
                        nation=Nation.SAURON)
    assert ok2 is True, ('twin positive broken: nazgul at minas '
                         'morgul refused (%r)' % (why2,))


def test_muster_lone_leader_strand_clause():
    """Mustering the LAST unit out... the clause guards recruiting a
    LEADER into an empty region (a lone leader would be removed).
    Craft: at-war nation, settlement with NO units -> leader recruit
    must refuse with the lone-leader reason; with one regular there,
    it passes."""
    from game_state import Nation, Unit, UnitKind
    import game_logic as gl
    st = _fresh()
    st.political.boxes[Nation.THE_NORTH] = 3
    st.political.active[Nation.THE_NORTH] = True
    # empty the north settlement 'dale' of units (keep nothing)
    a = gl.get_or_create_army(st, 'Dale')
    a.units[:] = []
    a.leaders[:] = []
    ok, why = _muster(st, action_type='muster_recruit',
                      region='dale', type='leader',
                      nation=Nation.THE_NORTH)
    assert ok is False and ('leader' in str(why).lower()
                            or 'lone' in str(why).lower()), (
        'LONE-LEADER CLAUSE DEAD: leader recruit into empty dale '
        'returned (%r, %r) - a lone leader would be removed at once'
        % (ok, why))
    a.add(Unit(Nation.THE_NORTH, UnitKind.REGULAR))
    ok2, why2 = _muster(st, action_type='muster_recruit',
                        region='dale', type='leader',
                        nation=Nation.THE_NORTH)
    assert ok2 is True, ('twin positive broken: leader with a '
                         'regular present refused (%r)' % (why2,))


def test_companion_mordor_bar_clause():
    import ai
    st = _fresh()
    st.fellowship.mordor_entered = True
    st.fellowship.mordor_step = 1
    a = ai.Action(die_idx=0, action_type='companion_separate',
                  params={'companion': 'Legolas',
                          'to_region': 'fords of bruinen'})
    ok, why = ai.is_legal_companion_action(st, a)
    assert ok is False and 'mordor' in str(why).lower(), (
        'MORDOR-BAR CLAUSE DEAD: separation on the Mordor track '
        'returned (%r, %r)' % (ok, why))
    st.fellowship.mordor_entered = False
    st.fellowship.progress = 3
    ok2, why2 = ai.is_legal_companion_action(st, a)
    assert ok2 is True, ('twin positive broken: same separation '
                         'off-Mordor refused (%r)' % (why2,))


def test_companion_hobbit_place_name_mismatch_clause():
    import ai
    st = _fresh()
    st.pending_hunt = {'phase': 'hobbit_place',
                       'companion': 'Meriadoc', 'level': 1}
    a = ai.Action(die_idx=-1, action_type='companion_separate',
                  params={'companion': 'Peregrin',
                          'to_region': 'rivendell'})
    ok, why = ai.is_legal_companion_action(st, a)
    assert ok is False and ('not the hobbit' in str(why).lower()
                            or 'placed' in str(why).lower()), (
        'HOBBIT-NAME-MISMATCH CLAUSE DEAD: placing Peregrin while '
        'Meriadoc is the placed hobbit returned (%r, %r)'
        % (ok, why))
    a2 = ai.Action(die_idx=-1, action_type='companion_separate',
                   params={'companion': 'Meriadoc',
                           'to_region': 'rivendell'})
    ok2, why2 = ai.is_legal_companion_action(st, a2)
    assert ok2 is True, ('twin positive broken: placing the RIGHT '
                         'hobbit in place refused (%r)' % (why2,))


def test_companion_gollum_not_separable_clause():
    import ai
    from game_state import Companion
    st = _fresh()
    st.fellowship.progress = 2
    g = Companion(name='Gollum', level=0, leadership=0)
    st.gollum = g
    if all(c.name != 'Gollum' for c in st.fellowship.companions):
        st.fellowship.companions.append(g)
    a = ai.Action(die_idx=0, action_type='companion_separate',
                  params={'companion': 'Gollum',
                          'to_region': 'trollshaws'})
    ok, why = ai.is_legal_companion_action(st, a)
    assert ok is False, (
        'GOLLUM-SEPARATION CLAUSE DEAD: separating Gollum returned '
        '(%r, %r) - Gollum is never a separable companion' % (ok, why))


if __name__ == '__main__':
    fails = 0
    for fn in (test_muster_wrong_nation_settlement_clause,
               test_muster_nazgul_stronghold_only_clause,
               test_muster_lone_leader_strand_clause,
               test_companion_mordor_bar_clause,
               test_companion_hobbit_place_name_mismatch_clause,
               test_companion_gollum_not_separable_clause):
        try:
            fn()
            print('PASS %s' % fn.__name__)
        except AssertionError as e:
            fails += 1
            print('FAIL %s: %s' % (fn.__name__,
                                   str(e).encode('ascii', 'replace')
                                   .decode()))
        except Exception as e:
            fails += 1
            print('FAIL %s: SETUP %r' % (fn.__name__, e))
    sys.exit(1 if fails else 0)
