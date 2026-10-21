"""Literal AiFeed records from the real Java downgrade button.

Controls.304 and Controls.296 render the same loss sentence. The actual
sidecar distinguishes a pending reduction from a whole Elite removal.
"""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'wotr_bridge')]
import ai,board,game_state as gs,game_logic as gl
from live import gesture_match as gm,live_parser
FP,SH=gs.Side.FREE_PEOPLES,gs.Side.SHADOW
E=gs.Nation.ELVES

def position(card=False):
    st=gs.setup_initial_game(board.load_board());gl.initialize_decks(st)
    if card:
        from cards import get_card
        import event_effects
        event_effects._route_card_casualties(st,'lorien',FP,2,
            get_card('SHADOW_Character_Dreadful_Spells'),{}, {'effects':[]})
    else:
        st.pending_combat=dict(phase='casualty_allocation',alloc_stage='attacker',
            attacker_region='lorien',defender_region='dol guldur',
            attacker_side=FP,defender_side=SH,deciding_side=FP,
            atk_hits=2,def_hits=0,rounds_fought=1,is_siege=False)
    return st

def native_records():
    literals=[('$AuditPlayer kills Elven Elite in Lorien.',
        {'seq':16,'raw_new':True,'nick':'AuditPlayer','tags':['Controls.304','Controls.305','Game.1928'],'pids':['092'],'segs':[[],['092'],[],[]]}),
        ('$AuditPlayer moves Elven Regular from reinforcements to Lorien.',
        {'seq':17,'raw_new':True,'nick':'AuditPlayer','tags':['Controls.313','Controls.307','Controls.314','Game.1928'],'pids':['268'],'segs':[[],['268'],[],[],[]]}),
        ('$AuditPlayer kills Elven Regular in Lorien.',
        {'seq':18,'raw_new':True,'nick':'AuditPlayer','tags':['Controls.296','Controls.297','Game.1928'],'pids':['268'],'segs':[[],['268'],[],[]]})]
    return [dict(live_parser.parse(line),meta=meta) for line,meta in literals]

class Window:
    die_side=die_face=None
    closed='test'
    def __init__(self,records): self.records=records
    def body_records(self): return self.records

class NativeCasualtyIntent(unittest.TestCase):
    def test_native_downgrade_prefix_never_commits_whole_elite(self):
        for card in (False,True):
            with self.subTest(card=card):
                st=position(card)
                result=gm.match_window(st,FP,Window(native_records()[:1]))
                self.assertNotEqual(result.status,'match',result)
                self.assertEqual(st.fp_casualties,{})

    def test_replacement_then_later_kill_preserve_two_distinct_casualties(self):
        for card in (False,True):
            with self.subTest(card=card):
                st=position(card)
                result=gm.match_window(st,FP,Window(native_records()[:2]))
                self.assertEqual(result.status,'match',result)
                st=result.post
                army=ai._armies_lookup(st,'lorien')
                self.assertEqual(sum(u.kind==gs.UnitKind.REGULAR for u in army.units),2)
                self.assertEqual(sum(u.kind==gs.UnitKind.ELITE for u in army.units),1)
                self.assertEqual(st.fp_casualties[E].get('regular',0),0)
                pending=st.pending_card_option if card else st.pending_combat
                self.assertEqual(pending['cas_hits' if card else 'atk_hits'],1)
                result=gm.match_window(st,FP,Window(native_records()[2:]))
                self.assertEqual(result.status,'match',result)
                self.assertEqual(result.post.fp_casualties[E]['regular'],1)
                self.assertEqual(result.post.fp_casualties[E]['elite'],1)

    def test_same_native_records_grouped_keep_reduction_identity(self):
        for card in (False,True):
            with self.subTest(card=card):
                result=gm.match_window(position(card),FP,Window(native_records()))
                self.assertEqual(result.status,'match',result)
                self.assertEqual(result.post.fp_casualties[E]['regular'],1)
                self.assertEqual(result.post.fp_casualties[E]['elite'],1)

    def test_native_whole_removal_button_commits_immediately(self):
        rec=native_records()[0]
        rec['meta']['tags']=['Controls.296','Controls.297','Game.1928']
        result=gm.match_window(position(),FP,Window([rec]))
        self.assertEqual(result.status,'match',result)
        self.assertEqual([a.params['operation'] for a in result.chain],['remove_elite'])
        self.assertEqual(result.post.fp_casualties[E].get('regular',0),0)

if __name__=='__main__': unittest.main(verbosity=2)
