"""Printed-rule probabilities plus independent engine enumeration for fixed choices.

These complete-round contexts do not represent adaptive owner policies.
"""
import unittest
from dataclasses import replace
import combat_dist as cd
import cert_combat_dist as cert
from game_state import Nation


class KernelCardTiming(unittest.TestCase):
    def check_oracle(self,context):
        actual=cd.exact_distribution(context,want_witness=context.defer_casualties)
        expected,nodes=cert.oracle_distribution(lambda:cert._run_engine_round(context))
        self.assertLess(cert.tv_distance(actual.outcomes,expected),1e-12)
        self.assertAlmostEqual(actual.total(),1,places=12)
        if actual.witness is not None:
            for outcome,plan in actual.witness.items():
                got=cd.run_with_plan(plan,False,lambda:cert._run_engine_round(context))
                self.assertEqual(got,outcome)
        return actual

    def test_charge_occurs_before_relentless_full_selfwipe(self):
        context=cd.ExchangeContext(cd.ArmySpec(0,1,Nation.GONDOR),
            cd.ArmySpec(2,0,Nation.SAURON),attacker_card='Charge',
            defender_card='Relentless Assault',choices=(('def_effect_count',2),),
            defer_casualties=True)
        dist=self.check_oracle(context)
        # One unmodified Charge die hits on5/6 before SH sacrifices survivors.
        self.assertAlmostEqual(sum(p for o,p in dist.outcomes.items() if o[2]==1),1/3)
        for outcome in dist.outcomes:
            self.assertEqual(outcome[0:2],(0,0))
            self.assertEqual(outcome[10:14],(0,0,False,True))

    def test_named_hobbit_sacrifice_probability_is_one_third(self):
        context=cd.ExchangeContext(cd.ArmySpec(1,0,Nation.SAURON),
            cd.ArmySpec(3,0,Nation.GONDOR,leaders=(cd.LeaderSpec('leader'),
                cd.LeaderSpec('companion','Meriadoc Brandybuck',1,1))),
            defender_card='Heroic Death',choices=(('def_effect_count',1),
                ('def_effect_target','Meriadoc Brandybuck')))
        dist=self.check_oracle(context)
        self.assertAlmostEqual(sum(p for o,p in dist.outcomes.items()
                                  if o[15]==('C:Meriadoc Brandybuck',)),1/3)
        self.assertTrue(all('L' not in o[15] and o[10]==3 for o in dist.outcomes))

    def test_blade_redirects_army_hit_after_mumakil_comparison(self):
        context=cd.ExchangeContext(
            cd.ArmySpec(1,0,Nation.GONDOR,leaders=(cd.LeaderSpec('companion','Meriadoc Brandybuck',1,1),)),
            cd.ArmySpec(0,1,Nation.SOUTHRON_EASTERLINGS,leaders=(cd.LeaderSpec('minion','Mouth of Sauron'),)),
            attacker_card='Blade of Westernesse',defender_card='Mumakil',defer_casualties=True)
        self.check_oracle(context)

    def test_selected_mighty_payer_and_fateful_nazgul_match_engine(self):
        mighty=cd.ExchangeContext(cd.ArmySpec(2,0,Nation.GONDOR,leaders=(
            cd.LeaderSpec('companion','Aragorn',3,2),cd.LeaderSpec('companion','Meriadoc Brandybuck',1,1))),
            cd.ArmySpec(2,0,Nation.SAURON),attacker_card='Mighty Attack',
            choices=(('atk_effect_target','Aragorn'),),defer_casualties=True)
        self.check_oracle(mighty)
        self.check_oracle(replace(mighty,attacker_card='Fateful Strike',
            defender=cd.ArmySpec(2,0,Nation.SAURON,leaders=(cd.LeaderSpec('nazgul'),
                cd.LeaderSpec('minion','Mouth of Sauron'))),choices=(('atk_effect_target','nazgul'),)))

    def test_natural_one_remains_rerollable_at_plus_four(self):
        # The existing independently enumerated named case includes Ent's Rage
        # with Deadly Strife: 1+4>=5 still leaves natural1 a rerollable miss.
        context=dict(cert.build_cases())['entsrage_def_vs_deadlystrife_atk_oneedge']
        self.check_oracle(context)


if __name__=='__main__':unittest.main()
