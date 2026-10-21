"""A source upgrade may retain historical backups, never relabel or replay them.

Package LBP (2026-09-12): a `kind: environment` hop carries a reviewed search-SETTING change
(REVIEWED_ENVIRONMENT_TRANSITION_KEYS, today WOTR_LEAF_BATCH) with its own approval schema;
every other environment key, and any hop that mixes kinds, keeps the refusal."""
from pathlib import Path
import copy,hashlib,json,shutil,tempfile,unittest
import batch_recovery as br

def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class RuntimeSourceTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.live=[self.root/'live_fp_ring.pt',self.root/'live_sh_military.pt']
        for i,path in enumerate(self.live):path.write_bytes(b'test-only checkpoint inventory '+bytes([i]))
        self.old={'sources':{'ai':'old'},'globals':{'lr':1e-5},'environment':{'reward':'profile'}}
        self.new=copy.deepcopy(self.old);self.new['sources']['ai']='new'
        self.journal=br.BatchJournal(self.root,self.live,[('fp','ring'),('sh','military')],self.old)
        self.journal.bootstrap();self.marker=self.journal._marker()
        self.record_path=self.make_record(0,self.old)
        self.marker.update(generation=0,record_sha256=br.digest(self.record_path));br._publish(self.journal.marker,self.marker)
        self.record_bytes=self.record_path.read_bytes();self.inventory=self.journal.inventory()

    def make_record(self,index,controls):
        folder=self.root/('.batch_journal/txn_%06d'%index);folder.mkdir(parents=True)
        before=self.journal.inventory()
        for rel in before:
            target=folder/'backup'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(self.root/rel,target)
        (folder/'rng.pt').write_bytes(b'test-only RNG inventory')
        record={'version':br.VERSION,'index':index,'controls':controls,'before':before,'inputs':{},'rng_sha256':br.digest(folder/'rng.pt')}
        br._publish(folder/'record.json',record);return folder/'record.json'

    def journal_for(self,controls):return br.BatchJournal(self.root,self.live,[('fp','ring'),('sh','military')],controls)

    def install(self,entries,final):
        br._publish(self.root/'.batch_journal/runtime_source_transitions.json',{'schema':'reviewed_runtime_source_transitions_v1','transitions':entries})
        marker=copy.deepcopy(self.marker);marker['controls']=final;br._publish(self.journal.marker,marker)
        return self.journal_for(final)

    def source_entry(self,before,after,approved=True,name='approval.json'):
        proof={'schema':'reviewed_runtime_source_transition_v1','passed':approved,'before_controls_sha256':identity(before),
               'after_controls_sha256':identity(after),'committed_history_preserved':True}
        approval=self.root/name;br._publish(approval,proof)
        return {'before':before,'after':after,'approval':{'path':str(approval),'sha256':br.digest(approval)}}

    def transition(self,after=None,approved=True):
        after=self.new if after is None else after
        return self.install([self.source_entry(self.old,after,approved)],after)

    def env_entry(self,before,after,keys,kind='environment',schema='reviewed_runtime_environment_transition_v1',proof_keys=None,name='approval_env.json'):
        proof={'schema':schema,'passed':True,'before_controls_sha256':identity(before),'after_controls_sha256':identity(after),
               'committed_history_preserved':True,'changed_environment_keys':sorted(keys) if proof_keys is None else proof_keys}
        approval=self.root/name;br._publish(approval,proof)
        entry={'kind':kind,'environment_keys':keys,'before':before,'after':after,'approval':{'path':str(approval),'sha256':br.digest(approval)}}
        return entry

    def env_transition(self,key='WOTR_LEAF_BATCH',value='8',keys=None,also_change_sources=False,**entry_options):
        after=copy.deepcopy(self.old);after['environment'][key]=value
        if also_change_sources:after['sources']['ai']='new'
        keys=[key] if keys is None else keys
        return self.install([self.env_entry(self.old,after,keys,**entry_options)],after)

    # ---- source hops (unchanged contract) ----
    def test_unapproved_source_change_is_still_refused(self):
        marker=copy.deepcopy(self.marker);marker['controls']=self.new;br._publish(self.journal.marker,marker)
        with self.assertRaises(RuntimeError):self.journal_for(self.new).recover()
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)

    def test_approved_transition_preserves_record_bytes_models_and_next_index(self):
        current=self.transition();current.recover()
        self.assertEqual(current.next_index(),1)
        self.assertEqual(current.inventory(),self.inventory)
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)
        self.assertEqual(current._marker()['record_sha256'],br.digest(self.record_path))

    def test_changed_reward_cannot_be_authorized_as_source_upgrade(self):
        after=copy.deepcopy(self.new);after['environment']['reward']='different'
        with self.assertRaisesRegex(RuntimeError,'learning controls'):self.transition(after).recover()
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)

    def test_false_or_changed_approval_refuses_before_pruning(self):
        with self.assertRaises(RuntimeError):self.transition(approved=False).recover()
        current=self.transition();(self.root/'approval.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'approval changed'):current.recover()
        self.assertTrue(self.record_path.exists())

    def test_pending_training_from_old_runtime_cannot_use_history_exception(self):
        current=self.transition();pending=self.make_record(1,self.old)
        br._publish(current.pending,{'index':1,'sha256':br.digest(pending)})
        with self.assertRaisesRegex(RuntimeError,'Pending training controls'):current.recover()
        self.assertEqual(current.inventory(),self.inventory)

    def test_normal_pruning_can_remove_obsolete_backup_after_new_commit(self):
        current=self.transition();current.recover()
        next_record=self.make_record(1,self.new)
        marker=current._marker();marker.update(generation=1,record_sha256=br.digest(next_record));br._publish(current.marker,marker)
        current.prune_backups()
        self.assertFalse(self.record_path.exists());self.assertTrue(next_record.exists())
        self.assertEqual(current.inventory(),self.inventory)

    # ---- environment hops (package LBP) ----
    def test_allowlist_is_exactly_the_leaf_batch_knob(self):
        self.assertEqual(br.REVIEWED_ENVIRONMENT_TRANSITION_KEYS,('WOTR_LEAF_BATCH',))

    def test_reviewed_leaf_batch_environment_transition_resolves_history(self):
        current=self.env_transition();current.recover()
        self.assertEqual(current.next_index(),1)
        self.assertEqual(current.inventory(),self.inventory)
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)
        self.assertTrue(current._historical_controls_match(self.old))
        self.assertEqual(current.controls['environment']['WOTR_LEAF_BATCH'],'8')

    def test_unapproved_leaf_batch_change_is_still_refused(self):
        after=copy.deepcopy(self.old);after['environment']['WOTR_LEAF_BATCH']='8'
        marker=copy.deepcopy(self.marker);marker['controls']=after;br._publish(self.journal.marker,marker)
        with self.assertRaises(RuntimeError):self.journal_for(after).recover()
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)

    def test_environment_transition_refuses_unlisted_key(self):
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.env_transition(key='reward',value='different').recover()
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.env_transition(key='WOTR_SIMS_MENU_SCALE',value='1').recover()
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)

    def test_environment_transition_refuses_source_change_in_the_same_hop(self):
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.env_transition(also_change_sources=True).recover()

    def test_environment_transition_keys_must_name_exactly_the_change(self):
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.env_transition(keys=[]).recover()
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.env_transition(keys=['WOTR_LEAF_BATCH','WOTR_LEAF_BATCH']).recover()
        # the entry names the knob, but the environment did not change at all (walked as the second hop of a
        # chain: a hop whose `before` already equals the live controls is never consulted)
        same=copy.deepcopy(self.new)
        entries=[self.source_entry(self.old,self.new),self.env_entry(self.new,same,['WOTR_LEAF_BATCH'])]
        with self.assertRaisesRegex(RuntimeError,'unapproved controls'):self.install(entries,same).recover()

    def test_environment_transition_requires_its_own_approval_schema_and_keys(self):
        with self.assertRaisesRegex(RuntimeError,'environment transition is unapproved'):self.env_transition(schema='reviewed_runtime_source_transition_v1').recover()
        with self.assertRaisesRegex(RuntimeError,'environment transition is unapproved'):self.env_transition(proof_keys=['WOTR_SIMS_MENU_SCALE']).recover()
        with self.assertRaisesRegex(RuntimeError,'environment transition is unapproved'):self.env_transition(proof_keys=[]).recover()

    def test_unknown_transition_kind_is_refused(self):
        with self.assertRaisesRegex(RuntimeError,'Unknown runtime transition kind'):self.env_transition(kind='globals').recover()

    def test_source_hop_then_environment_hop_chain_resolves_the_committed_record(self):
        # the production r3 shape: committed record (sources old, K unset) -> source upgrade -> K=8
        final=copy.deepcopy(self.new);final['environment']['WOTR_LEAF_BATCH']='8'
        entries=[self.source_entry(self.old,self.new),self.env_entry(self.new,final,['WOTR_LEAF_BATCH'])]
        current=self.install(entries,final);current.recover()
        self.assertEqual(current.next_index(),1)
        self.assertTrue(current._historical_controls_match(self.old))
        self.assertTrue(current._historical_controls_match(self.new))
        self.assertEqual(self.record_path.read_bytes(),self.record_bytes)
        # a chain can never resolve to a controls value it does not end at
        other=copy.deepcopy(final);other['environment']['WOTR_LEAF_BATCH']='16'
        self.assertFalse(current._historical_controls_match(other))

if __name__=='__main__':unittest.main()
