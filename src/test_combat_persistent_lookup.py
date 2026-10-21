"""Lossless combat-table storage, real AI integration, and engine witnesses."""
from pathlib import Path
import dataclasses,hashlib,json,os,random,shutil,sqlite3,subprocess,sys,tempfile,unittest,warnings
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
RUNTIME=HERE.parent
import ai
import combat_dist as cd,combat_lookup as lookup,build_combat_table as builder,cert_combat_dist as cert
COUNTS={}
BASELINE=lookup.runtime_manifest()
contexts=[dataclasses.replace(c,defer_casualties=True) for _,c in cert.build_cases()]
EXPECTED={}
OUT=DB=None
def equal(a,b):return (a.outcomes,a.hits_joint,a.witness)==(b.outcomes,b.hits_joint,b.witness)
def setUpModule():
 global OUT,DB,TEMP
 TEMP=tempfile.TemporaryDirectory(prefix='wotr_combat_lookup_')
 OUT=Path(TEMP.name).resolve();DB=OUT/'cert.sqlite'
 rs=random.getstate()
 try:
  builder.build(DB,contexts,{'kind':'existing-certification-contexts'})
  EXPECTED.update({c:cd.exact_distribution(c,True) for c in contexts})
 finally:random.setstate(rs)
def tearDownModule():
 if lookup._READER is not None:lookup._READER.close()
 lookup._READER=None;lookup._INITIALIZED=False;lookup._PID=None
 ai._KERNEL_MEMO.clear()
 TEMP.cleanup()
class Checks(unittest.TestCase):
 def setUp(self):
  self.env_patch=patch.dict(os.environ);self.env_patch.start();self.addCleanup(self.env_patch.stop)
  ai._KERNEL_MEMO.clear()
  if lookup._READER is not None:lookup._READER.close()
  lookup._READER=None;lookup._INITIALIZED=False;lookup._PID=None
  os.environ['WOTR_COMBAT_TABLE']=str(DB)
 def test_all_distribution_rows_and_rare_outcomes(self):
  r=lookup.Reader(DB);n=0;smallest=1
  for c,expected in EXPECTED.items():
   got=r.lookup(c,True);self.assertTrue(equal(got,expected));n+=len(got.outcomes)
   self.assertEqual(lookup.context_from_plain(lookup.plain(c)),c)
   smallest=min(smallest,min(got.outcomes.values()))
  r.close();COUNTS.update(contexts=len(EXPECTED),outcomes=n,smallest_mass=smallest)
 def test_stored_witnesses_execute_existing_engine(self):
  r=lookup.Reader(DB);n=0
  for c in contexts:
   got=r.lookup(c,True)
   for observation,plan in got.witness.items():
    def thunk():
     battle=cert.build_round(c)
     la,ld=list(battle.attacker_army.leaders),list(battle.defender_army.leaders)
     cert.cb.resolve_combat_round(battle,cert.round_choices(c))
     return cert.observe(battle,la,ld)
    self.assertEqual(cd.run_with_plan(plan,cd.plan_conf_case(c),thunk),observation);n+=1
  r.close();COUNTS['engine_witnesses']=n
 def test_readonly_refuses_writes(self):
  r=lookup.Reader(DB)
  with self.assertRaises(sqlite3.OperationalError):r.connection.execute('DELETE FROM entries')
  r.close()
 def test_candidate_hit_bypasses_exact_and_preserves_rng(self):
  c=contexts[0];rs=random.getstate()
  with patch.object(cd,'exact_distribution',side_effect=AssertionError('Unexpected exact calculation')):
   got=ai._kernel_dist(c,True)
  self.assertTrue(equal(got,EXPECTED[c]));self.assertEqual(random.getstate(),rs)
 def test_hot_memo_bypasses_persistent_layer(self):
  c=contexts[0];a=ai._kernel_dist(c,True)
  with patch.object(lookup,'lookup',side_effect=AssertionError('Hot memo touched persistent table')):
   self.assertIs(ai._kernel_dist(c,True),a)
 def test_uncatalogued_exact_fallback_rng(self):
  c=dataclasses.replace(contexts[0],shine=True,choices=(('disable_city_defense',True),))
  expected=cd.exact_distribution(c,True);rs=random.getstate()
  with patch.object(cd,'exact_distribution',wraps=cd.exact_distribution) as exact:
   got=ai._kernel_dist(c,True);self.assertEqual(exact.call_count,1)
  self.assertTrue(equal(got,expected));self.assertEqual(random.getstate(),rs)
 def test_non_deferred_no_witness_fallback(self):
  c=dataclasses.replace(contexts[0],defer_casualties=False)
  self.assertTrue(equal(ai._kernel_dist(c,False),cd.exact_distribution(c,False)))
  with self.assertRaises(ValueError):ai._kernel_dist(c,True)
 def test_disabled_without_configuration(self):
  del os.environ['WOTR_COMBAT_TABLE']
  c=contexts[0]
  with patch.object(cd,'exact_distribution',wraps=cd.exact_distribution) as exact:
   self.assertTrue(equal(ai._kernel_dist(c,True),EXPECTED[c]));self.assertEqual(exact.call_count,1)
 def test_fingerprint_change_refuses_and_falls_back(self):
  changed=dict(BASELINE,schema='bad')
  with patch.object(lookup,'runtime_manifest',return_value=changed):
   with self.assertRaises(ValueError):lookup.Reader(DB)
   with warnings.catch_warnings(record=True) as caught:
    self.assertTrue(equal(ai._kernel_dist(contexts[0],True),EXPECTED[contexts[0]]))
    self.assertTrue(any('disabled' in str(w.message) for w in caught))
 def test_corrupt_payload_refuses_and_falls_back(self):
  bad=OUT/'corrupt.sqlite';shutil.copy2(DB,bad)
  conn=sqlite3.connect(bad);conn.execute("UPDATE entries SET payload_sha='invalid'");conn.commit();conn.close()
  os.environ['WOTR_COMBAT_TABLE']=str(bad)
  with warnings.catch_warnings(record=True) as caught:
   self.assertTrue(equal(ai._kernel_dist(contexts[0],True),EXPECTED[contexts[0]]));self.assertTrue(caught)
 def test_context_identity_mismatch_refuses(self):
  bad=OUT/'bad_key.sqlite';shutil.copy2(DB,bad)
  conn=sqlite3.connect(bad);conn.execute("UPDATE entries SET context=x'00'");conn.commit();conn.close()
  r=lookup.Reader(bad)
  with self.assertRaises(ValueError):r.lookup(contexts[0],True)
  r.close()
 def test_two_independent_readers_same_immutable_file(self):
  before=hashlib.sha256(DB.read_bytes()).hexdigest()
  script='''import sys,hashlib,json;sys.path.insert(0,sys.argv[1]);import combat_lookup as c;from pathlib import Path;r=c.Reader(Path(sys.argv[2]));rows=r.connection.execute('SELECT context FROM entries').fetchall();n=0
for (raw,) in rows:
 d=r.lookup(c.context_from_plain(json.loads(raw)),True);n+=len(d.outcomes)
r.close();print(n)
'''
  children=[subprocess.Popen([sys.executable,'-c',script,str(RUNTIME/'src'),str(DB)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=os.environ.copy()) for _ in range(2)]
  outputs=[]
  for child in children:
   out,err=child.communicate(timeout=30);self.assertEqual(child.returncode,0,err);outputs.append(out.strip())
  self.assertEqual(outputs[0],outputs[1]);self.assertEqual(hashlib.sha256(DB.read_bytes()).hexdigest(),before)
 def test_offline_builder_never_overwrites_table(self):
  before=DB.read_bytes()
  with self.assertRaises(FileExistsError):builder.build(DB,[],{})
  self.assertEqual(DB.read_bytes(),before)

if __name__=='__main__':unittest.main(verbosity=2)
