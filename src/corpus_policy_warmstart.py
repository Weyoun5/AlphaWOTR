"""Explicit policy-only warm start; existing corpus arrays and pilot CE adapter.

This stage does not read z/phi or evaluate the value head. An explicit policy
admission manifest binds the exact consolidated arrays. No default CLI changes.
"""
from pathlib import Path
import collections,copy,hashlib,json,operator,os
import numpy as np
import torch
import ai,optimizer_continuity as oc,value_trainer_v2 as vt
from observation_training import ObservationTrainingScope
from trace_decisions import record_warm_start_update
from training_device import activate_training_device
POLICY_LR=1e-5

def sha(path):
 digest=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
 return digest.hexdigest()

def same(left,right):
 if isinstance(left,torch.Tensor):
  assert isinstance(right,torch.Tensor) and torch.equal(left.cpu(),right.cpu())
 elif isinstance(left,dict):
  assert isinstance(right,dict) and left.keys()==right.keys()
  for key in left:same(left[key],right[key])
 elif isinstance(left,(list,tuple)):
  assert type(left)==type(right) and len(left)==len(right)
  for a,b in zip(left,right):same(a,b)
 else:assert left==right

def inventory(folder):
 return {p.name:sha(p) for p in sorted(folder.iterdir()) if p.is_file()}

def chunks(array,bytes_per_chunk=8*1024*1024):
 # Bound temporary validation arrays even for a corpus larger than RAM.
 width=max(1,int(np.prod(array.shape[1:])))*array.dtype.itemsize
 step=max(1,bytes_per_chunk//max(1,width))
 for lo in range(0,len(array),step):yield array[lo:lo+step]


class PolicyRows:
 """Validated memory maps; convert only the rows in the current minibatch."""
 def __init__(self,fields):self.fields=fields
 def __len__(self):return len(self.fields['X'])
 def __getitem__(self,index):
  if isinstance(index,slice):return [self[i] for i in range(*index.indices(len(self)))]
  index=operator.index(index)
  if index<0:index+=len(self)
  if not 0<=index<len(self):raise IndexError(index)
  f=self.fields;lo,hi=int(f['off'][index]),int(f['off'][index+1])
  return {'x':np.array(f['X'][index]),'_menu':np.array(f['cands'][lo:hi],dtype='float32'),
      'regs':np.array(f['regs'][lo:hi],dtype='int64'),
      '_positive':np.array(f['pos'][lo:hi],dtype=bool),'phase':'consolidated'}


def load_rows(folder,side):
 # Outcome files can remain in the archive, but are never opened by this stage.
 fields={key:np.load(folder/(side+'_'+key+'.npy'),mmap_mode='r',allow_pickle=False)
         for key in ('X','off','cands','regs','pos','human')}
 x,off,cands,regs,pos,human=(fields[k] for k in ('X','off','cands','regs','pos','human'))
 n=len(x)
 if x.shape!=(n,ai.get_feature_dim()) or off.shape!=(n+1,) or human.shape!=(n,):raise ValueError('Invalid policy row layout')
 if off.dtype.kind not in 'iu' or human.dtype.kind not in 'iu' or off[0]!=0 or off[-1]!=len(cands):raise ValueError('Invalid candidate offsets')
 if cands.shape!=(len(cands),ai.ACTION_ENCODING_DIM) or regs.shape!=(len(cands),) or pos.shape!=(len(cands),):raise ValueError('Invalid candidate layout')
 if any(not np.isfinite(block).all() for array in (x,cands) for block in chunks(array)):raise ValueError('Nonfinite policy input')
 if regs.dtype.kind not in 'iu' or any(np.any(block < -1) or np.any(block>=ai.NUM_REGIONS) for block in chunks(regs)):raise ValueError('Invalid candidate region')
 if any(np.any((block!=0)&(block!=1)) for block in chunks(pos)):raise ValueError('Invalid positive flags')
 for lo in range(0,n,65536):
  hi=min(n,lo+65536);starts=off[lo:hi];ends=off[lo+1:hi+1];labels=human[lo:hi]
  # Compare before subtraction: unsigned descending offsets must not wrap.
  if np.any(ends<starts) or np.any(ends-starts<2) or np.any(ends>len(cands)):raise ValueError('Invalid candidate offsets')
  if np.any(labels<0) or np.any(labels>=ends-starts):raise ValueError('Human label is not a positive')
  chosen=starts.astype(np.int64)+labels.astype(np.int64)
  if not np.all(pos[chosen]==1):raise ValueError('Human label is not a positive')
 if not n:raise ValueError('No policy rows for '+side)
 return PolicyRows(fields)


def tensors(rows):
 xb=torch.as_tensor(np.stack([r['x'] for r in rows]),dtype=torch.float32,device=ai.DEVICE)
 return xb,[(torch.as_tensor(r['_menu'],device=ai.DEVICE),torch.as_tensor(r['regs'],dtype=torch.long,device=ai.DEVICE),torch.as_tensor(r['_positive'],device=ai.DEVICE)) for r in rows]

def scores(net,rows):
 xb,menus=tensors(rows);h,ne=net._process_input(xb);out=[]
 for i,(ae,ri,pos) in enumerate(menus):
  logits=net.action_scores(h[i:i+1],ne[i:i+1],ae,ri)
  out.append((logits,pos))
 return out

def evaluate(net,rows):
 net.eval();ce=correct=0;phases=collections.defaultdict(lambda:[0,0]);n=0
 with torch.no_grad():
  for lo in range(0,len(rows),16):
   batch=rows[lo:lo+16]
   for row,(logits,pos) in zip(batch,scores(net,batch)):
    loss=torch.logsumexp(logits,0)-torch.logsumexp(logits[pos],0)
    assert torch.isfinite(loss)
    ok=int(pos[logits.argmax()].item());ce+=float(loss);correct+=ok;n+=1
    phases[row['phase']][0]+=ok;phases[row['phase']][1]+=1
 return {'rows':n,'ce':ce/n,'correct':correct,'accuracy':correct/n,'phases':dict(phases)}

def train_epoch(net,rows,epoch):
 order=np.random.RandomState(95780+epoch).permutation(len(rows));total=0;seen=0
 with ObservationTrainingScope(net) as scope:
  heads=[p for name,p in net.named_parameters() if name.startswith('policy_delta')]
  opt=oc.optimizer_for(net,'policy',scope.parameter_groups(heads),scope,POLICY_LR)
  for lo in range(0,len(order),16):
   batch=[rows[i] for i in order[lo:lo+16]]
   losses=[torch.logsumexp(logits,0)-torch.logsumexp(logits[pos],0) for logits,pos in scores(net,batch)]
   loss=torch.stack(losses).mean();assert torch.isfinite(loss)
   opt.zero_grad(set_to_none=True);loss.backward()
   norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
   opt.step();oc.updated(net,'policy',opt);scope.record_step('policy');record_warm_start_update(net,'policy')
   total+=float(loss.detach())*len(batch);seen+=len(batch)
 return {'rows':seen,'ce':total/seen,'optimizer_steps':net._optimizer_continuity['components']['policy']['steps']}


def run(args):
 global POLICY_LR
 import train_corpus as bc
 if not (args.fp_seed and args.sh_seed and args.policy_output and args.policy_admission):
  raise ValueError('policy-warmstart requires both explicit seeds, --policy-output and --policy-admission')
 if args.epochs<1 or not np.isfinite(args.lr) or args.lr<=0:raise ValueError('Invalid policy training controls')
 if os.environ.get('WOTR_OPTIMIZER_STATE')!='persistent_adam_v1':raise ValueError('Persistent optimizer state is required')
 if os.environ.get('WOTR_OBSERVATION_TRAINING')!='full_network_v1':raise ValueError('Initial policy warm-start supports full_network_v1 only')
 if os.environ.get('WOTR_NATIVE','0')!='0':raise ValueError('Initial policy warm-start supports native=0 only')
 POLICY_LR=args.lr;activate_training_device();torch.set_num_threads(1)
 folder=Path(bc.DS_DIR).resolve();split=bc._require_dataset_contract()
 admission_path=Path(args.policy_admission).resolve()
 admission_bytes=admission_path.read_bytes();admission_sha256=hashlib.sha256(admission_bytes).hexdigest()
 admission=json.loads(admission_bytes)
 before=inventory(folder)
 if admission.get('scope')!='expert_policy_only_v1' or admission.get('policy_admitted') is not True or admission.get('value_admitted') is not False:raise ValueError('Explicit policy-only admission required')
 if admission.get('dataset_hashes')!=before:raise ValueError('Admitted policy dataset changed')
 if admission.get('learning_contract')!=split['learning_contract']:raise ValueError('Policy admission contract mismatch')
 seeds={'fp':Path(args.fp_seed).resolve(),'sh':Path(args.sh_seed).resolve()}
 if any(not p.is_file() for p in seeds.values()):raise ValueError('Missing explicit seed checkpoint')
 output=Path(args.policy_output).resolve()
 if output.exists() or output==folder or folder in output.parents or any(output==p or output in p.parents for p in seeds.values()):raise ValueError('Policy output must be a fresh independent directory')
 profiles={'fp':'ring','sh':'military'};ai.PLAYER_PROFILES.update(profiles)
 rows={side:load_rows(folder,side) for side in profiles}
 seed_hashes={side:sha(path) for side,path in seeds.items()}
 # Load and validate BOTH checkpoints before creating any output or training.
 nets={}
 for side,profile in profiles.items():
  net=ai.load_value_net(str(seeds[side]),side=side,profile=profile)
  if net is None:raise ValueError('Seed did not load')
  oc.load_training_state(net,str(seeds[side]));nets[side]=net
 def verify_inputs():
  if sha(admission_path)!=admission_sha256:raise ValueError('Policy admission changed during training')
  if before!=inventory(folder) or seed_hashes!={side:sha(path) for side,path in seeds.items()}:raise ValueError('Policy inputs changed during training')
 verify_inputs()
 output.mkdir(parents=True)
 result={'policy_only':True,'value_labels_used':False,'search_labels_used':False,'seed_hashes':seed_hashes,
   'dataset_hashes':before,'admission_sha256':admission_sha256,'results':{}}
 for side,net in nets.items():
  initial={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
  buffers={k:v.detach().cpu().clone() for k,v in net.named_buffers()}
  value_optimizer=copy.deepcopy(net._optimizer_continuity['components']['value'])
  policy_steps=(net._optimizer_continuity['components']['policy'] or {}).get('steps',0)
  report={'before_policy_steps':policy_steps,'rows':len(rows[side]),'epochs':[]}
  for epoch in range(1,args.epochs+1):
   verify_inputs()
   update=train_epoch(net,rows[side],epoch)
   assert update['optimizer_steps']==policy_steps+epoch*((len(rows[side])+15)//16)
   verify_inputs()
   path=output/(side+'_epoch'+str(epoch)+'.pt');vt.save_v2_checkpoint(net,side,str(path),profiles[side])
   restored=ai.load_value_net(str(path),side=side,profile=profiles[side]);oc.load_training_state(restored,str(path))
   same(net.state_dict(),restored.state_dict());same(value_optimizer,restored._optimizer_continuity['components']['value'])
   assert restored._optimizer_continuity['components']['policy']['steps']==update['optimizer_steps']
   report['epochs'].append(dict(update,checkpoint_sha256=sha(path),reload_weights_exact=True));net=restored
  for key,value in net.named_buffers():same(buffers[key],value)
  changed=[key for key,value in net.state_dict().items() if not torch.equal(initial[key],value.detach().cpu())]
  assert changed and not any(key.startswith('value_head') for key in changed)
  report.update(changed_parameters=changed,value_head_unchanged=True,value_optimizer_unchanged=True)
  verify_inputs()
  result['results'][side]=report
  (output/'progress.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 verify_inputs()
 result['passed']=True
 (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 return result
