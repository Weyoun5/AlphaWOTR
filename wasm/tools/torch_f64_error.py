"""How far torch's own float32 forward is from a float64 one on recorded leaves (the scale for the C++ parity bound).

    venv/bin/python -I wasm/tools/torch_f64_error.py ALPHAWOTR_CHECKOUT wasm/fixtures/nn_cases.ref
"""
import os,sys,struct,numpy as np
alpha=os.path.abspath(sys.argv[1]); cases=os.path.abspath(sys.argv[2])
sys.path[:0]=[alpha+'/src',alpha+'/wotr_bridge']
import launch_head as lh
for k,v in lh.HEAD_ENV: os.environ[k]=v
os.chdir(alpha+'/src')
import ai, torch, copy
from core import own_model
nets={0:own_model.load_own_net(lh.NET_TAG,ai.Side.FREE_PEOPLES),1:own_model.load_own_net(lh.NET_TAG,ai.Side.SHADOW)}
d64={s:copy.deepcopy(n).double().eval() for s,n in nets.items()}
data=open(cases,'rb').read(); off=8; cnt=struct.unpack_from('<I',data,4)[0]
worst=0; worstp=0
for c in range(cnt):
    side,rows,_=struct.unpack_from('<III',data,off); off+=12
    obs=np.frombuffer(data,'<f4',4745,off); off+=4745*4
    acts=np.frombuffer(data,'<f4',rows*199,off).reshape(rows,199); off+=rows*199*4
    regs=np.frombuffer(data,'<i4',rows,off); off+=rows*4
    off+=8; lg=np.frombuffer(data,'<f4',rows,off); off+=rows*4; pr=np.frombuffer(data,'<f4',rows,off); off+=rows*4; off+=1024
    if not rows: continue
    with torch.no_grad():
        n=d64[side]; x=torch.from_numpy(np.array([obs],np.float64)); bb,ne=n.forward_backbone(x)
        l64=n.action_scores(bb,ne,torch.from_numpy(acts.astype(np.float64)),torch.from_numpy(regs.astype(np.int64))).numpy()
    worst=max(worst,float(np.max(np.abs(lg-l64)/np.maximum(1,np.abs(l64)))))
    worstp=max(worstp,float(np.max(np.abs(pr-torch.softmax(torch.from_numpy(l64),0).numpy()))))
print('torch float32 vs float64: logits max rel %.3g, priors max abs %.3g'%(worst,worstp))
