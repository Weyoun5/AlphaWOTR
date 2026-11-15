"""Train the v42 nets on the v42 corpus, mirroring the recipe that
produced the CURRENT live nets, PLUS the two lessons of the v41 chain:

  1. SCALED2 HEAD, BELT OVER SUSPENDERS (SPCP-3 / SPTR-2 / run-7).
     The v41 chain ran env-less, ai's constructor defaulted to Tanh,
     and save() stamped tanh seeds while 42 percent of residual
     targets exceed the Tanh range.  train_corpus._fresh_head now
     defaults to scaled2 even with the env unset (SPB fix); this
     runner ALSO pins WOTR_VALUE_HEAD_ACT=scaled2 explicitly AND
     hard-asserts head_act=='scaled2' on every produced checkpoint in
     the self-check below.  A tanh stamp is a LOUD SystemExit - that
     assertion is the entire point of the re-seed.
  2. CONF-STAMP AFTER VALUE-TD (run_v41_scaled2.py recipe).
     value_td's torch.save writes a FRESH dict with no conf_* keys,
     so it DROPS the confidence stamps; conf-stamp must run last.

GENERATION vs VINTAGE (do not confuse the two):
  * WOTR_ACT_GEN=v41 is the ENCODING generation and stays v41 -
    ACTION_ENCODING_DIM 93, matching the corpus the samples were
    encoded with.  The checkpoints stamp action_dim 93, so
    ai.load_value_net's generation guard REFUSES to load them into a
    default (92) runtime - that is correct: the swap is a deliberate
    act, not an accident.
  * v42 is the CORPUS vintage (CORPUS_NET_TAG=v42): the extraction
    replayed through the 2026-08-29 engine (1.251-1.257, GL-5, 1.254
    casualty pendings, EX-8 payment, 1.230 declare window).

Stage list = the live nets' recipe (consolidate -> train (BC) -> eval
-> value-td, lam=0.8 rounds=3; NO train-joint - the live nets did not
have it) + conf-stamp last.  Outputs:
  wotr_{fp,sh}_corpus_v42.pt     (BC)
  wotr_{fp,sh}_corpus_v42td.pt   (value-td refit head; the SEEDS)
The USER's live nets are wotr_{fp,sh}_corpus_live.pt and are NOT
written to, so live play keeps running throughout the training.
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The env is set on os.environ (not just the child ENV) so the
# in-process self-check imports ai under the same generation the
# stages trained under.
os.environ['WOTR_ACT_GEN'] = 'v41'
os.environ['WOTR_VALUE_HEAD_ACT'] = 'scaled2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONHASHSEED'] = '0'
# GPU flip (USER 2026-08-30 'why is the training so slow'): the v40
# chain trained on GPU; the CPU pin was v41-runner inheritance.  The
# box's RTX 5090 passed the net-forward smoke; V42_CUDA overrides
# (default '0' = the GPU; set '-1' to force CPU).
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('V42_CUDA', '0')
sys.path.insert(0, os.path.join(ROOT, 'src'))

CORPUS = os.environ.get('V42_CORPUS') or os.path.join(
    ROOT, 'data', 'bc_corpus_v42_rendered')
DS = CORPUS + '_ds'
NET_TAG = os.environ.get('V42_NET_TAG', 'v42')
STAGES = os.environ.get('V42_STAGES',
                        'consolidate,train,eval,value-td,conf-stamp')

ENV = dict(os.environ,
           CORPUS_DIR=CORPUS,
           CORPUS_DS_DIR=DS,
           CORPUS_NET_TAG=NET_TAG,
           WOTR_ACT_GEN='v41',
           WOTR_VALUE_HEAD_ACT='scaled2',
           KMP_DUPLICATE_LIB_OK='TRUE',
           PYTHONIOENCODING='utf-8',
           PYTHONHASHSEED='0',
           CUDA_VISIBLE_DEVICES=os.environ.get('V42_CUDA', '0'))


def self_check():
    """Load every produced checkpoint and hard-assert the stamps.

    scaled2 assertion twice over: (a) the checkpoint meta field that
    load_value_net honors forever, (b) the actual head module after a
    real ai.load_value_net (full generation guards exercised under
    WOTR_ACT_GEN=v41).  conf_* keys must be present on the td pair
    (conf-stamp ran AFTER value-td dropped them)."""
    import torch
    import ai
    print('\n=== v42 SELF-CHECK (WOTR_ACT_GEN=v41, dim %d) ==='
          % ai.ACTION_ENCODING_DIM, flush=True)
    failures = []
    for tag in ('fp', 'sh'):
        for sfx in ('', 'td'):
            name = 'wotr_%s_corpus_%s%s.pt' % (tag, NET_TAG, sfx)
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                failures.append('%s: MISSING' % name)
                continue
            ck = torch.load(path, map_location='cpu',
                            weights_only=False)
            head_act = ck.get('head_act', 'tanh')
            adim = ck.get('action_dim')
            conf = sorted(k for k in ck if k.startswith('conf_'))
            print('[%s] head_act=%s action_dim=%s input_dim=%s\n'
                  '      trained_on=%s\n      conf keys: %s'
                  % (name, head_act, adim, ck.get('input_dim'),
                     ck.get('trained_on'), ','.join(conf) or 'NONE'),
                  flush=True)
            if head_act != 'scaled2':
                failures.append('%s: head_act=%r != scaled2 '
                                '(SPCP-3 regression - the tanh '
                                'mistake is BACK)' % (name, head_act))
            if adim != ai.ACTION_ENCODING_DIM:
                failures.append('%s: action_dim=%r != %d'
                                % (name, adim, ai.ACTION_ENCODING_DIM))
            if not conf:
                failures.append('%s: NO conf_* keys (conf-stamp did '
                                'not run after value-td)' % name)
            net = ai.load_value_net(path, side=tag)
            tail = type(net.value_head[-1]).__name__
            if tail != '_ScaledTanhHead':
                failures.append('%s: loaded head module is %s, not '
                                '_ScaledTanhHead' % (name, tail))
            print('      load_value_net OK, head module %s'
                  % tail, flush=True)
    if failures:
        for f in failures:
            print('SELF-CHECK FAIL: %s' % f, flush=True)
        raise SystemExit('v42 self-check FAILED (%d)' % len(failures))
    print('=== v42 SELF-CHECK PASS (4 checkpoints scaled2/93/conf-'
          'stamped) ===', flush=True)


def main():
    print('v42 training', flush=True)
    print('  corpus : %s' % CORPUS, flush=True)
    print('  dataset: %s' % DS, flush=True)
    print('  outputs: wotr_{fp,sh}_corpus_%s.pt + %std.pt  '
          '(live nets untouched)' % (NET_TAG, NET_TAG), flush=True)
    for stage in [s.strip() for s in STAGES.split(',') if s.strip()]:
        t0 = time.time()
        print('\n=== stage %s ===' % stage, flush=True)
        cmd = [sys.executable, '-u',
               os.path.join(ROOT, 'src', 'train_corpus.py'),
               '--stage', stage]
        if stage == 'value-td':
            cmd += ['--td-lambda', '0.8', '--td-rounds', '3']
        if stage == 'train' and os.environ.get('V42_BC_EPOCHS'):
            # pilot-only knob: a short smoke train; NEVER set on the
            # full run (the recipe default is train_corpus's 10)
            cmd += ['--epochs', os.environ['V42_BC_EPOCHS']]
        rc = subprocess.call(cmd, cwd=ROOT, env=ENV)
        print('=== stage %s rc=%d (%.1f min) ==='
              % (stage, rc, (time.time() - t0) / 60.0), flush=True)
        if rc != 0:
            print('STOPPING: stage %s failed' % stage, flush=True)
            return rc
    self_check()
    print('\nv42 TRAINING COMPLETE', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
