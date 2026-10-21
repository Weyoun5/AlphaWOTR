"""v42 FULL re-extraction: the whole corpus replayed through TODAY's
engine (2026-08-29 fix waves: amendments 1.251-1.257 legal surface,
GL-5 hunt-sacrifice menu, 1.254 casualty pendings, EX-8 payment
pending, 1.230 in-place declare window, combat kernel alive).

GENERATION vs VINTAGE (do not confuse the two):
  * WOTR_ACT_GEN=v41 is the ENCODING generation and stays v41 -
    ACTION_ENCODING_DIM 93 with the 1.149 table-kill tail bit.
  * v42 is the CORPUS vintage: same encoding, new engine semantics.

Writes to data/bc_corpus_v42_rendered - a NEW directory.  v41 (the
corpus name the USER's current live nets were trained on - the
directory itself is gone, SPCP-1) is never touched, so live play keeps
working on the current nets throughout.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get('RX_OUT_DIR') or os.path.join(
    ROOT, 'data', 'bc_corpus_v42_rendered')
os.environ['RX_OUT_DIR'] = OUT
os.environ['WOTR_ACT_GEN'] = 'v41'
# Belt (SPCP-3): extraction builds no nets, but the v40 chain pinned
# this globally and any child stage inheriting this env must never
# construct a Tanh-head net.
os.environ['WOTR_VALUE_HEAD_ACT'] = 'scaled2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
sys.path.insert(0, os.path.join(ROOT, 'src'))

import rendered_extract as rx  # noqa: E402

if __name__ == '__main__':
    w = int(os.environ.get('V42_WORKERS', '5'))
    print('v42 full extraction -> %s (workers=%d)' % (OUT, w), flush=True)
    rx.run_driver(w, source='corpus')
    print('v42 extraction COMPLETE', flush=True)
