"""Sequential evaluation bookkeeping shared by isolated reference helpers."""
from contextlib import contextmanager
import random


@contextmanager
def preserved_evaluation_context(ai):
    """Preserve caller confidence state and process RNG in a sequential worker."""
    names=('_LAST_LEAF_CONF','_LAST_LEAF_PHI','_CONF_ROOT_C','_CONF_MEAN')
    saved={name:getattr(ai,name) for name in names}
    mean_contents=list(saved['_CONF_MEAN'])
    py=random.getstate();numpy=ai.np.random.get_state()
    try:
        yield
    finally:
        saved['_CONF_MEAN'][:]=mean_contents
        for name,value in saved.items():setattr(ai,name,value)
        random.setstate(py);ai.np.random.set_state(numpy)
