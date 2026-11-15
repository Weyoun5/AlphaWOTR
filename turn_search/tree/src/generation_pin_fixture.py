"""Test-only: run one test module under a pinned WOTR_ACT_GEN without leaking it into the process.

ai.py reads WOTR_ACT_GEN ONCE, at import (selfplay_actor_critic/contracts.py does the same). A test module
that sets os.environ at import time and imports ai there poisons every other test module loaded into the same process
(`python -m unittest a b`, discover): their lazy `import ai` returns the pinned generation, and the env update itself
outlives the module. R3-TEST-ISOLATION repro: `python -m unittest test_audit0905_action_contract
test_audit0905_companion_warmstart` gave 7F/5E in action_contract, because companion_warmstart imported ai under v46 at
load time, before any test ran.

GenerationPin.enter() (call it from setUpModule):
  * snapshots os.environ, sys.path, the working directory (board.py chdirs at import) and every PROJECT source module
    in sys.modules (a .py file under the repo root, absolute path);
  * removes those project modules from sys.modules, so the module's own imports re-execute them under the pin;
  * applies the pinned environment (a value of None unsets the variable).
GenerationPin.exit() (call it from tearDownModule; idempotent):
  * removes every project source module imported under the pin and puts the snapshot entries back (the same module
    objects other test modules already hold);
  * restores os.environ, sys.path and the working directory.

Never touched, by design:
  * third-party modules (torch, numpy, sympy, ...): dropping torch._dynamo after it wrapped torch.manual_seed makes the
    next manual_seed re-register TORCH_LIBRARY _inductor_test (R2-TR-test-round17-credit-warmstart). torch.ops and
    torch.classes carry RELATIVE fake __file__ values, which the absolute-path test excludes;
  * test modules (any module whose last name component starts with 'test_') and this helper: unittest finds a test
    module's tearDownModule through sys.modules, so dropping one would skip its teardown;
  * extension modules (.pyd/.so): they cannot be re-initialised in one process.
"""
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parent.parent


def is_project_source(name, module, root=REPO):
    """True for a project .py module that a generation pin must re-import (see the module docstring)."""
    if module is None or name in ('__main__', __name__):
        return False
    if name.rsplit('.', 1)[-1].startswith('test_'):
        return False
    path = getattr(module, '__file__', None)
    if not isinstance(path, str) or not path.endswith('.py') or not os.path.isabs(path):
        return False
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except (ValueError, OSError):
        return False
    return True


class GenerationPin:
    """Pin WOTR_ACT_GEN (None = unset) plus any other env vars for one test module; see the module docstring."""

    def __init__(self, generation, *, root=REPO, **env):
        self.env = dict(WOTR_ACT_GEN=generation, **env)
        self.root = Path(root)
        self._snapshot = None

    @property
    def active(self):
        return self._snapshot is not None

    def _project_modules(self):
        return [name for name, module in list(sys.modules.items()) if is_project_source(name, module, self.root)]

    def enter(self):
        if self.active:
            raise RuntimeError('GenerationPin.enter() called twice without exit()')
        saved_modules = {name: sys.modules[name] for name in self._project_modules()}
        self._snapshot = (dict(os.environ), list(sys.path), os.getcwd(), saved_modules)
        for name in saved_modules:
            del sys.modules[name]
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        return self

    def exit(self):
        if not self.active:
            return
        environ, path, cwd, saved_modules = self._snapshot
        self._snapshot = None
        for name in self._project_modules():
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = path
        if dict(os.environ) != environ:
            os.environ.clear()
            os.environ.update(environ)
        os.chdir(cwd)
