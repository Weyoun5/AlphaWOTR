"""Optional semantic key acceleration. Activate once before starting search threads.

WOTR_POLICY_KEYS=python (default) needs no extension. Explicit native requests
require an individually built, source-bound binary; no runtime compilation.
"""
from pathlib import Path
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import sys
import sysconfig

HERE = Path(__file__).resolve().parent
# R3-TOOLS-MISC (R2-TR-NATIVE-EXTENSIONS design): the extension is bound to the two functions it replaces, not to the
# whole ai_policies.py file (a whole-file SOURCE_SHA256 pin disabled native activation on every unrelated edit).  Both
# functions are self-contained (_freeze recurses into itself only), so their source pins cover what the copies replace;
# the .pyx carries the same FUNCTIONS_SHA256 table and its bind() re-checks it.
PYX_SHA256 = '93727136445009afa758fedf1907159a7a7e21261f482db9dc6df30c2eef118d'
FUNCTIONS_SHA256 = {
    '_freeze': 'd99782f2e2adb537474e0ac88446eb6c051fe05d43a451bbdd9a1ceffb84695a',
    'action_policy_key': 'cd2d399815b6a989a65436a933fa0ff2acd3f6e82520c7e548c4bcc43515dd50',
}
_activated = None


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def abi():
    return dict(implementation=sys.implementation.name,
                cache_tag=sys.implementation.cache_tag,
                python_version=list(sys.version_info[:3]),
                system=platform.system(), machine=platform.machine(),
                platform=sysconfig.get_platform(),
                soabi=sysconfig.get_config_var('SOABI'),
                extension_suffix=sysconfig.get_config_var('EXT_SUFFIX'))


def _require(ok, reason):
    if not ok:
        raise RuntimeError('Native policy keys unavailable: ' + reason)


def activate(policy):
    """Select explicit backend and return JSON-safe startup diagnostics.

No model, action, key or RNG is touched. A failed validation does not replace
any policy function. Call before creating search threads; not a hot-switch API.
"""
    global _activated
    mode = os.environ.get('WOTR_POLICY_KEYS', 'python').strip().lower()
    _require(mode in ('python', 'native'), 'WOTR_POLICY_KEYS must be python or native')
    if mode == 'python':
        _require(_activated is None, 'cannot switch an activated process to Python')
        return dict(backend='python', requested=mode, activated=False)
    try:
        manifest = json.loads((HERE / 'policy_keys_build.json').read_text(encoding='utf-8'))
        _require(manifest['schema'] == 'wotr_policy_keys_build_v1', 'build schema mismatch')
        _require(manifest['abi'] == abi(), 'ABI/platform differs; rebuild on this host')
        # manifest['source_sha256'] (the whole policy file at build time) is provenance only; the binding is by function.
        _require(manifest['functions_sha256'] == FUNCTIONS_SHA256, 'function binding changed')
        _require(manifest['pyx_sha256'] == PYX_SHA256 == sha(HERE / 'wotr_policy_keys.pyx'), 'Cython source changed')
        filename = 'wotr_policy_keys' + abi()['extension_suffix']
        _require(manifest['binary'] == filename, 'binary filename mismatch')
        binary = HERE / filename
        _require(sha(binary) == manifest['binary_sha256'], 'binary hash mismatch')
        if _activated is not None:
            prior_policy, native, diagnostics = _activated
            _require(policy is prior_policy, 'backend is bound to another policy module')
            _require(all(getattr(policy, n) is getattr(native, n) for n in FUNCTIONS_SHA256), 'activated functions were replaced')
            return dict(diagnostics)
        for name, expected in FUNCTIONS_SHA256.items():
            _require(hashlib.sha256(inspect.getsource(getattr(policy, name)).encode()).hexdigest() == expected,
                     'runtime function changed: ' + name)
        loaded = sys.modules.get('wotr_policy_keys')
        _require(loaded is None, 'extension was loaded outside this backend')
        spec = importlib.util.spec_from_file_location('wotr_policy_keys', binary)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        native.bind(policy)
        sys.modules['wotr_policy_keys'] = native
        policy._freeze = native._freeze
        policy.action_policy_key = native.action_policy_key
        diagnostics = dict(backend='native', requested=mode, activated=True,
                           binary=str(binary), binary_sha256=manifest['binary_sha256'],
                           policy_source_sha256=sha(policy.__file__), functions_sha256=dict(FUNCTIONS_SHA256),
                           pyx_sha256=PYX_SHA256, abi=abi())
        _activated = (policy, native, diagnostics)
        return dict(diagnostics)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        raise RuntimeError('Native policy keys unavailable: ' + str(exc)) from exc
