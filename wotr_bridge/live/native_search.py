"""Load the native own-value controller once for a live bridge process.

The Java bridge retains its Python game/gesture driver. Only speculative
search uses C++ snapshots; the root player's checkpoint is its sole model.
Source and binary identities are checked at startup, never inside simulations.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / 'native_backend'
CORE = BACKEND / 'experiments/native_search_core'
OWN = CORE / 'own_value_controller_sep13'
DEFAULT_BOX = BACKEND / 'live_search_oct19'
_loaded = None
_lock = threading.Lock()


def verify_reference(ai, reference):
    """Validate a relocatable release pin against this process's Python engine."""
    pin=json.loads(reference.read_text(encoding='utf-8'))
    runtime=Path(pin['runtime'])
    if not runtime.is_absolute():
        runtime=(reference.parent/runtime).resolve()
    source=runtime/'src'
    if Path(ai.__file__).resolve() != source/'ai.py':
        raise RuntimeError('Live native reference selects a different Python engine')
    if not pin.get('source_sha256',{}).get('ai.py'):
        raise RuntimeError('Live native reference has no engine identity')
    for folder,key in ((source,'source_sha256'),(runtime/'data','data_sha256')):
        for name,expected in pin[key].items():
            actual=hashlib.sha256((folder/name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Live native reference source drift: '+name)
    return {'reference_path':str(reference),
            'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'reference_runtime':str(runtime),
            'reference_ai_sha256':pin['source_sha256']['ai.py']}


def load(ai):
    """Return the verified controller and its startup receipt; no fallback."""
    global _loaded
    box = Path(os.environ.get('WOTR_LIVE_NATIVE_BOX', DEFAULT_BOX)).resolve()
    reference = Path(os.environ.get('WOTR_LIVE_NATIVE_REFERENCE',
                                   box / 'reference_live.json')).resolve()
    with _lock:
        if _loaded is not None:
            old_ai, old_box, old_reference, native, identity = _loaded
            if old_ai is not ai or old_box != box or old_reference != reference:
                raise RuntimeError('Live native engine selection changed within one process')
            return native, dict(identity)
        identity = verify_reference(ai, reference)
        for path in (BACKEND, CORE / 'inference'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import bridge
        import controller_binding
        # The source check above certifies the already imported live engine.
        # Native table builders must use that instance, without re-importing it
        # or replacing the live RNG/environment with a self-play configuration.
        bridge.load_reference = lambda: ai
        library = controller_binding.candidate_library(box)
        path = OWN / 'orchestration/orchestrator.py'
        spec = importlib.util.spec_from_file_location('live_own_native_orchestrator', path)
        native = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = native
        spec.loader.exec_module(native)
        import own_binding
        own_binding.configure(library)
        capabilities = controller_binding.capabilities(library)
        for name in ('root_priors_exchange', 'observer_events', 'combat_dist'):
            if not capabilities.get(name):
                raise RuntimeError('Live native controller lacks ' + name)
        native._decorated(ai)
        native.resources(ai)  # Initialize immutable tables before the first move.
        manifest = json.loads((box / 'build/manifest.json').read_text(encoding='utf-8'))
        identity.update(backend='native', controller_box=str(box),
                        controller_binary_sha256=manifest['binary_sha256'],
                        orchestrator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        opponent_model_loaded=False)
        _loaded = ai, box, reference, native, identity
        return native, dict(identity)
