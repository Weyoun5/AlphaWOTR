"""Load the native own-value controller once for a live bridge process.

The Java bridge retains its Python game/gesture driver. Only speculative
search uses C++ snapshots; the root player's checkpoint is its sole model.
Source, Python native boundary and binary identities are checked at startup,
never inside simulations.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
from types import ModuleType

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
    # A development pin may also preserve non-runtime tests/build witnesses.
    # Its separately recorded runtime projection contains every shipped engine
    # dependency; changing a test cannot invalidate an unchanged game runtime.
    source_hashes = pin.get('runtime_source_sha256', pin.get('source_sha256', {}))
    if not source_hashes.get('ai.py'):
        raise RuntimeError('Live native reference has no engine identity')
    if any(pin.get('source_sha256', {}).get(name) != digest
           for name,digest in source_hashes.items()):
        raise RuntimeError('Live native runtime projection disagrees with build reference')
    for folder,hashes in ((source,source_hashes),(runtime/'data',pin['data_sha256'])):
        for name,expected in hashes.items():
            actual=hashlib.sha256((folder/name).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError('Live native reference source drift: '+name)
    return {'reference_path':str(reference),
            'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
            'reference_runtime':str(runtime),
            'reference_ai_sha256':pin['source_sha256']['ai.py']}


def boundary_receipt(modules):
    """Hash every native_backend Python file among `modules` (Q1-06).

    The frozen box DLL is driven by the shared tree's Python marshalling layer
    (snapshot/schema encoding, observation and action tables, ctypes bindings,
    numeric replies). Neither the pin's src/data hashes nor the box manifest
    (C++ sources) covers those files, so the startup receipt records the
    identity of each one this process actually executed; load() compares it
    with the pin's native_boundary_sha256 (X2-01).
    """
    files = {}
    for module in modules:
        if not isinstance(module, ModuleType):
            continue
        origin = getattr(module, '__file__', None)
        if not isinstance(origin, str):
            continue
        path = Path(origin).resolve()
        if path.is_relative_to(BACKEND):
            files[path.relative_to(BACKEND).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
    return files, digest


def pinned_boundary(reference):
    """Verify the pinned native boundary on disk BEFORE any of it executes (X2-01).

    The reference pin's native_boundary_sha256 maps native_backend-relative
    paths to digests (written by native_backend/pin_current_runtime.py). A pin
    without it cannot certify the Python half of the native interface and is
    refused; any pinned file that changed is refused before it is imported.
    """
    pin = json.loads(Path(reference).read_text(encoding='utf-8'))
    pinned = pin.get('native_boundary_sha256')
    if not isinstance(pinned, dict) or not pinned:
        raise RuntimeError('Live native reference has no native boundary identity '
                           '(native_boundary_sha256); re-pin with native_backend/pin_current_runtime.py')
    backend = BACKEND.resolve()
    for name, expected in sorted(pinned.items()):
        path = (backend / name).resolve()
        if not path.is_relative_to(backend):
            raise RuntimeError('Live native boundary pin escapes native_backend: ' + name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Live native boundary drift: ' + name)
    return dict(pinned)


# R6-NATIVE-TEST-HARNESS: native_backend/tests is the native test suite. Its modules
# (a unittest driver run as __main__ or imported, and the helpers it imports, such as
# test_snapshot.compare_graph) are the CALLER of load(), never the Python half of the
# native interface: no live entry point imports them and the pin tool's runtime
# projection never contains them. A driver living under wotr_bridge/tests is outside
# BACKEND and never counted; one under native_backend/tests was refused as an unpinned
# boundary file, so test_bridge_combat_metadata / test_prepaid_army_pair could not load.
NATIVE_TEST_TREE = 'tests/'


def verify_executed_boundary(files, pinned):
    """Every boundary file this process executed must be pinned, unchanged (X2-01).

    An unpinned file of the native test tree (NATIVE_TEST_TREE) is accepted: it is the
    test driver, not the boundary. It stays in the receipt, and a pinned file there
    is still held to its digest."""
    for name, digest in sorted(files.items()):
        if name not in pinned:
            if name.startswith(NATIVE_TEST_TREE):
                continue
            raise RuntimeError('Live native boundary file is not pinned: ' + name)
        if digest != pinned[name]:
            raise RuntimeError('Live native boundary drift: ' + name)


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
        boundary_pin = pinned_boundary(reference)  # before any boundary code runs (X2-01)
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
        # Resources load helpers by file path (forcing, combat facts, search
        # metadata); those modules never enter sys.modules, so record them here.
        file_loaded = []
        load_file = native._load

        def recorded_load(name, relative):
            module = load_file(name, relative)
            file_loaded.append(module)
            return module

        native._load = recorded_load
        try:
            native.resources(ai)  # Initialize immutable tables before the first move.
        finally:
            native._load = load_file
        boundary, boundary_digest = boundary_receipt(
            tuple(sys.modules.copy().values()) + tuple(file_loaded))
        verify_executed_boundary(boundary, boundary_pin)
        manifest = json.loads((box / 'build/manifest.json').read_text(encoding='utf-8'))
        identity.update(backend='native', controller_box=str(box),
                        controller_binary_sha256=manifest['binary_sha256'],
                        orchestrator_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        boundary_sha256=boundary,
                        boundary_identity_sha256=boundary_digest,
                        opponent_model_loaded=False)
        _loaded = ai, box, reference, native, identity
        return native, dict(identity)
