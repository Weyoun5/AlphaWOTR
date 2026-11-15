"""Explicit local build; requires installed Cython/setuptools and a C compiler.

Run with the same Python used for self-play. Never invoked by the backend.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import subprocess
import sys
import time
import policy_keys_backend as backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-source', type=Path, default=Path(__file__).with_name('ai_policies.py'))
    args = parser.parse_args()
    source = args.policy_source.resolve()
    here = Path(__file__).resolve().parent
    if backend.sha(source) != backend.SOURCE_SHA256:
        raise RuntimeError('Policy source does not match this native implementation')
    if backend.sha(here / 'wotr_policy_keys.pyx') != backend.PYX_SHA256:
        raise RuntimeError('Cython source changed; review/regenerate source bindings first')
    text = source.read_text(encoding='utf-8')
    lines = text.splitlines(keepends=True)
    functions = {n.name: hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()
                 for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name in backend.FUNCTIONS_SHA256}
    if functions != backend.FUNCTIONS_SHA256:
        raise RuntimeError('Policy function bindings differ')
    setup = """from setuptools import setup
from Cython.Build import cythonize
setup(name='wotr_policy_keys', ext_modules=cythonize(['wotr_policy_keys.pyx'], compiler_directives={'language_level':3,'annotation_typing':False,'infer_types':False,'boundscheck':True,'wraparound':True,'cdivision':False}), script_args=['build_ext','--inplace','--build-temp','build_policy_keys_temp'])
"""
    started = time.perf_counter()
    result = subprocess.run([sys.executable, '-c', setup], cwd=here, capture_output=True, text=True)
    (here / 'policy_keys_build.stdout.log').write_text(result.stdout, encoding='utf-8')
    (here / 'policy_keys_build.stderr.log').write_text(result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Local compiler failed; see policy_keys_build.stderr.log')
    import Cython
    binary = here / ('wotr_policy_keys' + backend.abi()['extension_suffix'])
    manifest = dict(schema='wotr_policy_keys_build_v1', abi=backend.abi(),
                    binary=binary.name, binary_sha256=backend.sha(binary),
                    source_sha256=backend.sha(source), functions_sha256=functions,
                    pyx_sha256=backend.sha(here / 'wotr_policy_keys.pyx'),
                    build_script_sha256=backend.sha(__file__), cython_version=Cython.__version__,
                    python=sys.version, build_seconds=time.perf_counter()-started)
    (here / 'policy_keys_build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
