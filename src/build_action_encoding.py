"""Explicit portable Cython build using the same Python as the search worker."""
from pathlib import Path
import argparse,ast,json,subprocess,sys,time
import action_encoding_backend as backend

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ai-source',type=Path,default=Path(__file__).with_name('ai.py'))
    args=parser.parse_args();here=Path(__file__).resolve().parent;source=args.ai_source.resolve()
    approved=json.loads((here/'action_encoding_source.json').read_text(encoding='utf-8'))
    backend.require(approved['schema']=='wotr_action_encoding_source_v1','source schema differs')
    backend.require(approved['helper_sha256']==backend.sha(here/'search_action_encoding.py'),'Python source changed')
    backend.require(approved['pyx_sha256']==backend.sha(here/'wotr_action_encoding.pyx'),'Cython source changed')
    text=source.read_text(encoding='utf-8');lines=text.splitlines(keepends=True)
    import hashlib
    functions={n.name:hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()
               for n in ast.walk(ast.parse(text)) if isinstance(n,ast.FunctionDef) and n.name in approved['ai_functions_sha256']}
    backend.require(functions==approved['ai_functions_sha256'],'AI encoding dependencies changed; review implementation first')
    source_hash=backend.sha(source);binding_hash=backend.sha(here/'action_encoding_source.json')
    setup="""import numpy
from setuptools import setup
from Cython.Build import cythonize
setup(name='wotr_action_encoding', include_dirs=[numpy.get_include()], ext_modules=cythonize(['wotr_action_encoding.pyx'], compiler_directives={'language_level':3,'boundscheck':True,'wraparound':True,'cdivision':False}), script_args=['build_ext','--inplace','--build-temp','build_action_encoding_temp'])
"""
    start=time.perf_counter();result=subprocess.run([sys.executable,'-c',setup],cwd=here,capture_output=True)
    (here/'action_encoding_build.stdout.log').write_bytes(result.stdout)
    (here/'action_encoding_build.stderr.log').write_bytes(result.stderr)
    backend.require(result.returncode==0,'compiler failed; see build stderr log')
    backend.require(source_hash==backend.sha(source) and binding_hash==backend.sha(here/'action_encoding_source.json'),'inputs changed during build')
    backend.require(approved['helper_sha256']==backend.sha(here/'search_action_encoding.py') and approved['pyx_sha256']==backend.sha(here/'wotr_action_encoding.pyx'),'implementations changed during build')
    binary=here/('wotr_action_encoding'+backend.abi()['extension_suffix'])
    import Cython
    manifest=dict(schema='wotr_action_encoding_build_v1',abi=backend.abi(),binary=binary.name,binary_sha256=backend.sha(binary),
                  ai_source_sha256=source_hash,ai_functions_sha256=functions,source_manifest_sha256=binding_hash,
                  build_script_sha256=backend.sha(__file__),cython_version=Cython.__version__,build_seconds=time.perf_counter()-start)
    (here/'action_encoding_build.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(manifest))
if __name__=='__main__':main()
