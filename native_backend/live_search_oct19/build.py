"""Strict-FP native+static mimalloc builder; production has no allocator test hooks."""
from pathlib import Path
import argparse,hashlib,json,os,platform,re,shutil,subprocess,time
HERE=Path(__file__).resolve().parent
VENDOR=HERE/'vendor/mimalloc-3.5.3'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dependencies(files):
    found=set();pending=list(files)
    while pending:
        p=pending.pop().resolve()
        if p in found:continue
        if not p.is_relative_to(HERE):raise RuntimeError('Source escaped isolated folder')
        found.add(p)
        for name in re.findall(r'^\s*#\s*include\s*"([^"\n]+)"',p.read_text(encoding='utf-8'),re.MULTILINE):
            dep=next((x for x in (p.parent/name,HERE/'include'/name) if x.is_file()),None)
            if dep is None:raise RuntimeError('Missing include: '+name)
            pending.append(dep)
    return found
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('plain','instrument','optimize'),required=True)
    parser.add_argument('--with-tests',action='store_true',help='Audit-only plain DLL; never used for production PGO')
    a=parser.parse_args()
    if os.name!='nt':raise RuntimeError('This experiment requires MSVC')
    if a.with_tests and a.stage!='plain':raise RuntimeError('Tests are excluded from production PGO')
    game=[HERE/'cpp'/n for n in ('position.cpp','snapshot.cpp','random_stream.cpp','allocator_mimalloc.cpp')]
    if a.with_tests:game.append(HERE/'audit/allocator_traits.cpp')
    files=dependencies(game)|{HERE/'build.py',HERE/'vendor/PROVENANCE.json',VENDOR/'LICENSE',VENDOR/'CMakeLists.txt'}
    # The complete vendor closure includes angle-bracket includes and all platform
    # fragments; copying manifest sources alone can reproduce the allocator build.
    files.update(p for d in ('include','src') for p in (VENDOR/d).rglob('*') if p.is_file())
    sources={str(p.relative_to(HERE)):sha(p) for p in sorted(files)}
    profile_inputs={};original_profile=None
    if a.stage=='optimize':
        trained=json.loads((HERE/'training.json').read_text())
        instrument=json.loads((HERE/'manifest_instrument.json').read_text())
        if instrument['sources']!=sources:raise RuntimeError('PGO training belongs to different sources')
        original_profile=Path(trained['profile'])
        if original_profile.resolve()!=Path(instrument['pgo_profile']).resolve():raise RuntimeError('Unexpected profile origin')
        if sha(original_profile)!=trained['profile_sha256']:raise RuntimeError('Training profile changed')
        profile_inputs={'profile_sha256':sha(original_profile),'training_sha256':sha(HERE/'training.json'),'instrument_binary_sha256':instrument['binary_sha256']}
    identity=hashlib.sha256(json.dumps({'sources':sources,'stage':a.stage,'with_tests':a.with_tests,'pgo_inputs':profile_inputs},sort_keys=True).encode()).hexdigest()[:20]
    output=HERE/'build'/identity;output.mkdir(parents=True,exist_ok=True)
    binary=output/'wotr_native.dll';record_path=output/'manifest.json';profile=output/'profile.pgd'
    manifest_name='manifest_test.json' if a.with_tests else 'manifest_'+a.stage+'.json'
    if record_path.exists():
        prev=json.loads(record_path.read_text())
        if prev['sources']!=sources or sha(binary)!=prev['binary_sha256']:raise RuntimeError('Build identity changed')
        (HERE/'build/manifest.json').write_text(json.dumps(prev,indent=2));(HERE/manifest_name).write_text(json.dumps(prev,indent=2))
        print(json.dumps({'passed':True,'reused':True,'binary':str(binary)}));return
    from setuptools._distutils._msvccompiler import _get_vc_env
    env=dict(os.environ);env.update({k.upper():v for k,v in _get_vc_env('x64').items()})
    cl=shutil.which('cl.exe',path=env['PATH']);dump=shutil.which('dumpbin.exe',path=env['PATH'])
    common=[cl,'/nologo','/std:c++17','/O2','/EHsc','/MD','/fp:strict','/bigobj']
    if a.stage!='plain':common+=['/GL']
    obj=output/'mimalloc_static.obj'
    allocator=common+['/c','/TP','/Zc:__cplusplus','/DNDEBUG','/DMI_DEBUG=0','/DMI_STAT=0','/DMI_STATIC_LIB','/I'+str(VENDOR/'include'),str(VENDOR/'src/static.c'),'/Fo'+str(obj)]
    command=common+['/LD','/DMI_STATIC_LIB','/I'+str(HERE/'include'),'/I'+str(VENDOR/'include'),*[str(p) for p in game],str(obj),'/link','/OUT:'+str(binary),'/IMPLIB:'+str(output/'wotr_native.lib'),'/MAP:'+str(output/'wotr_native.map'),'psapi.lib','shell32.lib','user32.lib','advapi32.lib','bcrypt.lib']
    if a.stage!='plain':
        if a.stage=='optimize':shutil.copy2(original_profile,profile)
        command+=['/LTCG',('/GENPROFILE:PGD=' if a.stage=='instrument' else '/USEPROFILE:PGD=')+str(profile)]
    start=time.perf_counter();logs=[]
    for cmd in (allocator,command):
        r=subprocess.run(cmd,cwd=output,env=env,capture_output=True,text=True)
        logs.append(r.stdout+r.stderr);(output/'build.log').write_text('\n'.join(logs))
        if r.returncode:raise RuntimeError('Build failed: '+str(output/'build.log'))
    imported=subprocess.run([dump,'/imports',str(binary)],env=env,capture_output=True,text=True,check=True).stdout
    exported=subprocess.run([dump,'/exports',str(binary)],env=env,capture_output=True,text=True,check=True).stdout
    (output/'imports.txt').write_text(imported);(output/'exports.txt').write_text(exported)
    assert 'mimalloc-redirect' not in imported.lower()
    if not a.with_tests:assert 'wotr_test_allocator_' not in exported
    runtime={}
    if a.stage=='instrument':
        support=Path(cl).parent/'pgort140.dll';shutil.copy2(support,output/support.name)
        runtime={'name':support.name,'sha256':sha(support)}
    record={'passed':True,'abi':1,'snapshot_abi':1,'sources':sources,'binary':str(binary),'binary_sha256':sha(binary),'command':command,'allocator_command':allocator,'compiler_sha256':sha(cl),'platform':platform.platform(),'seconds':time.perf_counter()-start,'python_headers_or_libraries':False,'allocator':'mimalloc 3.5.3 static, C++ new/delete only; no malloc override/redirect','compiler_variant':('pgo_'+a.stage if a.stage!='plain' else 'plain'),'pgo_profile':str(profile) if a.stage!='plain' else None,'pgo_inputs':profile_inputs,'instrument_runtime':runtime,'allocator_test_hooks':a.with_tests,'scope':'isolated allocator experiment; not installed'}
    record_path.write_text(json.dumps(record,indent=2));(HERE/'build/manifest.json').write_text(json.dumps(record,indent=2));(HERE/manifest_name).write_text(json.dumps(record,indent=2))
    print(json.dumps({'passed':True,'stage':a.stage,'with_tests':a.with_tests,'binary':str(binary),'seconds':record['seconds']}))
if __name__=='__main__':main()
