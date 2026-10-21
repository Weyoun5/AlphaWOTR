"""Generate Cython mirrors from the actual frozen Python module sources."""

from pathlib import Path
import ast
import hashlib
import inspect
import json
import sys
import textwrap
import symtable

REQUIRED = (
    "clone_state",
    "encode_state",
    "_state_sig",
    "encode_action",
    "_action_cache_key",
)
AUXILIARY = (
    "_side_of_nation",
    "_armies_lookup",
    "_encode_action_cached",
    "_region_index_cached",
    "_get_action_region_index",
)
ALIASES = {
    "state_sig": "_state_sig",
    "action_cache_key": "_action_cache_key",
    "side_of_nation": "_side_of_nation",
    "armies_lookup": "_armies_lookup",
    "encode_action_cached": "_encode_action_cached",
    "region_index_cached": "_region_index_cached",
    "get_action_region_index": "_get_action_region_index",
}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class Rewrite(ast.NodeTransformer):
    def __init__(self, names, owner):
        self.names = names
        self.owner = owner

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id in self.names:
            return ast.copy_location(
                ast.Subscript(
                    value=ast.Name(id=self.owner, ctx=ast.Load()),
                    slice=ast.Constant(node.id),
                    ctx=ast.Load(),
                ),
                node,
            )
        return node

    def visit_FunctionDef(self, node):
        node.returns = None
        for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
            a.annotation = None
        if node.args.vararg:
            node.args.vararg.annotation = None
        if node.args.kwarg:
            node.args.kwarg.annotation = None
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is None:
            return None
        return ast.copy_location(
            ast.Assign(targets=[node.target], value=self.visit(node.value)), node
        )


def child(out):
    src = out / "engine/src"
    sys.path.insert(0, str(src))
    import ai
    import board

    if (
        Path(ai.__file__).resolve().parent != src
        or Path(board.__file__).resolve().parent != src
    ):
        raise RuntimeError("generator imported outside the frozen source snapshot")
    originals = {
        name: inspect.getsource(getattr(ai, name)) for name in REQUIRED + AUXILIARY
    }
    originals["sanitize"] = inspect.getsource(board.sanitize)
    bodies = []
    dependencies = {}
    for name, source in originals.items():
        module = board if name == "sanitize" else ai
        tree = ast.parse(textwrap.dedent(source))
        fn = tree.body[0]
        if not (
            isinstance(fn, ast.FunctionDef)
            and fn.name == name
            and not fn.decorator_list
        ):
            raise ValueError(f"unsupported function definition for {name}")
        # Resolve globals through Python's own scope analysis. Current mirrors have
        # nested helper functions/comprehensions but do not shadow their module
        # dependency names. Refuse future shadowing rather than rewrite a local.
        table = symtable.symtable(textwrap.dedent(source), name, "exec").get_children()[
            0
        ]
        tables = []

        def collect(t):
            tables.append(t)
            for c in t.get_children():
                collect(c)

        collect(table)
        names = {
            s.get_name()
            for t in tables
            for s in t.get_symbols()
            if s.is_global() and s.is_referenced() and s.get_name() in vars(module)
        }
        locals_ = {
            s.get_name() for t in tables for s in t.get_symbols() if s.is_local()
        }
        if names & locals_:
            raise ValueError(
                f"unsupported dependency shadowing in {name}: {names & locals_}"
            )
        # No global writes occur in the selected routines; keep this fail-closed.
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(fn)):
            raise ValueError(f"unsupported global/nonlocal writes in {name}")
        dependencies[name] = sorted(names)
        tree = Rewrite(
            names, "_BOARD_GLOBALS" if name == "sanitize" else "_AI_GLOBALS"
        ).visit(tree)
        if name == "sanitize":
            # board switches sanitization during its own import, before init_ai.
            tree.body[0].body[1:1] = ast.parse(
                """global _BOARD_GLOBALS
if not _BOARD_GLOBALS:
    import board as source_board
    _BOARD_GLOBALS = vars(source_board)
"""
            ).body
        ast.fix_missing_locations(tree)
        bodies.append(ast.unparse(tree))
    full = {name: sha(src / name) for name in ("ai.py", "board.py")}
    fp = hashlib.sha1("".join(originals[n] for n in REQUIRED).encode()).hexdigest()
    header = (
        """# cython: language_level=3, annotation_typing=False, infer_types=False
# cython: boundscheck=True, wraparound=True, cdivision=False
import sys
_AI_GLOBALS = {}
_BOARD_GLOBALS = vars(sys.modules['board']) if 'board' in sys.modules else {}
_INITED = False
PURE_ACCELERATORS = ()
"""
        + f"MIRROR_FINGERPRINT={fp!r}\nSOURCE_SHA256={full!r}\n"
    )
    header += """
def _check_sources(source_dir):
    import hashlib
    from pathlib import Path
    for name, wanted in SOURCE_SHA256.items():
        path = Path(source_dir) / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise RuntimeError('Generated native source mismatch: ' + name)
from pathlib import Path as _NativePath
_check_sources(_NativePath(__file__).resolve().parent)
"""
    tail = (
        "\n".join(alias + "=" + name for alias, name in ALIASES.items())
        + """

def init_ai(ai_module):
    import hashlib
    from pathlib import Path
    import board
    for name, wanted in SOURCE_SHA256.items():
        path = Path(ai_module.__file__).resolve().parent / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
            raise RuntimeError('Generated native source mismatch: ' + name)
    global _AI_GLOBALS, _BOARD_GLOBALS, _INITED
    _AI_GLOBALS = vars(ai_module)
    _BOARD_GLOBALS = vars(board)
    _INITED = True

def init_state_encoder(ai_module):
    if not _INITED:
        raise RuntimeError('init_ai required')

def init_clone(ai_module):
    if not _INITED:
        raise RuntimeError('init_ai required')
"""
    )
    pyx = src / "wotr_native.pyx"
    pyx.write_text(header + "\n\n".join(bodies) + "\n\n" + tail, encoding="utf-8")
    (src / "setup_generated.py").write_text(
        """from setuptools import setup
from Cython.Build import cythonize

setup(
    name="wotr_native_generated_private",
    ext_modules=cythonize(
        ["wotr_native.pyx"],
        compiler_directives={
            "language_level": 3,
            "annotation_typing": False,
            "infer_types": False,
            "boundscheck": True,
            "wraparound": True,
            "cdivision": False,
        },
        build_dir="generated_build",
    ),
    script_args=["build_ext", "--inplace", "--build-temp", "generated_build"],
)
"""
    )
    manifest = {
        "source_sha256": full,
        "mirror_fingerprint": fp,
        "original_function_source": originals,
        "dynamic_module_dependencies": dependencies,
        "generated_pyx_sha256": sha(pyx),
        "generator_sha256": sha(Path(__file__).resolve()),
        "source_imports_from": str(src),
    }
    (out / "generation.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    child(Path(sys.argv[1]).resolve())
