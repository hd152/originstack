"""Every process pool must use the ``spawn`` context (src.utils.mp_context).

With Linux's default ``fork``, a worker copied from a parent that had already started the
Rust kernels' thread pool inherited its state but not its threads and hung on its first
parallel call: a packaged Linux build sat in Phase 1 for two hours with four idle workers.
Windows only has spawn, so no Windows run can catch this; guard it at the source.
"""
import ast
import pathlib

from src.utils import mp_context

SRC = pathlib.Path(__file__).resolve().parent.parent / 'src'


def test_context_is_spawn():
    assert mp_context().get_start_method() == 'spawn'


def test_every_process_pool_in_src_passes_the_spawn_context():
    missing = []
    for path in sorted(SRC.glob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.Call) and getattr(node.func, 'id', getattr(node.func, 'attr', None))
                    == 'ProcessPoolExecutor'):
                kw = {k.arg: k.value for k in node.keywords}
                ok = (isinstance(kw.get('mp_context'), ast.Call)
                      and getattr(kw['mp_context'].func, 'id', None) == 'mp_context')
                if not ok:
                    missing.append(f'{path.name}:{node.lineno}')
    assert not missing, f'ProcessPoolExecutor without mp_context=mp_context(): {missing}'
