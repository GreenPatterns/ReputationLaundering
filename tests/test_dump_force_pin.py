"""Pin for ruling R27: every writer CLI passes its force flag to its final keeper.laundering.core.dump.

An AST scan over keeper/ finds every call of core.dump under any name a module binds to it
(`from keeper.laundering import core as C`, `import keeper.laundering.core as C`,
`from keeper.laundering.core import dump`). Each call must pass a `force` keyword whose value is not a
literal (so it carries the caller's flag), and each module that calls it must define a `--force`
option. The scan must find the ten writers of 2026-10-27, so an alias the scan misses cannot hide them;
a new writer is checked as soon as it calls dump. keeper/scem/stage_b.py writes through its own save(),
and every save() call must pass a force argument too. Nothing runs; CPU only.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEPER = ROOT / "keeper"
CORE = "keeper.laundering.core"
KNOWN_WRITERS = {
    "keeper/cppr/cp_vpe_gpu.py", "keeper/cppr/vpe_frontier.py", "keeper/federated/fed_laundering.py",
    "keeper/laundering/aggregate4.py", "keeper/laundering/calibrate_operators.py",
    "keeper/laundering/incidental.py", "keeper/laundering/probe4.py",
    "keeper/laundering/probe_baselines.py", "keeper/laundering/threshold.py",
    "keeper/laundering/verification.py",
}
STAGE_B_SAVES = 5          # engine, trace, equiv, model, aggregate


def dotted(node):
    """'a.b.c' for a Name/Attribute chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def dump_names(tree):
    """Every dotted name under which the module can call core.dump."""
    names = {f"{CORE}.dump"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "keeper.laundering":
            names |= {f"{a.asname or a.name}.dump" for a in node.names if a.name == "core"}
        elif isinstance(node, ast.ImportFrom) and node.module == CORE:
            names |= {a.asname or a.name for a in node.names if a.name == "dump"}
        elif isinstance(node, ast.Import):
            names |= {f"{a.asname}.dump" for a in node.names if a.name == CORE and a.asname}
    return names


def calls_to(tree, names):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and dotted(n.func) in names]


def has_force_option(tree):
    return any(isinstance(n, ast.Call) and (dotted(n.func) or "").endswith("add_argument")
               and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == "--force"
               for n in ast.walk(tree))


def writers():
    """{module path: (tree, dump calls)} for every keeper module that calls core.dump."""
    found = {}
    for path in sorted(KEEPER.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith("keeper/third_party/") or rel == "keeper/laundering/core.py":
            continue
        tree = ast.parse(path.read_text(), filename=rel)
        calls = calls_to(tree, dump_names(tree))
        if calls:
            found[rel] = (tree, calls)
    return found


def lacks_force_flag(call):
    force = {k.arg: k.value for k in call.keywords}.get("force")
    return force is None or isinstance(force, ast.Constant)


def test_scan_finds_every_known_writer():
    assert KNOWN_WRITERS <= set(writers())


def test_every_dump_call_passes_the_callers_force_flag():
    bad = [f"{rel}:{c.lineno}" for rel, (_, calls) in writers().items() for c in calls if lacks_force_flag(c)]
    assert not bad, f"core.dump without the caller's force flag: {bad}"


def test_every_writer_has_a_force_option():
    missing = [rel for rel, (tree, _) in writers().items() if not has_force_option(tree)]
    assert not missing, f"writers without a --force option: {missing}"


def test_the_check_rejects_a_dump_call_without_force():
    tree = ast.parse("from keeper.laundering import core as K\nK.dump(out, name, D)\nK.dump(out, name, force=True)\n")
    assert [lacks_force_flag(c) for c in calls_to(tree, dump_names(tree))] == [True, True]


def test_stage_b_save_calls_pass_force():
    tree = ast.parse((KEEPER / "scem" / "stage_b.py").read_text())
    calls = calls_to(tree, {"save"})
    assert len(calls) == STAGE_B_SAVES
    for c in calls:
        positional = len(c.args) >= 3 and not isinstance(c.args[2], ast.Constant)
        assert positional or not lacks_force_flag(c), f"stage_b.py:{c.lineno} save() without force"
