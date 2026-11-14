"""The shared credibility module keeper.credibility.tiers: the legacy tiers, moved verbatim (ruling R6).

legacy_levels is keeper.cppr.cp_vpe_gpu.compute_credibility_levels moved without change, and legacy_bfs is
its seeded variant. The original is laundering/cp_vpe_gpu.py at tag pre-revision-2026-10-26, read with
`git show` into pytest's temporary directory. Only its compute_credibility_levels function is executed,
in a namespace that provides np and keeper.laundering.signed as S, because the whole module imports the
pre-move package layout and a GPU stack.

1. Source: legacy_levels equals the original function apart from its name.
2. Behaviour: on the training graphs of Bitcoin-Alpha, Bitcoin-OTC and Wiki-RfA, legacy_bfs and the
   unseeded variant equal the original function exactly.
3. Old names: cp_vpe_gpu imports the moved function and weights back, so compute_credibility_levels is
   legacy_levels and CRED_WEIGHTS is TIER_WEIGHTS; tiers.py imports nothing from keeper.cppr (no cycle).
4. Hand cases pin the legacy semantics: seeds from snapshots 0 to 4, one forward pass over the edges in
   their stored order, Tier 0, and the terciles. CPU only.

The BTP tiers of Task 3.1 are tested in tests/test_credibility_btp.py; both files share
tests/credibility_common.py and the training_graphs fixture of tests/conftest.py.
"""
import ast
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from keeper.credibility.tiers import TIER_WEIGHTS, legacy_bfs, legacy_levels
from keeper.laundering import signed as S
from tests.credibility_common import NETWORKS, RING, graph

ROOT = Path(__file__).resolve().parents[1]
TIERS = ROOT / "keeper" / "credibility" / "tiers.py"
ORIGINAL = "pre-revision-2026-10-26:laundering/cp_vpe_gpu.py"
ORIGINAL_NAME = "compute_credibility_levels"


def original_source() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "show", ORIGINAL],
                              check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError):   # no git, a shallow clone or a missing tag
        pytest.fail("tag pre-revision-2026-10-26 not found; fetch tags")


def function_node(source: str, name: str) -> ast.FunctionDef:
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, f"expected one top-level function {name}, found {len(nodes)}"
    return nodes[0]


@pytest.fixture(scope="module")
def original_function(tmp_path_factory):
    """The ORIGINAL compute_credibility_levels, executed alone from the tagged file."""
    path = tmp_path_factory.mktemp("original") / "cp_vpe_gpu.py"
    path.write_text(original_source())
    text = path.read_text()
    source = ast.get_source_segment(text, function_node(text, ORIGINAL_NAME))
    namespace = {"np": np, "S": S}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[ORIGINAL_NAME]


# ------------------------------------------------------------------ 1. source
def test_legacy_levels_is_the_original_function_verbatim():
    original = function_node(original_source(), ORIGINAL_NAME)
    original.name = "legacy_levels"
    assert ast.dump(function_node(TIERS.read_text(), "legacy_levels")) == ast.dump(original)


def test_legacy_bfs_is_the_seeded_variant():
    call = [n for n in ast.walk(function_node(TIERS.read_text(), "legacy_bfs")) if isinstance(n, ast.Call)]
    assert len(call) == 1 and ast.unparse(call[0]) == "legacy_levels(Tr, n_nodes, seeded=True)"


# ------------------------------------------------------------------ 2. behaviour on the three networks
@pytest.mark.parametrize("ds", NETWORKS)
def test_moved_tiers_equal_the_original_function(ds, original_function, training_graphs):
    Tr, n = training_graphs[ds]
    for seeded, got in ((True, legacy_bfs(Tr, n)), (False, legacy_levels(Tr, n, seeded=False))):
        want = original_function(Tr, n, seeded=seeded)
        assert got.dtype == want.dtype and got.shape == want.shape == (n,)
        assert np.array_equal(got, want), f"{ds}, seeded={seeded}: tiers differ from the original"
    assert np.array_equal(legacy_levels(Tr, n), legacy_bfs(Tr, n))      # seeded=True is the default


# ------------------------------------------------------------------ 3. old names, no cycle
def test_cp_vpe_gpu_keeps_its_old_names_as_imports():
    from keeper.cppr import cp_vpe_gpu as G
    assert G.compute_credibility_levels is legacy_levels
    assert G.CRED_WEIGHTS is TIER_WEIGHTS
    assert G.CPVPEConfig().cred_weights is TIER_WEIGHTS


def test_tier_weights_are_the_legacy_values():
    assert TIER_WEIGHTS == (0.01, 0.40, 0.70, 1.00)


def test_tiers_module_does_not_load_keeper_cppr():
    probe = ("import sys, keeper.credibility.tiers; "
             "print(sorted(m for m in sys.modules if m.startswith('keeper.cppr')))")
    res = subprocess.run([sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "[]"


# ------------------------------------------------------------------ 4. hand cases of the legacy semantics
def test_one_positive_rating_from_an_endorsed_account_admits_a_sybil_ring():
    # Seeded: endorsed senders only. pos_in = 2 (node 4), 1 (nodes 5, 6); score (p + 1)/(p + 2) * log(1 + p)
    # is 0.75 log 3 for node 4 and (2/3) log 2 for nodes 5 and 6, so 5 and 6 sit at the lower tercile
    # bound (Tier 1) and 4 above the upper one (Tier 3). Node 0, a seed, has no endorsed positive in-edge.
    assert legacy_bfs(graph(RING), 8).tolist() == [0, 0, 0, 0, 3, 1, 1, 0]


def test_unseeded_variant_counts_every_positive_rating():
    # Unseeded: node 0 (pos 2) and node 7 (pos 1, from node 3) are endorsed too; node 3 has pos 1 = neg 1.
    assert legacy_levels(graph(RING), 8, seeded=False).tolist() == [3, 0, 0, 0, 3, 1, 1, 1]


def test_endorsement_is_one_pass_in_the_stored_edge_order():
    # The same edges, with 4 -> 5 stored before 0 -> 4 in snapshot 5: when 4 -> 5 is read, node 4 is not yet
    # endorsed, so 5 is not admitted, and 5 -> 6 and 6 -> 4 then admit nothing. Node 5 still counts the
    # positive rating of node 4, which is endorsed by the end of the pass, so 4 and 5 (one endorsed positive
    # in-edge each) share Tier 1; node 6 drops to Tier 0.
    reordered = [(1, 0, +1, 0), (2, 0, +1, 1), (0, 3, -1, 2),
                 (4, 5, +1, 5), (0, 4, +1, 5), (5, 6, +1, 6), (6, 4, +1, 6), (7, 3, +1, 7), (3, 7, +1, 8)]
    assert legacy_bfs(graph(reordered), 8).tolist() == [0, 0, 0, 0, 1, 1, 0, 0]


def test_seeds_use_snapshots_zero_to_four_only():
    # Node 0's two positive in-edges arrive in snapshot 5: it is no seed, so nothing is endorsed.
    late = [(1, 0, +1, 5), (2, 0, +1, 5), (0, 4, +1, 6)]
    assert legacy_bfs(graph(late), 5).tolist() == [0, 0, 0, 0, 0]
    early = [(1, 0, +1, 4), (2, 0, +1, 4), (0, 4, +1, 6)]
    assert legacy_bfs(graph(early), 5).tolist() == [0, 0, 0, 0, 1]   # one endorsed account: Tier 1


def test_tier_scores_follow_the_terciles():
    # Seed 0 admits nodes 4 and 5, which each also receive one negative rating and so stay in Tier 0. Nodes
    # 1, 2 and 3 receive 1, 2 and 3 endorsed positive ratings, the only endorsed accounts. With three scores
    # s1 < s2 < s3 the tercile bounds are s1 + (2/3)(s2 - s1) and s2 + (1/3)(s3 - s2), so the three accounts
    # take Tiers 1, 2 and 3.
    rows = [(9, 0, +1, 0), (8, 0, +1, 0), (0, 4, +1, 5), (9, 4, -1, 5), (0, 5, +1, 5), (9, 5, -1, 5),
            (0, 1, +1, 6), (0, 2, +1, 6), (4, 2, +1, 6), (0, 3, +1, 6), (4, 3, +1, 6), (5, 3, +1, 6)]
    score = [(p + 1) / (p + 2) * math.log1p(p) for p in (1, 2, 3)]
    q1, q2 = score[0] + 2 / 3 * (score[1] - score[0]), score[1] + 1 / 3 * (score[2] - score[1])
    assert score[0] <= q1 < score[1] <= q2 < score[2]
    assert legacy_bfs(graph(rows), 10).tolist() == [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]
