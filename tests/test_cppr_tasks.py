"""keeper.cluster.cppr_tasks and keeper.cluster.protocol_settings: the E3 and E4 task lists (Task 4.2a). CPU only.

The registered protocol (docs/preregistration.md) fixes the command of every CPPR run (Section 6) and which runs
exist (Sections 2.2, 2.5, 2.6). This file states that protocol a second time, as literals copied from the Task 4.2a
brief (LARGEST_FIRST, EPS_CAPPED, EPS_UNCAPPED, CELLS, ...), and checks the generated commands against it as strings.
It also parses every generated command with the driver's own parser (keeper.cppr.cp_vpe_gpu.build_parser) and builds
the driver's RunProtocol from it, so that a flag the driver does not know, or a combination it refuses, fails here
and not on a GPU.

The generators only build strings, so most tests use a results root that exists nowhere (RESULTS). The tests of the
command line and of the root guard write into pytest's tmp_path.
"""
import math
import re
import shlex
import subprocess
import sys
from collections import Counter
from itertools import groupby
from pathlib import Path
from types import MappingProxyType

import pytest

from keeper.cluster import cppr_tasks, protocol_settings, taskpool
from keeper.cppr import cp_vpe_gpu
from keeper.credibility import tiers

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "/data/keeper"
PREREG = "docs/preregistration.md"
LAUNCH = ["env", "-u", "PYTHONPATH", "-u", "VIRTUAL_ENV", "~/.conda/envs/ps/bin/python", "-m", "keeper.cppr.cp_vpe_gpu"]

# ------------------------------------------------------------------ the registered protocol, stated again
LARGEST_FIRST = ["wiki-rfa", "bitcoinotc", "bitcoinalpha"]        # the order of the task list (brief, "Order")
EPS_CAPPED = {"bitcoinalpha": 0.02, "bitcoinotc": 0.01, "wiki-rfa": 0.5}      # Section 2.2; every eps below comes
EPS_UNCAPPED = {"bitcoinalpha": 0.05, "bitcoinotc": 0.05, "wiki-rfa": 0.5}    # from these two tables
E3_SEEDS = list(range(5, 15))
SEEDS_5 = list(range(5, 10))
# Section 2.6, (g, endorsers, topology, s): g = 1 random ring s in {3, 50} and star s = 50; g = 1 max_trust star
# s in {3, 50} and ring s = 3; g = 2 random ring s = 3 and max_trust star s = 3. Each runs under BFS and capped BTP.
CELLS = [(1, "random", "ring", 3), (1, "random", "ring", 50), (1, "random", "star", 50),
         (1, "max_trust", "star", 3), (1, "max_trust", "star", 50), (1, "max_trust", "ring", 3),
         (2, "random", "ring", 3), (2, "max_trust", "star", 3)]
UNCAPPED_CELLS = [(1, "max_trust", "star", 3), (1, "max_trust", "star", 50)]      # the only cells that run uncapped
PAIRED_CELL = (1, "random", "ring", 3)      # under capped BTP it also trains Plain VPE, because E5 pairs with it
ENDORSER_TAG = {"random": "rand", "max_trust": "maxb"}
BASE_CACHE = f"{RESULTS}/.keeper-cache/base"        # E4 only (rulings R59, R61)
E3_ID = re.compile(r"e3_(?P<variant>btpc|bfs|btpu)_(?P<ds>[a-z-]+)_s(?P<seed>\d+)")
E4_ID = re.compile(r"e4_(?P<variant>btpc|bfs|btpu)_(?P<tag>rand|maxb)_(?P<topology>ring|star)_k(?P<s>\d+)_g(?P<g>\d+)"
                   r"_(?P<ds>[a-z-]+)_s(?P<seed>\d+)")


def tier_flags(variant: str, ds: str) -> dict:
    """The flags that set the tier method, by Section 6: BFS has no eps; only the uncapped run lifts the cap."""
    if variant == "bfs":
        return {"--tiers": "bfs"}
    if variant == "btpc":
        return {"--tiers": "btp", "--btp-eps": str(EPS_CAPPED[ds])}
    return {"--tiers": "btp", "--btp-eps": str(EPS_UNCAPPED[ds]), "--btp-edge-cap": "none"}


def expected_e3_flags(row) -> dict:
    """Every flag of an E3 command (Sections 2.5 and 6), the id and output of the row being the only inputs."""
    m = E3_ID.fullmatch(row.id)
    variant, ds = m["variant"], m["ds"]
    flags = {"--ds": ds, "--seed": m["seed"], "--epochs": "150", "--hold-test-snapshot": None, "--prereg": PREREG,
             "--out": row.done_path, **tier_flags(variant, ds), "--arms": "direct,whitewash,badmouth,control",
             "--dump-targets": f"{RESULTS}/results/cppr/e3/targets"}
    if variant != "btpu":
        flags["--dump-verdicts"] = f"{RESULTS}/results/dumps/e3"
    if variant == "btpc":
        flags["--ungated-baseline"] = None
    return flags


def expected_e4_flags(row) -> dict:
    """Every flag of an E4 command (Sections 2.6 and 6, and the base cache of rulings R59 and R61)."""
    m = E4_ID.fullmatch(row.id)
    variant, ds = m["variant"], m["ds"]
    endorsers = {tag: name for name, tag in ENDORSER_TAG.items()}[m["tag"]]
    cell = (int(m["g"]), endorsers, m["topology"], int(m["s"]))
    methods = "plain_vpe,cppr" if (variant, cell) == ("btpc", PAIRED_CELL) else "cppr"
    return {"--ds": ds, "--seed": m["seed"], "--epochs": "150", "--hold-test-snapshot": None, "--prereg": PREREG,
            "--out": row.done_path, **tier_flags(variant, ds), "--pool-eps": str(EPS_CAPPED[ds]),
            "--arms": "whitewash", "--methods": methods, "--attack-edges": m["g"], "--k-fake": m["s"],
            "--sybil-topology": m["topology"], "--endorser-select": endorsers,
            "--dump-targets": f"{RESULTS}/results/cppr/e4/targets", "--base-cache": BASE_CACHE}


def parse(command: str) -> dict:
    """{flag: value} of a driver command, None for a switch. A different launcher, a stray word or a repeated flag
    fails the test."""
    words = shlex.split(command)
    assert words[:len(LAUNCH)] == LAUNCH, command
    rest, flags, i = words[len(LAUNCH):], {}, 0
    while i < len(rest):
        flag = rest[i]
        assert flag.startswith("--") and flag not in flags, f"stray or repeated {flag!r} in {command}"
        has_value = i + 1 < len(rest) and not rest[i + 1].startswith("--")
        flags[flag] = rest[i + 1] if has_value else None
        i += 2 if has_value else 1
    return flags


def variant_of(flags: dict) -> str:
    if flags["--tiers"] == "bfs":
        return "bfs"
    return "btpu" if flags.get("--btp-edge-cap") == "none" else "btpc"


def cell_of(flags: dict) -> tuple:
    return (int(flags["--attack-edges"]), flags["--endorser-select"], flags["--sybil-topology"], int(flags["--k-fake"]))


def cell_tuples(cells) -> list:
    return [(c.g, c.endorsers, c.topology, c.s) for c in cells]


@pytest.fixture(scope="module")
def e3() -> list:
    return cppr_tasks.e3_tasks(RESULTS)


@pytest.fixture(scope="module")
def e4() -> list:
    return cppr_tasks.e4_tasks(RESULTS)


# ------------------------------------------------------------------ protocol_settings
def test_the_registered_constants_are_the_values_of_the_protocol():
    ps = protocol_settings
    assert ps.NETWORKS == ("bitcoinalpha", "bitcoinotc", "wiki-rfa")
    assert ps.EPS_CAPPED == EPS_CAPPED
    assert ps.EPS_UNCAPPED == EPS_UNCAPPED
    assert ps.K_EXPECTED == {"bitcoinalpha": 12, "bitcoinotc": 13, "wiki-rfa": 14}
    assert ps.N_REF_EXPECTED == {"bitcoinalpha": 3629, "bitcoinotc": 5684, "wiki-rfa": 10152}
    assert ps.EDGE_CAP_UNITS == 2.0
    assert ps.NEG_FROM == "trusted"
    assert list(ps.SEEDS_E3) == E3_SEEDS and list(ps.SEEDS_5) == SEEDS_5
    assert ps.PYTHON == "env -u PYTHONPATH -u VIRTUAL_ENV ~/.conda/envs/ps/bin/python"
    assert ps.PREREG == "docs/preregistration.md"
    assert ps.EPOCHS == 150


def test_the_registered_grid_is_the_one_of_sections_2_5_and_2_6():
    ps = protocol_settings
    assert (ps.CAPPED, ps.BFS, ps.UNCAPPED) == ("btpc", "bfs", "btpu")
    assert ps.TIER_LABELS == (ps.CAPPED, ps.BFS, ps.UNCAPPED)
    assert ps.E3_ARMS == "direct,whitewash,badmouth,control"
    assert (ps.E4_ARMS, ps.E4_METHODS, ps.E4_PAIRED_METHODS) == ("whitewash", "cppr", "plain_vpe,cppr")
    assert isinstance(ps.E4_CELLS, tuple) and isinstance(ps.E4_UNCAPPED_CELLS, tuple)     # the analyses call .index
    assert cell_tuples(ps.E4_CELLS) == CELLS
    assert cell_tuples(ps.E4_UNCAPPED_CELLS) == UNCAPPED_CELLS
    paired_cell, paired_label = ps.E4_PAIRED_RUN
    assert (cell_tuples([paired_cell])[0], paired_label) == (PAIRED_CELL, "btpc")
    assert set(ps.E4_UNCAPPED_CELLS) <= set(ps.E4_CELLS) and paired_cell in ps.E4_CELLS


@pytest.mark.parametrize("label", ["btpc", "bfs", "btpu"])
@pytest.mark.parametrize("ds", LARGEST_FIRST)
def test_the_flag_set_of_a_tier_label_is_the_one_of_section_6(label, ds):
    flags = protocol_settings.tier_flags(label, ds)
    assert flags == tier_flags(label, ds)
    assert list(flags) == list(tier_flags(label, ds))                                         # in command order
    flags["--tiers"] = "changed"
    assert protocol_settings.tier_flags(label, ds)["--tiers"] != "changed"                    # a fresh copy each time


def test_an_unknown_tier_label_or_network_is_refused():
    with pytest.raises(ValueError, match="unknown tier setting"):
        protocol_settings.tier_flags("btp", "wiki-rfa")
    with pytest.raises(KeyError):
        protocol_settings.tier_flags("btpc", "epinions-dated")


@pytest.mark.parametrize("table", ["EPS_CAPPED", "EPS_UNCAPPED", "K_EXPECTED", "N_REF_EXPECTED"])
def test_every_per_network_table_covers_exactly_the_three_networks_and_is_read_only(table):
    mapping = getattr(protocol_settings, table)
    assert set(mapping) == set(protocol_settings.NETWORKS)
    with pytest.raises(TypeError):
        mapping["wiki-rfa"] = 1
    with pytest.raises(TypeError):
        mapping["epinions-dated"] = 1


def test_the_five_seeds_of_the_uncapped_runs_are_among_the_e3_seeds():
    assert set(protocol_settings.SEEDS_5) <= set(protocol_settings.SEEDS_E3)


def test_the_registered_edge_cap_is_the_cap_the_code_defaults_to():
    assert protocol_settings.EDGE_CAP_UNITS == tiers.KEEPER_EDGE_CAP_UNITS


@pytest.mark.parametrize("ds", LARGEST_FIRST)
def test_the_registered_k_is_ceil_log2_of_the_registered_n_ref(ds):
    n_ref = protocol_settings.N_REF_EXPECTED[ds]
    assert protocol_settings.K_EXPECTED[ds] == math.ceil(math.log2(n_ref))
    assert protocol_settings.K_EXPECTED[ds] == tiers.default_K(n_ref)                  # the code's exact integer form


# ------------------------------------------------------------------ counts, identity, order
def test_task_counts(e3, e4):
    assert len(e3) == 75
    assert len(e4) == 270
    assert Counter(E3_ID.fullmatch(row.id)["ds"] for row in e3) == {ds: 25 for ds in LARGEST_FIRST}
    assert Counter(E4_ID.fullmatch(row.id)["ds"] for row in e4) == {ds: 90 for ds in LARGEST_FIRST}


@pytest.mark.parametrize("family", ["e3", "e4"])
def test_ids_and_outputs_are_unique(family, e3, e4):
    rows = {"e3": e3, "e4": e4}[family]
    assert len({row.id for row in rows}) == len(rows)
    assert len({row.done_path for row in rows}) == len(rows)
    assert len({row.log_path for row in rows}) == len(rows)


@pytest.mark.parametrize("family", ["e3", "e4"])
def test_the_id_is_the_output_stem_and_every_path_is_absolute_under_the_results_root(family, e3, e4):
    for row in {"e3": e3, "e4": e4}[family]:
        assert row.done_path == f"{RESULTS}/results/cppr/{family}/{row.id}.json"
        assert row.log_path == f"{RESULTS}/results/cppr/{family}/logs/{row.id}.log"
        assert Path(row.done_path).stem == row.id
        assert parse(row.command)["--out"] == row.done_path


@pytest.mark.parametrize("family", ["e3", "e4"])
def test_wiki_rfa_comes_first_then_otc_then_alpha(family, e3, e4):
    rows = {"e3": e3, "e4": e4}[family]
    blocks = [ds for ds, _ in groupby(parse(row.command)["--ds"] for row in rows)]
    assert blocks == LARGEST_FIRST


def test_the_task_order_is_the_documented_one(e3, e4):
    """Networks largest first, seeds ascending; E3 runs capped, BFS, uncapped; E4 follows CELLS, BFS before capped."""
    expected_e3 = [f"e3_{variant}_{ds}_s{seed}" for ds in LARGEST_FIRST for seed in E3_SEEDS
                   for variant in ("btpc", "bfs", "btpu")[:3 if seed in SEEDS_5 else 2]]
    expected_e4 = [f"e4_{variant}_{ENDORSER_TAG[endorsers]}_{topology}_k{s}_g{g}_{ds}_s{seed}"
                   for ds in LARGEST_FIRST for seed in SEEDS_5 for (g, endorsers, topology, s) in CELLS
                   for variant in ("bfs", "btpc", "btpu")[:3 if (g, endorsers, topology, s) in UNCAPPED_CELLS else 2]]
    assert [row.id for row in e3] == expected_e3
    assert [row.id for row in e4] == expected_e4


# ------------------------------------------------------------------ flags every command carries
def test_every_command_holds_the_snapshot_fixes_the_protocol_hash_and_trains_150_epochs(e3, e4):
    for row in e3 + e4:
        assert row.command.startswith(" ".join(LAUNCH) + " --ds ")
        for text in ("--hold-test-snapshot", "--prereg docs/preregistration.md", "--epochs 150"):
            assert text in row.command, (text, row.command)
        flags = parse(row.command)
        assert flags["--hold-test-snapshot"] is None
        assert flags["--prereg"] == PREREG and flags["--epochs"] == "150"


@pytest.mark.parametrize("family", ["e3", "e4"])
def test_the_seed_and_network_of_a_command_are_those_of_its_id(family, e3, e4):
    rows = {"e3": e3, "e4": e4}[family]
    pattern = {"e3": E3_ID, "e4": E4_ID}[family]
    for row in rows:
        flags, m = parse(row.command), pattern.fullmatch(row.id)
        assert (flags["--ds"], flags["--seed"]) == (m["ds"], m["seed"])


def test_no_command_passes_the_credible_negatives_guard_because_trusted_is_the_drivers_default(e3, e4):
    assert not any("neg-from" in row.command for row in e3 + e4)


# ------------------------------------------------------------------ tier flags
@pytest.mark.parametrize("family", ["e3", "e4"])
def test_the_eps_of_every_command_is_the_registered_one_of_its_network_and_tier_method(family, e3, e4):
    for row in {"e3": e3, "e4": e4}[family]:
        flags = parse(row.command)
        variant, ds = variant_of(flags), flags["--ds"]
        if variant == "bfs":
            assert "--btp-eps" not in flags
            continue
        registry = {"btpc": protocol_settings.EPS_CAPPED, "btpu": protocol_settings.EPS_UNCAPPED}[variant]
        restated = {"btpc": EPS_CAPPED, "btpu": EPS_UNCAPPED}[variant]
        assert float(flags["--btp-eps"]) == registry[ds]
        assert flags["--btp-eps"] == str(restated[ds])                      # and in the registered spelling


@pytest.mark.parametrize("family", ["e3", "e4"])
def test_uncapped_commands_alone_lift_the_cap_and_capped_ones_keep_the_default(family, e3, e4):
    pattern = {"e3": E3_ID, "e4": E4_ID}[family]
    for row in {"e3": e3, "e4": e4}[family]:
        flags, variant = parse(row.command), pattern.fullmatch(row.id)["variant"]
        if variant == "btpu":
            assert flags["--btp-edge-cap"] == "none" and "--btp-edge-cap none" in row.command
        else:
            assert "--btp-edge-cap" not in row.command, row.id
        assert variant == variant_of(flags)


def test_capped_btp_alone_runs_the_ungated_baseline_and_only_in_e3(e3, e4):
    assert {variant_of(parse(row.command)) for row in e3 if "--ungated-baseline" in row.command} == {"btpc"}
    assert sum("--ungated-baseline" in row.command for row in e3) == 30
    assert not any("--ungated-baseline" in row.command for row in e4)


# ------------------------------------------------------------------ E3
def test_e3_runs_capped_btp_and_bfs_on_ten_seeds_and_uncapped_btp_on_five(e3):
    runs = Counter((m["variant"], m["ds"], int(m["seed"])) for m in (E3_ID.fullmatch(row.id) for row in e3))
    expected = Counter()
    for ds in LARGEST_FIRST:
        expected.update({("btpc", ds, seed): 1 for seed in E3_SEEDS})
        expected.update({("bfs", ds, seed): 1 for seed in E3_SEEDS})
        expected.update({("btpu", ds, seed): 1 for seed in SEEDS_5})
    assert runs == expected


def test_every_e3_command_has_exactly_the_flags_of_section_2_5(e3):
    for row in e3:
        assert parse(row.command) == expected_e3_flags(row), row.id


def test_e3_dumps_verdicts_for_the_capped_and_bfs_runs_only(e3):
    for row in e3:
        flags = parse(row.command)
        assert ("--dump-verdicts" in flags) == (variant_of(flags) != "btpu"), row.id
        assert flags["--dump-targets"] == f"{RESULTS}/results/cppr/e3/targets"


def test_e3_relies_on_the_code_defaults_for_the_attack_edges_and_takes_no_cache(e3):
    for row in e3:
        for flag in ("--attack-edges", "--k-fake", "--sybil-topology", "--endorser-select", "--pool-eps",
                     "--methods", "--base-cache"):
            assert flag not in row.command, (flag, row.id)


# ------------------------------------------------------------------ E4
def test_e4_has_the_eighteen_cells_of_section_2_6_per_network_and_seed(e4):
    expected = Counter({(variant, cell): 1 for cell in CELLS for variant in ("bfs", "btpc")})
    expected.update({("btpu", cell): 1 for cell in UNCAPPED_CELLS})
    assert sum(expected.values()) == 18
    by_run = {}
    for row in e4:
        flags = parse(row.command)
        by_run.setdefault((flags["--ds"], flags["--seed"]), Counter())[(variant_of(flags), cell_of(flags))] += 1
    assert set(by_run) == {(ds, str(seed)) for ds in LARGEST_FIRST for seed in SEEDS_5}
    assert all(cells == expected for cells in by_run.values())      # each run once: a Counter shows a duplicate


def test_e4_runs_uncapped_btp_only_in_the_two_g1_max_trust_star_cells(e4):
    uncapped = [parse(row.command) for row in e4 if "--btp-edge-cap" in row.command]
    assert len(uncapped) == 2 * 15
    assert Counter(cell_of(flags) for flags in uncapped) == {cell: 15 for cell in UNCAPPED_CELLS}


def test_every_e4_command_has_the_flags_of_section_2_6_and_section_6(e4):
    for row in e4:
        assert parse(row.command) == expected_e4_flags(row), row.id


def test_every_e4_command_takes_pool_eps_equal_to_eps_c_and_trains_cppr(e4):
    for row in e4:
        flags = parse(row.command)
        assert float(flags["--pool-eps"]) == protocol_settings.EPS_CAPPED[flags["--ds"]]
        assert f"--pool-eps {EPS_CAPPED[flags['--ds']]}" in row.command
        assert "--methods cppr" in row.command or "--methods plain_vpe,cppr" in row.command
        assert flags["--methods"].split(",")[-1] == "cppr"


def test_only_the_capped_g1_random_ring_s3_cell_also_trains_plain_vpe(e4):
    paired = [row for row in e4 if "--methods plain_vpe,cppr" in row.command]
    assert len(paired) == 15
    assert {E4_ID.fullmatch(row.id)["variant"] for row in paired} == {"btpc"}
    assert {cell_of(parse(row.command)) for row in paired} == {PAIRED_CELL}
    assert sum("--methods cppr" in row.command for row in e4) == 270 - 15


def test_the_bfs_cells_take_no_btp_eps_and_the_others_take_the_eps_of_their_tier_method(e4):
    bfs = [row for row in e4 if "--tiers bfs" in row.command]
    assert len(bfs) == 8 * 15
    assert not any("--btp-eps" in row.command for row in bfs)
    assert all("--btp-eps" in row.command for row in e4 if "--tiers btp" in row.command)


def test_every_e4_command_takes_the_base_cache_under_the_results_root_outside_results(e4):
    for row in e4:
        cache = parse(row.command)["--base-cache"]
        assert cache == f"{RESULTS}/.keeper-cache/base", row.id
        assert not cache.startswith(f"{RESULTS}/results/")


def test_e4_file_names_tell_the_flags_of_their_run(e4):
    for row in e4:
        m, flags = E4_ID.fullmatch(row.id), parse(row.command)
        assert m["variant"] == variant_of(flags)
        assert (ENDORSER_TAG[flags["--endorser-select"]], flags["--sybil-topology"]) == (m["tag"], m["topology"])
        assert (flags["--k-fake"], flags["--attack-edges"]) == (m["s"], m["g"])


# ------------------------------------------------------------------ the driver's own parser
def refuse(message):
    raise ValueError(message)


def under_the_driver(command: str) -> tuple:
    """(parsed arguments, RunProtocol) of a command under keeper.cppr.cp_vpe_gpu's own parser. A flag or a combination
    that the driver refuses raises ValueError instead of ending the process."""
    parser = cp_vpe_gpu.build_parser()
    parser.error = refuse
    args = parser.parse_args(shlex.split(command)[len(LAUNCH):])
    return args, cp_vpe_gpu.protocol_from_args(args)


def check_common_protocol(row, args, protocol, flags):
    variant = variant_of(flags)
    assert (args.ds, args.seed, args.epochs, args.out) == (flags["--ds"], int(flags["--seed"]), 150, row.done_path)
    # the code defaults that Section 2.5 registers (15 edge and 5 account targets, rho 0.02) and the count gate
    assert (args.n_edge, args.n_acc, args.rho, args.granularity, args.gate_norm) == (15, 5, 0.02, 1, "count")
    assert protocol.hold_test_snapshot is True and protocol.prereg == PREREG and not protocol.is_legacy()
    assert protocol.tiers == ("bfs" if variant == "bfs" else "btp")
    assert protocol.btp_eps == (None if variant == "bfs" else float(flags["--btp-eps"]))
    assert protocol.btp_edge_cap == (None if variant == "btpu" else protocol_settings.EDGE_CAP_UNITS)
    # the credible-negatives guard is not passed, so the driver's default must be the registered one (R56)
    assert getattr(protocol, "btp_neg_from", protocol_settings.NEG_FROM) == protocol_settings.NEG_FROM


def check_e3_protocol(args, protocol, flags):
    variant = variant_of(flags)
    assert protocol.arms == ("direct", "whitewash", "badmouth", "control")
    assert protocol.methods == ("plain_erasure", "plain_vpe", "cppr")            # all three, as the stored runs
    assert (protocol.attack_edges, protocol.sybil_topology, protocol.endorser_select) == (0, "ring", "random")
    assert (args.k_fake, protocol.pool_eps, protocol.base_cache) == (3, None, None)
    assert protocol.ungated_baseline == (variant == "btpc")
    assert protocol.dump_targets == f"{RESULTS}/results/cppr/e3/targets"
    assert protocol.dump_verdicts == (None if variant == "btpu" else f"{RESULTS}/results/dumps/e3")


def check_e4_protocol(args, protocol, flags):
    cell = cell_of(flags)
    paired = (variant_of(flags), cell) == ("btpc", PAIRED_CELL)
    assert protocol.arms == ("whitewash",)
    assert protocol.methods == (("plain_vpe", "cppr") if paired else ("cppr",))
    assert (protocol.attack_edges, protocol.endorser_select, protocol.sybil_topology, args.k_fake) == cell
    assert protocol.pool_eps == EPS_CAPPED[flags["--ds"]]
    assert protocol.base_cache == BASE_CACHE
    assert protocol.dump_targets == f"{RESULTS}/results/cppr/e4/targets"
    assert protocol.dump_verdicts is None and protocol.ungated_baseline is False


def test_the_driver_parses_every_generated_command_into_the_registered_run_protocol(e3, e4):
    for row in e3 + e4:
        flags = parse(row.command)
        args, protocol = under_the_driver(row.command)
        check_common_protocol(row, args, protocol, flags)
        (check_e3_protocol if row.id.startswith("e3_") else check_e4_protocol)(args, protocol, flags)


def test_the_driver_refuses_a_command_with_a_flag_it_does_not_know_or_a_missing_pool_eps(e4):
    command = e4[0].command
    with pytest.raises(ValueError, match="unrecognized"):
        under_the_driver(command + " --no-such-flag 1")
    with pytest.raises(ValueError, match="pool-eps"):
        under_the_driver(command.replace(" --pool-eps 0.5", ""))


# ------------------------------------------------------------------ whole commands
def test_golden_commands(e3, e4):
    e3_by_id, e4_by_id = ({row.id: row.command for row in rows} for rows in (e3, e4))
    prefix = " ".join(LAUNCH)
    common = "--epochs 150 --hold-test-snapshot --prereg docs/preregistration.md"
    e3_tail = "--arms direct,whitewash,badmouth,control --dump-targets /data/keeper/results/cppr/e3/targets"
    wiki, otc, alpha = EPS_CAPPED["wiki-rfa"], EPS_CAPPED["bitcoinotc"], EPS_CAPPED["bitcoinalpha"]
    assert e3_by_id["e3_btpc_wiki-rfa_s5"] == (
        f"{prefix} --ds wiki-rfa --seed 5 {common} --out /data/keeper/results/cppr/e3/e3_btpc_wiki-rfa_s5.json "
        f"--tiers btp --btp-eps {wiki} {e3_tail} --dump-verdicts /data/keeper/results/dumps/e3 --ungated-baseline")
    assert e3_by_id["e3_bfs_bitcoinalpha_s14"] == (
        f"{prefix} --ds bitcoinalpha --seed 14 {common} "
        f"--out /data/keeper/results/cppr/e3/e3_bfs_bitcoinalpha_s14.json --tiers bfs {e3_tail} "
        "--dump-verdicts /data/keeper/results/dumps/e3")
    assert e3_by_id["e3_btpu_bitcoinotc_s7"] == (
        f"{prefix} --ds bitcoinotc --seed 7 {common} --out /data/keeper/results/cppr/e3/e3_btpu_bitcoinotc_s7.json "
        f"--tiers btp --btp-eps {EPS_UNCAPPED['bitcoinotc']} --btp-edge-cap none {e3_tail}")
    e4_tail = ("--dump-targets /data/keeper/results/cppr/e4/targets "
               "--base-cache /data/keeper/.keeper-cache/base")
    assert e4_by_id["e4_btpc_rand_ring_k3_g1_wiki-rfa_s5"] == (
        f"{prefix} --ds wiki-rfa --seed 5 {common} "
        f"--out /data/keeper/results/cppr/e4/e4_btpc_rand_ring_k3_g1_wiki-rfa_s5.json --tiers btp --btp-eps {wiki} "
        f"--pool-eps {wiki} --arms whitewash --methods plain_vpe,cppr --attack-edges 1 --k-fake 3 "
        f"--sybil-topology ring --endorser-select random {e4_tail}")
    assert e4_by_id["e4_bfs_maxb_ring_k3_g1_bitcoinotc_s9"] == (
        f"{prefix} --ds bitcoinotc --seed 9 {common} "
        f"--out /data/keeper/results/cppr/e4/e4_bfs_maxb_ring_k3_g1_bitcoinotc_s9.json --tiers bfs "
        f"--pool-eps {otc} --arms whitewash --methods cppr --attack-edges 1 --k-fake 3 "
        f"--sybil-topology ring --endorser-select max_trust {e4_tail}")
    assert e4_by_id["e4_btpu_maxb_star_k50_g1_bitcoinalpha_s6"] == (
        f"{prefix} --ds bitcoinalpha --seed 6 {common} "
        f"--out /data/keeper/results/cppr/e4/e4_btpu_maxb_star_k50_g1_bitcoinalpha_s6.json --tiers btp "
        f"--btp-eps {EPS_UNCAPPED['bitcoinalpha']} --btp-edge-cap none --pool-eps {alpha} --arms whitewash "
        f"--methods cppr --attack-edges 1 --k-fake 50 --sybil-topology star --endorser-select max_trust {e4_tail}")


def test_every_registered_value_is_read_from_protocol_settings(monkeypatch):
    ps = protocol_settings
    monkeypatch.setattr(ps, "PYTHON", "pyX")
    monkeypatch.setattr(ps, "PREREG", "somewhere/else.md")
    monkeypatch.setattr(ps, "EPOCHS", 7)
    monkeypatch.setattr(ps, "E3_ARMS", "direct")
    monkeypatch.setattr(ps, "NETWORKS", ("bitcoinotc", "wiki-rfa"))
    monkeypatch.setattr(ps, "SEEDS_E3", range(20, 23))
    monkeypatch.setattr(ps, "SEEDS_5", range(20, 22))
    monkeypatch.setattr(ps, "EPS_CAPPED", MappingProxyType({**ps.EPS_CAPPED, "wiki-rfa": 0.25}))
    monkeypatch.setattr(ps, "EPS_UNCAPPED", MappingProxyType({**ps.EPS_UNCAPPED, "bitcoinotc": 0.125}))
    rows = cppr_tasks.e3_tasks(RESULTS) + cppr_tasks.e4_tasks(RESULTS)
    assert len(rows) == 2 * (3 * 2 + 2) + 2 * 2 * 18
    assert all(row.command.startswith("pyX -m keeper.cppr.cp_vpe_gpu --ds ") for row in rows)
    assert all("--prereg somewhere/else.md" in row.command and "--epochs 7 " in row.command for row in rows)
    assert all("--arms direct " in row.command for row in rows[:16])
    assert [ds for ds, _ in groupby(re.search(r"--ds (\S+)", row.command)[1] for row in rows[:14])] == \
        ["wiki-rfa", "bitcoinotc"]
    assert {re.search(r"--seed (\d+)", row.command)[1] for row in rows} == {"20", "21", "22"}
    assert any("--tiers btp --btp-eps 0.25 " in row.command for row in rows)
    assert any("--btp-eps 0.125 --btp-edge-cap none" in row.command for row in rows)


def test_the_e4_grid_is_read_from_protocol_settings(monkeypatch):
    ps = protocol_settings
    cell = ps.Cell(3, "random", "star", 7)
    other = ps.Cell(1, "max_trust", "ring", 9)
    monkeypatch.setattr(ps, "E4_CELLS", (cell, other))
    monkeypatch.setattr(ps, "E4_UNCAPPED_CELLS", (other,))
    monkeypatch.setattr(ps, "E4_PAIRED_RUN", (cell, ps.BFS))
    monkeypatch.setattr(ps, "E4_ARMS", "badmouth")
    monkeypatch.setattr(ps, "E4_METHODS", "cppr,plain_erasure")
    monkeypatch.setattr(ps, "E4_PAIRED_METHODS", "plain_erasure,cppr")
    rows = cppr_tasks.e4_tasks(RESULTS)
    assert len(rows) == 3 * 5 * (2 + 3)
    assert all("--arms badmouth " in row.command for row in rows)
    paired = [row for row in rows if "--methods plain_erasure,cppr" in row.command]
    assert len(paired) == 15 and all(row.id.startswith("e4_bfs_rand_star_k7_g3_") for row in paired)
    assert sum("--methods cppr,plain_erasure" in row.command for row in rows) == len(rows) - 15
    assert sum("--btp-edge-cap none" in row.command for row in rows) == 15
    assert {row.id.split("_")[1] for row in rows} == {"bfs", "btpc", "btpu"}


def test_the_names_the_analyses_import_from_cppr_tasks_are_those_of_protocol_settings():
    ps = protocol_settings
    for name in ("BFS", "CAPPED", "UNCAPPED", "Cell", "E4_CELLS", "E4_UNCAPPED_CELLS", "E4_PAIRED_RUN"):
        assert getattr(cppr_tasks, name) is getattr(ps, name), name
    assert cppr_tasks.ENDORSER_TAGS == ENDORSER_TAG


# ------------------------------------------------------------------ the results root
def test_a_results_root_with_a_space_is_quoted_in_the_command_and_not_in_the_paths():
    rows = cppr_tasks.e3_tasks("/data/my repo") + cppr_tasks.e4_tasks("/data/my repo")
    for row in rows:
        flags = parse(row.command)
        assert row.done_path.startswith("/data/my repo/results/cppr/")
        assert flags["--out"] == row.done_path
        assert flags["--dump-targets"].startswith("/data/my repo/results/cppr/")
        if row.id.startswith("e4_"):
            assert flags["--base-cache"] == "/data/my repo/.keeper-cache/base"


def test_a_results_root_is_normalised():
    assert cppr_tasks.e3_tasks("/data/keeper/") == cppr_tasks.e3_tasks("/data/other/../keeper")
    assert cppr_tasks.e4_tasks("/data/keeper/") == cppr_tasks.e4_tasks("/data/other/../keeper")


@pytest.mark.parametrize("root", ["data/keeper", "./keeper", "", "/data/ke\teper", "/data/ke\neper", "/data/ke\reper"])
@pytest.mark.parametrize("generate", ["e3_tasks", "e4_tasks"])
def test_a_results_root_that_is_not_absolute_or_holds_a_tab_or_line_break_is_refused(generate, root):
    with pytest.raises(ValueError, match="results root"):
        getattr(cppr_tasks, generate)(root)


def make_git_worktree(path: Path) -> Path:
    """A folder whose .git is a file, as in a git worktree."""
    path.mkdir()
    (path / ".git").write_text("gitdir: /repo/.git/worktrees/" + path.name + "\n")
    return path


@pytest.mark.parametrize("generate", ["e3_tasks", "e4_tasks"])
def test_a_results_root_that_is_a_git_worktree_is_refused(generate, tmp_path):
    worktree = make_git_worktree(tmp_path / "worktree")
    with pytest.raises(ValueError, match="git worktree"):
        getattr(cppr_tasks, generate)(str(worktree))
    with pytest.raises(ValueError, match="git worktree"):
        getattr(cppr_tasks, generate)(str(worktree) + "/")                 # the check follows the normalised path


def test_a_results_root_with_a_git_folder_or_none_is_accepted(tmp_path):
    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()
    for root in (main, plain, tmp_path / "does_not_exist_yet"):
        assert len(cppr_tasks.e3_tasks(str(root))) == 75


# ------------------------------------------------------------------ the task file
@pytest.mark.parametrize("family", ["e3", "e4"])
def test_the_task_file_parses_with_the_task_pools_own_parser(family, e3, e4):
    rows = {"e3": e3, "e4": e4}[family]
    text = cppr_tasks.render(rows)
    parsed = taskpool.parse_tasks(text)
    assert [(t.id, t.done_path, t.log_path, t.command) for t in parsed] == [tuple(row) for row in rows]
    assert len(parsed) == {"e3": 75, "e4": 270}[family]
    assert text.endswith("\n") and "\r" not in text
    assert all(len(line.split("\t")) == 4 for line in text.splitlines())


def test_the_task_pool_accepts_both_files_in_a_dry_run(tmp_path, capsys):
    for family, count in (("e3", 75), ("e4", 270)):
        out = tmp_path / f"{family}.tsv"
        assert cppr_tasks.main([family, "--results-root", str(tmp_path / "repo"), "--out", str(out)]) == 0
        capsys.readouterr()
        assert len(taskpool.load_tasks(str(out))) == count
        assert taskpool.main([str(out), "--gpus", "0", "--dry-run"]) == 0
        plan = capsys.readouterr().out
        # every task is pending in a fresh root; the pool's summary may count other states between the two figures
        assert re.search(rf"\b{count} tasks\b.*\b{count} pending\b", plan), plan
    assert sorted(path.name for path in tmp_path.iterdir()) == ["e3.tsv", "e4.tsv"]      # a dry run creates nothing


# ------------------------------------------------------------------ the command line
@pytest.mark.parametrize("family, count", [("e3", 75), ("e4", 270)])
def test_the_command_line_writes_the_task_file(tmp_path, capsys, family, count):
    out = tmp_path / "tasks.tsv"
    assert cppr_tasks.main([family, "--results-root", RESULTS, "--out", str(out)]) == 0
    generate = {"e3": cppr_tasks.e3_tasks, "e4": cppr_tasks.e4_tasks}[family]
    assert out.read_text(encoding="utf-8") == cppr_tasks.render(generate(RESULTS))
    assert f"wrote {count} tasks" in capsys.readouterr().out


def test_the_same_arguments_give_the_same_bytes(tmp_path):
    first, second = tmp_path / "first.tsv", tmp_path / "second.tsv"
    for out in (first, second):
        assert cppr_tasks.main(["e4", "--results-root", RESULTS, "--out", str(out)]) == 0
    assert first.read_bytes() == second.read_bytes()


def test_an_existing_task_file_is_refused_without_force_and_replaced_with_it(tmp_path, capsys):
    out = tmp_path / "e3.tsv"
    out.write_text("precious\n")
    args = ["e3", "--results-root", RESULTS, "--out", str(out)]
    assert cppr_tasks.main(args) == 1
    assert out.read_text() == "precious\n"
    err = capsys.readouterr().err
    assert str(out) in err and "--force" in err
    assert cppr_tasks.main([*args, "--force"]) == 0
    assert len(out.read_text().splitlines()) == 75


def test_an_empty_existing_file_is_refused_too(tmp_path):
    out = tmp_path / "e3.tsv"
    out.write_text("")
    assert cppr_tasks.main(["e3", "--results-root", RESULTS, "--out", str(out)]) == 1
    assert out.read_text() == ""


def test_a_folder_that_does_not_exist_is_reported_not_raised(tmp_path, capsys):
    out = tmp_path / "nowhere" / "e3.tsv"
    assert cppr_tasks.main(["e3", "--results-root", RESULTS, "--out", str(out)]) == 1
    assert "cannot write" in capsys.readouterr().err
    assert not out.parent.exists()


@pytest.mark.parametrize("root", ["relative/root", "/data/ke\neper"])
def test_a_bad_results_root_is_a_usage_error_and_writes_nothing(tmp_path, capsys, root):
    out = tmp_path / "e3.tsv"
    with pytest.raises(SystemExit) as stop:
        cppr_tasks.main(["e3", "--results-root", root, "--out", str(out)])
    assert stop.value.code == 2
    assert "results root" in capsys.readouterr().err
    assert not out.exists()


def test_a_git_worktree_as_results_root_is_a_usage_error_and_writes_nothing(tmp_path, capsys):
    worktree = make_git_worktree(tmp_path / "worktree")
    out = tmp_path / "e4.tsv"
    with pytest.raises(SystemExit) as stop:
        cppr_tasks.main(["e4", "--results-root", str(worktree), "--out", str(out)])
    assert stop.value.code == 2
    err = capsys.readouterr().err
    assert "git worktree" in err and "main repository" in err
    assert not out.exists()


@pytest.mark.parametrize("args", [["e5", "--results-root", RESULTS, "--out", "x.tsv"],
                                  ["e3", "--out", "x.tsv"], ["e3", "--results-root", RESULTS]])
def test_an_unknown_experiment_or_a_missing_option_is_a_usage_error(args):
    with pytest.raises(SystemExit) as stop:
        cppr_tasks.main(args)
    assert stop.value.code == 2


def test_runs_as_a_module(tmp_path):
    out = tmp_path / "e4.tsv"
    res = subprocess.run([sys.executable, "-m", "keeper.cluster.cppr_tasks", "e4", "--results-root", RESULTS,
                          "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert len(taskpool.load_tasks(str(out))) == 270
    again = subprocess.run([sys.executable, "-m", "keeper.cluster.cppr_tasks", "e4", "--results-root", RESULTS,
                            "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
    assert again.returncode == 1 and "--force" in again.stderr
