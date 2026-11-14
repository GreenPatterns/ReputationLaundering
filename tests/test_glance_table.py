import json
import shutil
from pathlib import Path

from keeper.paperkit import make_glance_table as g


def test_check_passes_on_committed_outputs():
    assert g.main(["--check"]) == 0


def test_perturbed_source_changes_cell(tmp_path):
    values = json.loads((g.ROOT / g.VALUES).read_text())
    files = set(values["sha256"]) | {g.EXTRA, g.LEGACY}
    for f in files:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(g.ROOT / f, tmp_path / f)
    base, _ = g.build(tmp_path)
    extra = tmp_path / g.EXTRA
    data = json.loads(extra.read_text())
    data["bitcoinalpha"]["amplification"]["mean_gain_u2"] += 0.1
    extra.write_text(json.dumps(data))
    perturbed, _ = g.build(tmp_path)
    assert perturbed != base and "+0.452" in perturbed
