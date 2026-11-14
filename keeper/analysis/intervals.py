"""Interval plumbing for the re-analyses: bootstrap design per dataset, Boot streams and output blocks.

Every headline field of keeper.analysis gets its 95% interval here (ruling R30): the hull of the
crossed bootstrap interval (ruling R29, or the nested one of ruling R21 for a dataset whose seeds drew
different targets) and a t-interval over the per-seed statistics; a 0/1-valued field at exactly 0 or 1 gets
the Clopper-Pearson interval over its distinct units instead. The headline interval is then clipped to the
field's mathematical support, which the caller declares per field (ruling R31). A Block collects a
section's fields with all their intervals; Boot gives each (dataset, field, design) its own reproducible
random stream.
"""
import zlib

from keeper.analysis.bootstrap import (DESIGNS, N_BOOT, clopper_pearson, column_means_stat, combine_mean,
                                       combine_ratio, hier_bootstrap, hull, mean_stat, n_units,
                                       per_seed_values, ratio_seed_t_interval, seed_t_interval)

BOOT_SEED = 20261026


def ci_design_for(samples: dict) -> tuple:
    """("crossed", None) if every seed drew the same targets (ruling R29), else ("nested", reason)."""
    sets = {s: frozenset(v) for s, v in samples.items()}
    if len(set(sets.values())) <= 1:
        return "crossed", None
    return "nested", ("the sampled targets differ across seeds, so the crossed bootstrap cannot align "
                      "them; the nested bootstrap (ruling R21) gives the headline intervals")


class Boot:
    """Bootstrap intervals with a reproducible stream per (dataset, field) and design.

    ``design`` is the headline design: "crossed" (ruling R29) or, for a dataset whose seeds drew
    different targets, "nested". The nested interval (ruling R21) is always computed as a sensitivity.
    """

    def __init__(self, ds: str, n_boot: int = N_BOOT, base_seed: int = BOOT_SEED, design: str = "crossed"):
        if design not in DESIGNS:
            raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")
        self.ds, self.n_boot, self.base_seed, self.design = ds, int(n_boot), int(base_seed), design
        self.undefined = {}                    # field -> {design: share of undefined replicates}, if any

    def seed_for(self, name: str, design: str = "nested") -> list:
        base = [self.base_seed, zlib.crc32(f"{self.ds}/{name}".encode())]
        return base if design == "nested" else base + [1]

    def _run(self, name: str, groups, stat, combine, design: str) -> tuple:
        info = {}
        point, lo, hi = hier_bootstrap(groups, stat, combine, n_boot=self.n_boot, design=design,
                                       seed=self.seed_for(name, design), info=info)
        if info.get("undefined_share"):
            self.undefined.setdefault(name, {})[design] = info["undefined_share"]
        return point, [lo, hi]

    def __call__(self, name: str, groups, stat=mean_stat, combine=combine_mean) -> tuple:
        """(point, headline bootstrap [lo, hi], nested [lo, hi])."""
        point, nested = self._run(name, groups, stat, combine, "nested")
        if self.design == "nested":
            return point, nested, nested
        point, crossed = self._run(name, groups, stat, combine, "crossed")
        return point, crossed, nested


CI_KEYS = ("ci95", "ci95_bootstrap", "ci95_nested", "ci95_seed_t", "ci95_source")


def seed_t_for(groups, stat, combine) -> list | None:
    """Ruling R30: t-interval over the per-seed statistics; linearized for a ratio of seed-averaged means."""
    if combine is combine_ratio:
        pairs = per_seed_values(groups, column_means_stat)
        return ratio_seed_t_interval([None if p is None else p[0] for p in pairs],
                                     [None if p is None else p[1] for p in pairs])
    return seed_t_interval(per_seed_values(groups, stat))


def headline_interval(point: float, boot_ci: list, seed_t: list | None, groups, unit: str | None) -> tuple:
    """(ci95, source): Clopper-Pearson for a 0/1 field at exactly 0 or 1, else the hull of the bootstrap
    and seed-t intervals (ruling R30)."""
    if unit is not None and point in (0.0, 1.0):
        n = n_units(groups, unit)
        units = "distinct targets" if unit == "targets" else "evaluations"
        return clopper_pearson(n if point == 1.0 else 0, n), f"Clopper-Pearson over {n} {units}"
    if seed_t is None:
        return boot_ci, "bootstrap (fewer than two seeds for a t-interval)"
    return hull(boot_ci, seed_t), "hull of the bootstrap and seed-t intervals"


PROPORTION = (0.0, 1.0)            # proportions, escape rates, percentiles stored as fractions, proxies, tau
NON_NEGATIVE = (0.0, None)         # magnitudes, and ratios of non-negative quantities


def clip_to_support(ci: list | None, support: tuple | None) -> tuple:
    """(interval clipped to support (lo, hi), changed); either bound may be None for unbounded (R31)."""
    if ci is None or support is None:
        return ci, False
    lo, hi = support
    clipped = [ci[0] if lo is None else max(lo, ci[0]), ci[1] if hi is None else min(hi, ci[1])]
    return clipped, clipped != list(ci)


def _support_text(support: tuple) -> str:
    lo, hi = support
    return f"[{'-inf' if lo is None else f'{lo:g}'}, {'inf' if hi is None else f'{hi:g}'}]"


class Block:
    """One output block: headline fields, per-seed values (by_seed) and their intervals (CI_KEYS)."""

    def __init__(self, name: str, boot: Boot):
        self.name, self.boot = name, boot
        self.fields, self.by_seed, self.extra = {}, {}, {}
        self.ci = {key: {} for key in CI_KEYS}
        self.support, self.unclipped = {}, {}

    def stat(self, field: str, groups, stat=mean_stat, combine=combine_mean, shown=None,
             unit: str | None = None, *, support: tuple | None) -> float:
        """Point estimate and intervals of one field.

        ``unit`` ("targets" or "evaluations") marks a 0/1-valued field and names the units of its exact
        interval at a boundary. ``support`` is required (ruling R31): (lo, hi), either bound None, is the
        field's mathematical support, and the headline interval is clipped to it; None declares a signed
        quantity or a ratio with a signed part, which is not clipped.
        """
        point, boot_ci, nested = self.boot(f"{self.name}/{field}", groups, stat, combine)
        seed_t = seed_t_for(groups, stat, combine)
        ci, source = headline_interval(point, boot_ci, seed_t, groups, unit)
        clipped, changed = clip_to_support(ci, support)
        if changed:
            self.unclipped[field] = ci
            source += f", clipped to {_support_text(support)}"
        self.fields[field] = point
        for key, value in zip(CI_KEYS, (clipped, boot_ci, nested, seed_t, source)):
            self.ci[key][field] = value
        self.support[field] = None if support is None else list(support)
        self.by_seed[field] = per_seed_values(groups, shown or stat)
        return point

    def exact(self, field: str, value, source: str = "cohort census") -> None:
        """A census value, an extreme or a count: no interval (ci95_source says which)."""
        self.fields[field] = value
        for key in CI_KEYS:
            self.ci[key][field] = None
        self.ci["ci95_source"][field] = source

    def to_dict(self) -> dict:
        return {**self.fields, **self.extra, "by_seed": self.by_seed, **self.ci,
                "ci95_unclipped": self.unclipped, "ci95_support": self.support}
