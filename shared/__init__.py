"""Temporary compatibility layer: the old `shared` modules, now provided by the pta package.

Notebooks that still say `from shared.<module> import ...` keep working while they are moved to
`import pta` (see MOVES.md). The paths, runs and scans now come from pta.toml (shared/paths.py
is a real file because the notebooks look for it to find the repository). This folder goes
once no notebook uses it.
"""
import importlib
import sys
import types
from pathlib import Path

try:
    import pta
except ImportError:                          # pta not installed yet: use the copy in this repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import pta
from pta import config
from pta import run as _run


def _module(name, sources=(), **extra):
    """Register shared.<name> as the names of the given pta modules (plus extra)."""
    module = types.ModuleType(f"{__name__}.{name}")
    for source in sources:
        module.__dict__.update({key: value for key, value in vars(importlib.import_module(source)).items()
                                if not key.startswith("__")})
    module.__dict__.update(extra)
    sys.modules[module.__name__] = module
    globals()[name] = module


from .paths import BRMS, external_run, run_dir      # shared/paths.py: the old path names, from pta.toml


# ---- shared.loading: pta.run, plus the beta_p=... shortcut ----
def load_field(field, t, run=None, tag="ApJ", transpose=True, *, beta_p=None, beta_e=1, brms=BRMS):
    """One field snapshot of a run folder, or of the beta-scan run given by beta_p (and beta_e, brms)."""
    if run is None:
        if beta_p is None:
            raise ValueError("give the run folder, or beta_p (and optionally beta_e and brms)")
        run = run_dir(beta_p, beta_e, brms)
    return _run.load(field, t, run, tag=tag, transpose=transpose)


def load_fields(fields, t, run=None, tag="ApJ", transpose=True, **run_keys):
    return tuple(load_field(field, t, run, tag, transpose, **run_keys) for field in fields)


_module("loading", ["pta.run"], load_field=load_field, load=load_field, load_fields=load_fields)


# ---- shared.runs: the scans, from pta.toml ----
def _scan(name):
    return {item["name"]: item.get("label", item["name"])
            for item in config.settings().get("scans", {}).get(name, {}).get("runs", [])}


def runs_with_labels(scan, group="ResistivityCheck"):
    """[(run folder, legend label), ...] for one of the scans."""
    return [(external_run(group, name), label) for name, label in scan.items()]


_module("runs", RESISTIVITY_SCAN=_scan("resistivity"), PARTICLES_PER_CELL_SCAN=_scan("particles_per_cell"),
        EXTREME_BETA_SCAN=_scan("extreme_beta"), runs_with_labels=runs_with_labels)

# ---- the other modules moved to pta unchanged ----
_module("fields", ["pta.fields", "pta.spectra"])
_module("filtering", ["pta.filtering"])
_module("detection", ["pta.sheets.detection"])
_module("thickness", ["pta.sheets.thickness"])
_module("stats", ["pta.statistics"])
_module("plotting", ["pta.plotting"])
_module("parallel", ["pta.parallel"])
_module("utils", ["pta.utils"])
