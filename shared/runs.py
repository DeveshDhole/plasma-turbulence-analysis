"""Named sets of runs used by several notebooks: one place to add, remove or relabel a run.

    from shared.runs import RESISTIVITY_SCAN, runs_with_labels
    for run, label in runs_with_labels(RESISTIVITY_SCAN): ...
"""
from .paths import external_run

# runs in EXTERNAL_DATA_ROOT/DEVESH/ResistivityCheck/: folder name -> legend label
RESISTIVITY_SCAN = {
    "0.625e-4": "η = 0.625×10⁻⁴",
    "1.25e-4": "η = 1.25×10⁻⁴",
    "2.5e-4_2048": "η = 2.5×10⁻⁴",
    "5e-4": "η = 5×10⁻⁴",
}

# the eta = 2.5e-4 run with different numbers of particles per cell: folder name -> legend label
PARTICLES_PER_CELL_SCAN = {
    "2.5e-4_256": "256 ppc",
    "2.5e-4_512": "512 ppc",
    "2.5e-4_1024": "1024 ppc",
    "2.5e-4_2048": "2048 ppc",
}

# runs in EXTERNAL_DATA_ROOT/DEVESH/ExtremeBetaSimulations/: folder name (beta) -> legend label
EXTREME_BETA_SCAN = {
    "16": "16",
    "0.0625": "0.0625",
}


def runs_with_labels(scan, group="ResistivityCheck"):
    """[(run folder, legend label), ...] for one of the scans above."""
    return [(external_run(group, name), label) for name, label in scan.items()]
