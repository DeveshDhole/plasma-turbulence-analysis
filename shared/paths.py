"""
Where simulation data, results and caches live. Every notebook and script takes its
paths from here, so moving a folder means changing one line (or one environment variable).

  NAS_ROOT      the workstation's NAS (other datasets, e.g. NAS_ROOT/DATA/Brms_0.24)
  DATA_ROOT     simulation outputs of the beta scan (HDF5), NAS_ROOT/simulations_beta_scan
  EXTERNAL_DATA_ROOT  the /DATA disk with collaborators' simulations (Luca's QMUL runs, the Nancy runs)
  RESULTS_ROOT  results (CSVs, figures, frames); default <repository>/results
  CACHE_ROOT    regenerable intermediate files; default <repository>/cache

The environment variables PTA_NAS, PTA_DATA, PTA_EXTERNAL_DATA, PTA_RESULTS and PTA_CACHE override the defaults,
e.g. `export PTA_RESULTS=/somewhere/else/results`.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NAS_ROOT = Path(os.environ.get("PTA_NAS", "/home/UNN/w24021992/NAS"))
DATA_ROOT = Path(os.environ.get("PTA_DATA", NAS_ROOT / "simulations_beta_scan"))
EXTERNAL_DATA_ROOT = Path(os.environ.get("PTA_EXTERNAL_DATA", "/DATA"))
RESULTS_ROOT = Path(os.environ.get("PTA_RESULTS", REPO_ROOT / "results"))
CACHE_ROOT = Path(os.environ.get("PTA_CACHE", REPO_ROOT / "cache"))


def run_dir(beta_p, beta_e=1):
    """Folder of one simulation of the beta scan, e.g. run_dir(1) -> .../beta_p_1/beta_e_1
    (the same naming as the old f"{base}/beta_p_{beta}/beta_e_1/" strings)."""
    return DATA_ROOT / f"beta_p_{beta_p}" / f"beta_e_{beta_e}"


def results_dir(*parts, create=True):
    """results_dir("paper1_methods", "cs_thickness_pooled_k") -> RESULTS_ROOT/paper1_methods/cs_thickness_pooled_k"""
    p = RESULTS_ROOT.joinpath(*map(str, parts))
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir(*parts, create=True):
    """cache_dir("cs_identification", "filtered") -> CACHE_ROOT/cs_identification/filtered"""
    p = CACHE_ROOT.joinpath(*map(str, parts))
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
