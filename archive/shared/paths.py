"""
Where simulation data, results and caches live. Every notebook and script takes its
paths from here, so moving a folder means changing one line (or one environment variable).

  NAS_ROOT      the workstation's NAS
  SIM_ROOT      simulation outputs on the NAS, NAS_ROOT/DATA; the beta scans are in
                SIM_ROOT/Brms_<B_rms>/beta_p_<beta_p>/beta_e_<beta_e>/
  BRMS          B_rms of the main beta scan (0.24)
  DATA_ROOT     the main beta scan, SIM_ROOT/Brms_0.24 (until October 2026: NAS_ROOT/simulations_beta_scan)
  EXTERNAL_DATA_ROOT  the /DATA disk with collaborators' simulations (Luca's QMUL runs, the Nancy runs)
  RESULTS_ROOT  results (CSVs, figures, frames); default <repository>/results
  CACHE_ROOT    regenerable intermediate files; default <repository>/cache

The environment variables PTA_NAS, PTA_SIM, PTA_DATA, PTA_EXTERNAL_DATA, PTA_RESULTS and PTA_CACHE
override the defaults, e.g. `export PTA_RESULTS=/somewhere/else/results`.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NAS_ROOT = Path(os.environ.get("PTA_NAS", "/home/UNN/w24021992/NAS"))
SIM_ROOT = Path(os.environ.get("PTA_SIM", NAS_ROOT / "DATA"))
BRMS = 0.24
DATA_ROOT = Path(os.environ.get("PTA_DATA", SIM_ROOT / f"Brms_{BRMS}"))
EXTERNAL_DATA_ROOT = Path(os.environ.get("PTA_EXTERNAL_DATA", "/DATA"))
RESULTS_ROOT = Path(os.environ.get("PTA_RESULTS", REPO_ROOT / "results"))
CACHE_ROOT = Path(os.environ.get("PTA_CACHE", REPO_ROOT / "cache"))


def run_dir(beta_p, beta_e=1, brms=BRMS):
    """Folder of one simulation:
    run_dir(1)          -> NAS/DATA/Brms_0.24/beta_p_1/beta_e_1
    run_dir(1, 1, 0.12) -> NAS/DATA/Brms_0.12/beta_p_1/beta_e_1"""
    base = DATA_ROOT if brms == BRMS else SIM_ROOT / f"Brms_{brms}"
    return base / f"beta_p_{beta_p}" / f"beta_e_{beta_e}"


def external_run(*parts):
    """Folder of a run on the /DATA disk: external_run("ResistivityCheck", "2.5e-4_2048")
    -> EXTERNAL_DATA_ROOT/DEVESH/ResistivityCheck/2.5e-4_2048"""
    return EXTERNAL_DATA_ROOT.joinpath("DEVESH", *map(str, parts))


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
