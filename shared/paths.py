"""Temporary: the old shared.paths names, now read from pta.toml (see pta.config). Notebooks moved to the pta
package use pta.config instead; this file goes with the rest of shared/ once none use it."""
import os
from pathlib import Path

from pta import config

_paths = config.paths()
REPO_ROOT = config.project_root()
NAS_ROOT = _paths["nas"]
SIM_ROOT = _paths["simulations"]
BRMS = config.settings().get("run_defaults", {}).get("beta_scan", {}).get("brms", 0.24)
DATA_ROOT = Path(os.environ["PTA_DATA"]) if os.environ.get("PTA_DATA") else SIM_ROOT / f"Brms_{BRMS}"
EXTERNAL_DATA_ROOT = _paths["external_data"]
RESULTS_ROOT = _paths["results"]
CACHE_ROOT = _paths["cache"]
results_dir = config.results_dir
cache_dir = config.cache_dir


def run_dir(beta_p, beta_e=1, brms=BRMS):
    """Folder of one run of the beta scans: NAS/DATA/Brms_<brms>/beta_p_<beta_p>/beta_e_<beta_e>"""
    base = DATA_ROOT if brms == BRMS else SIM_ROOT / f"Brms_{brms}"
    return base / f"beta_p_{beta_p}" / f"beta_e_{beta_e}"


def external_run(*parts):
    """Folder of a run on the /DATA disk: external_run("ResistivityCheck", "2.5e-4_2048")"""
    return _paths["devesh"].joinpath(*map(str, parts))
