"""Reading CAMELIA output files.

Every field snapshot is one HDF5 file, <run>/<field>_<tag>_t<t>.h5, with the data in the dataset "DS1".
The notebooks all use the transposed array (DS1.T), so that axis 0 is x; these functions do the same.

    from shared.loading import load

    Bx = load("Bx", 200, beta_p=1)                               # NAS/DATA/Brms_0.24/beta_p_1/beta_e_1
    Bx = load("Bx", 200, beta_p=1, beta_e=4, brms=0.12)          # another run of the scan
    Bx = load("Bx", 200, run_dir(1))                             # or give the run folder itself
    Bx, By, Bz = load_fields(["Bx", "By", "Bz"], 200, beta_p=0.0625)
    n = load("Dn1", 48, f"{EXTERNAL_DATA_ROOT}/QMUL/...", tag="EL4modes")   # another simulation tag
"""
from pathlib import Path

import h5py
import numpy as np

from .paths import BRMS, run_dir


def field_file(field, t, run, tag="ApJ"):
    """Path of one snapshot file: <run>/<field>_<tag>_t<t>.h5"""
    return Path(run) / f"{field}_{tag}_t{t}.h5"


def load_field(field, t, run=None, tag="ApJ", transpose=True, *, beta_p=None, beta_e=1, brms=BRMS):
    """Read one field snapshot.

    field     : field name as in the file name, e.g. "Bx", "Ez", "Dn1", "Ux1", "Ppar1"
    t         : output time as in the file name (200 -> ..._t200.h5)
    run       : run folder; or leave it out and give beta_p (and beta_e, brms) instead
    tag       : simulation tag in the file name ("ApJ" for the beta scans)
    transpose : return DS1.T, the convention used in all the notebooks (axis 0 = x)
    beta_p, beta_e, brms : the run of the beta scans, NAS/DATA/Brms_<brms>/beta_p_<beta_p>/beta_e_<beta_e>
    """
    if run is None:
        if beta_p is None:
            raise ValueError("give the run folder, or beta_p (and optionally beta_e and brms)")
        run = run_dir(beta_p, beta_e, brms)
    path = field_file(field, t, run, tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (check the run, field name, tag and time)")
    with h5py.File(path, "r") as f:
        data = f["DS1"][:]
    return data.T if transpose else data


def load_fields(fields, t, run=None, tag="ApJ", transpose=True, **run_keys):
    """Read several fields of one snapshot: Bx, By, Bz = load_fields(["Bx", "By", "Bz"], 200, beta_p=1)."""
    return tuple(load_field(f, t, run, tag, transpose, **run_keys) for f in fields)


def load_dataset(fp):
    """First dataset of an HDF5 file, untransposed (energy and time histories such as EnTot_ApJ_.h5)."""
    with h5py.File(fp, "r") as f:
        key = list(f.keys())[0]        # get first dataset name, e.g. 'DS1'
        return f[key][:]


def load_history(quantity, run, tag="ApJ"):
    """A time history written by CAMELIA, e.g. load_history("EnTot", run) reads <run>/EnTot_ApJ_.h5
    (other quantities: "Time", "EnMa", ...)."""
    return load_dataset(Path(run) / f"{quantity}_{tag}_.h5")


BOX_LENGTH = 256.0         # d_p: side of the square box in all the CAMELIA runs here (2048 x 0.125, 4096 x 0.0625)


def output_times(run, field="Bx", tag="ApJ"):
    """Every output time present in the run folder, sorted (from the <field>_<tag>_t<t>.h5 file names)."""
    times = sorted(int(p.stem.rsplit("_t", 1)[1]) for p in Path(run).glob(f"{field}_{tag}_t*.h5")
                   if p.stem.rsplit("_t", 1)[1].isdigit())
    if not times:
        raise FileNotFoundError(f"no {field}_{tag}_t*.h5 files in {run}")
    return np.array(times)


def grid_spacing(run, field="Bx", tag="ApJ", box_length=BOX_LENGTH):
    """(dx, dy) in d_p: the box length divided by the number of cells in the run's output files."""
    first = Path(run) / f"{field}_{tag}_t{output_times(run, field, tag)[0]}.h5"
    with h5py.File(first, "r") as f:
        ny_cells, nx_cells = f["DS1"].shape      # as stored; load() transposes to (x, y)
    return box_length / nx_cells, box_length / ny_cells


load = load_field          # short name
