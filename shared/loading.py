"""Reading CAMELIA output files.

Every field snapshot is one HDF5 file, <run>/<field>_<tag>_t<t>.h5, with the data in the dataset "DS1".
The notebooks all use the transposed array (DS1.T), so that axis 0 is x; these functions do the same.

    from shared.loading import load_field, load_fields
    from shared.paths import run_dir

    Bx = load_field("Bx", 200, run_dir(1))                       # beta scan: beta_p = 1, beta_e = 1
    Bx, By, Bz = load_fields(["Bx", "By", "Bz"], 200, run_dir(0.0625))
    n = load_field("Dn1", 48, f"{EXTERNAL_DATA_ROOT}/QMUL/...", tag="EL4modes")   # another run and tag
"""
from pathlib import Path

import h5py


def field_file(field, t, run, tag="ApJ"):
    """Path of one snapshot file: <run>/<field>_<tag>_t<t>.h5"""
    return Path(run) / f"{field}_{tag}_t{t}.h5"


def load_field(field, t, run, tag="ApJ", transpose=True):
    """Read one field snapshot.

    field     : field name as in the file name, e.g. "Bx", "Ez", "Dn1", "Ux1", "Ppar1"
    t         : output time as in the file name (200 -> ..._t200.h5)
    run       : run directory, e.g. run_dir(1) or f"{DATA_ROOT}/beta_p_1/beta_e_1"
    tag       : simulation tag in the file name ("ApJ" for the beta scan)
    transpose : return DS1.T, the convention used in all the notebooks (axis 0 = x)
    """
    path = field_file(field, t, run, tag)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (check the run directory, field name, tag and time)")
    with h5py.File(path, "r") as f:
        data = f["DS1"][:]
    return data.T if transpose else data


def load_fields(fields, t, run, tag="ApJ", transpose=True):
    """Read several fields of one snapshot, e.g. Bx, By, Bz = load_fields(["Bx", "By", "Bz"], 200, run)."""
    return tuple(load_field(f, t, run, tag, transpose) for f in fields)


load = load_field          # short name
