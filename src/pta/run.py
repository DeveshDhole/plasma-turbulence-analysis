"""Simulation runs: a folder of output files, one HDF5 file per field and output time.

    from pta import Run
    run = Run("/path/to/run")               # CAMELIA: <field>_ApJ_t<time>.h5, data in the dataset "DS1"
    run.times                               # the output times present in the folder
    run.dx, run.dy                          # grid spacing in d_p: box length / number of cells
    Bx = run.load("Bx", 200)                # a 2D array with x along axis 0
    Bx, By = run.load_many(["Bx", "By"], 200)
    energy = run.history("EnTot")           # a time history (EnTot_ApJ_.h5)
    rms = run.map(lambda time: run.load("Bx", time).std())    # the same analysis at every time, in parallel

Other codes or file names: Run(path, pattern="{field}_{time:04d}.h5", dataset=None, transpose=False, ...).
The functions at the end (load, output_times, grid_spacing, ...) do the same for a folder given as a path.
"""
import re
from pathlib import Path

import h5py
import numpy as np

from .parallel import parallel_map

BOX_LENGTH = 256.0          # d_p: side of the square box of the CAMELIA runs (2048 x 0.125, 4096 x 0.0625)


class Run:
    """One simulation's output folder.

    path            : the folder
    tag             : simulation tag in the file names ("ApJ")
    pattern         : name of the file of one field at one time; {field}, {time} and {tag} are filled in
    history_pattern : name of a time-history file; {quantity} and {tag} are filled in
    dataset         : dataset inside each field file (None: the first one)
    box_length      : side of the box in d_p, one number or (Lx, Ly); sets dx and dy
    transpose       : the files store arrays as (y, x); True returns them with x along axis 0
    reference_field : the field whose files give the output times and the grid
    """

    def __init__(self, path, tag="ApJ", pattern="{field}_{tag}_t{time}.h5", history_pattern="{quantity}_{tag}_.h5",
                 dataset="DS1", box_length=BOX_LENGTH, transpose=True, reference_field="Bx"):
        self.path = Path(path)
        self.tag, self.pattern, self.history_pattern = tag, pattern, history_pattern
        self.dataset, self.box_length, self.transpose = dataset, box_length, transpose
        self.reference_field = reference_field
        self._times, self._shape = {}, None

    def __repr__(self):
        return f"Run({str(self.path)!r}, tag={self.tag!r})"

    def __fspath__(self):                           # a Run can be used wherever a folder path is expected
        return str(self.path)

    @property
    def name(self):
        return self.path.name

    @property
    def parent(self):
        return self.path.parent

    # ---- files and times ----
    def file(self, field, time):
        """Path of the file of one field at one output time."""
        return self.path / self.pattern.format(field=field, time=time, tag=self.tag)

    def output_times(self, field=None):
        """Every output time with a file for `field` (default: the reference field), sorted."""
        field = field or self.reference_field
        if field not in self._times:
            marked = re.sub(r"\{time(?::[^}]*)?\}", "\0", self.pattern)
            before, after = marked.format(field=field, tag=self.tag).split("\0")
            file_name = re.compile(re.escape(before) + r"(\d+)" + re.escape(after) + r"\Z")
            times = sorted(int(match.group(1)) for entry in self.path.iterdir()
                           if (match := file_name.match(entry.name)))
            if not times:
                raise FileNotFoundError(f"no {before}<time>{after} files in {self.path}")
            self._times[field] = np.array(times)
        return self._times[field]

    @property
    def times(self):
        """Every output time present in the folder (from the reference field's files)."""
        return self.output_times()

    # ---- grid ----
    @property
    def shape(self):
        """(nx, ny): the shape of the arrays that load returns."""
        if self._shape is None:
            with h5py.File(self.file(self.reference_field, self.times[0]), "r") as f:
                stored = self._dataset(f).shape
            self._shape = tuple(reversed(stored)) if self.transpose else tuple(stored)
        return self._shape

    @property
    def dx(self):
        return self._box_lengths()[0] / self.shape[0]

    @property
    def dy(self):
        return self._box_lengths()[1] / self.shape[1]

    def _box_lengths(self):
        return tuple(self.box_length) if np.ndim(self.box_length) else (self.box_length, self.box_length)

    # ---- data ----
    def _dataset(self, handle):
        return handle[self.dataset] if self.dataset else handle[next(iter(handle.keys()))]

    def load(self, field, time):
        """One field at one output time, as a 2D array (x along axis 0 when transpose is True)."""
        path = self.file(field, time)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (check the run, field name, tag and time)")
        with h5py.File(path, "r") as f:
            data = self._dataset(f)[:]
        return data.T if self.transpose else data

    def load_many(self, fields, time):
        """Several fields at one output time: Bx, By = run.load_many(["Bx", "By"], 200)."""
        return tuple(self.load(field, time) for field in fields)

    def history(self, quantity):
        """A time history written by the code, e.g. "EnTot" or "Time" (the file's first dataset, as stored)."""
        return load_dataset(self.path / self.history_pattern.format(quantity=quantity, tag=self.tag))

    # ---- the same analysis at many output times ----
    def map(self, function, times=None, workers=None, threads_per_worker=1):
        """[function(time) for time in times] (default: every output time), computed in parallel processes
        and returned in time order. See pta.parallel.parallel_map for workers and threads_per_worker."""
        return parallel_map(function, self.times if times is None else times, workers=workers,
                            threads_per_worker=threads_per_worker)


def as_run(run, **options):
    """run itself if it is a Run, otherwise Run(run, **options) for a folder path."""
    return run if isinstance(run, Run) else Run(run, **options)


def field_file(field, time, run, tag="ApJ"):
    """Path of one field file of the run folder: <run>/<field>_<tag>_t<time>.h5"""
    return as_run(run, tag=tag).file(field, time)


def load(field, time, run, tag="ApJ", transpose=True):
    """One field at one output time of a run (a Run or a folder), x along axis 0."""
    return as_run(run, tag=tag, transpose=transpose).load(field, time)


def load_fields(fields, time, run, tag="ApJ", transpose=True):
    """Several fields at one output time: Bx, By, Bz = load_fields(["Bx", "By", "Bz"], 200, run)."""
    return as_run(run, tag=tag, transpose=transpose).load_many(fields, time)


def load_dataset(path):
    """The first dataset of an HDF5 file, as stored (for time histories such as EnTot_ApJ_.h5)."""
    with h5py.File(path, "r") as f:
        return f[next(iter(f.keys()))][:]


def load_history(quantity, run, tag="ApJ"):
    """A time history of a run, e.g. load_history("EnTot", run) reads <run>/EnTot_ApJ_.h5."""
    return as_run(run, tag=tag).history(quantity)


def output_times(run, field="Bx", tag="ApJ"):
    """Every output time present in a run folder, sorted."""
    return as_run(run, tag=tag).output_times(field)


def grid_spacing(run, field="Bx", tag="ApJ", box_length=BOX_LENGTH):
    """(dx, dy) of a run in d_p: the box length divided by the number of cells."""
    run = as_run(run, tag=tag, box_length=box_length, reference_field=field)
    return run.dx, run.dy
