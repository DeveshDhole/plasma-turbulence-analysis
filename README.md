# plasma-turbulence-analysis

Analysis code for my PhD on current sheets and magnetic reconnection in kinetic plasma
turbulence (CAMELIA hybrid-PIC simulations, DiRAC COSMA). The reusable part is the Python
package `pta` in `src/pta/`; the research notebooks in `projects/` use it.

## Layout

```
src/pta/               the pta package: everything reusable (import pta)
  run.py               Run: one simulation's output folder (output times, grid spacing, load, map over times)
  config.py            where data, results and caches are, read from pta.toml
  fields.py            spectral and finite-difference derivatives, current density
  spectra.py           radial spectra, summed spectra, the B, J and n spectra at one time
  filtering.py         iterative filtering (fifpy), incl. the k-cutoff version; Gaussian low-pass filter
  sheets/              current sheets: detection.py, thickness.py
  statistics.py        lognormal fits, bootstrap and PDF helpers
  parallel.py          parallel_map: one function over many items, in parallel processes
  plotting.py          plot style: inferno (magnitudes), RdBu_r (signed fields), gray (masks), save_figure at 300 dpi
  utils.py             small array helpers
tests/                 tests of the package: python -m pytest tests
pyproject.toml         installs the package: pip install -e .
pta.toml               this project's data locations, run layouts and sets of runs (read by pta.config)
projects/              one folder per piece of work (code only)
  cs_identification/   current-sheet detection, thickness methods, beta scans, Luca's runs
  paper1_methods/      Paper 1 (method paper, beta_p = beta_e = 1)
  single_sheet/        single current-sheet analyses
  jpar_jperp/          J_parallel / J_perp analyses and spectra
  quality_checks/      energy conservation, spectra, SNR, resistivity checks
  experiments/         filtering experiments (Gaussian, iterative filtering)
  apj2015/             ApJ 2015 reproduction
  dirac_images/        DiRAC image competition 2026 (finished)
shared/                temporary: the old module names, now served from pta (see "Moving to pta")
  filters.py           filtering module (used by experiments/FilteringCheckEmanuele)
  current_sheets_id_char_collisionless_turbulence.py   current-sheet identification/characterisation module
results/               outputs, mirroring projects/ (git-ignored); where notebooks of one project would
                       overwrite each other's files, each writes to results/<project>/<notebook>/
cache/                 regenerable intermediate files (git-ignored; safe to delete)
archive/               old versions kept for reference (not used any more; MOVES.md says what is where)
  cs_detection_development/          history of the detection pipeline
  luca_simulation_analysis_per_run/  Luca's five per-run notebooks (replaced by one notebook)
  jpar_jperp_spectra/                the three spectrum notebooks (replaced by JparJperpSpectrum)
  quality_checks_superseded/         older and merged quality-check notebooks
  apj2015/                           ApJ2015-Copy1 (its one new cell is now in ApJ2015)
  shared/                            the shared modules that pta replaced
  py_exports/                        .py exports of notebooks that are kept as notebooks
  dirac_superseded/                  older DiRAC script versions
  to_delete/                         safe to delete
MOVES.md               record of the October 2026 reorganisation (old -> new paths)
```

## Installing

```bash
cd plasma-turbulence-analysis
pip install -e ".[filtering,parallel,test]"     # editable: changes in src/pta take effect at once
python -m pytest tests
```

The install name is `plasma-turbulence-analysis` (`pta` is taken on PyPI); the package is imported as `pta`.
The extras add iterativefiltering, which provides Emanuele Papini's `fifpy`, and pyfftw (`filtering`: iterative
filtering), threadpoolctl (`parallel`: one numpy thread per worker process) and pytest (`test`).

## Using pta

```python
import pta
from pta import config
from pta.spectra import summed_spectrum

run = config.run("beta_scan", beta_p=1)             # a pta.Run, from the run layouts in pta.toml
# run = pta.Run("/path/to/any/run")                # or any folder of CAMELIA output
Bx, By = run.load_many(["Bx", "By"], 200)           # x along axis 0
k, spectrum = summed_spectrum((Bx, By), run.dx, run.dy)

def B_perp_rms(time):
    Bx, By = run.load_many(["Bx", "By"], time)
    return ((Bx**2 + By**2).mean()) ** 0.5

series = run.map(B_perp_rms)                        # every output time, in parallel processes, in time order
RESULTS = config.results_dir("<project>", "<notebook>")
```

Output of other codes, or other file names: `pta.Run(folder, pattern="{field}_{time:04d}.h5", dataset=None,
transpose=False, box_length=(Lx, Ly))`.

## Where things are: pta.toml

`pta.toml` says where this project's data, results and caches are, gives run folders by name and lists
named sets of runs. Each `[paths]` entry can be overridden on a machine with an environment variable
`PTA_<NAME>`:

| entry | environment variable | default here | contents |
|---|---|---|---|
| `nas` | `PTA_NAS` | `/home/UNN/w24021992/NAS` | the workstation's NAS |
| `simulations` | `PTA_SIMULATIONS` | `{nas}/DATA` | simulation outputs: `Brms_<B_rms>/beta_p_<beta_p>/beta_e_<beta_e>/`, `high_res`, `Particle` |
| `external_data` | `PTA_EXTERNAL_DATA` | `/DATA` | Luca's QMUL runs, the Nancy runs |
| `devesh` | `PTA_DEVESH` | `{external_data}/DEVESH` | our runs on the /DATA disk (resistivity and extreme-beta checks) |
| `results` | `PTA_RESULTS` | `results/` | outputs |
| `cache` | `PTA_CACHE` | `cache/` | regenerable intermediate files |

`[runs]` gives run folders by name (`config.run("beta_scan", beta_p=4, beta_e=16)`), `[run_defaults]` their
default parameters (`brms = 0.24`, `beta_e = 1`), and `[scans]` named sets of runs with legend labels
(`config.scan("resistivity")`). `PTA_CONFIG` can point to a different `pta.toml`.

## Moving to pta

The notebooks are being moved from the old `shared` modules to `import pta`, project by project (MOVES.md
records each step). Until it is moved, a notebook keeps its old header, which finds the repository and
imports from `shared`; `shared/` now only serves the old names from pta (and the paths from pta.toml), so
those notebooks work unchanged, even before pta is installed. A moved notebook needs only `import pta`.

## Rules

- Results live in `results/` (or wherever PTA_RESULTS points), never next to the code or the data; only small
  final data a paper depends on is kept with the code, inside that paper's folder.
- Data paths only through `pta.toml` and `pta.config` (no hard-coded `/home/UNN/...` or `/DATA/...`).
- Caches are disposable: anything that rerunning the code recreates can be deleted.
- One copy of each function, in the pta package (`src/pta/`) with a test in `tests/`; superseded versions go to
  `archive/` (or git history), not `_v2` / `_new` / `-Copy1`. A deliberately different version stays in its
  notebook, with a note in MOVES.md.
- Every figure is saved, not just shown: `save_figure(fig, RESULTS, "descriptive_name")` writes it at 300 dpi to
  `results/<project>/<notebook>/`; colour maps and line colours come from `pta.plotting` (`use_style()`).
- Nothing tied to a particular run is typed into the analysis: files are read through a `pta.Run`, which gives the
  output times (`run.times`) and the grid spacing (`run.dx`, `run.dy`); choices (which time, which scales) are
  named settings at the top of the notebook.
- The same operation at every output time runs in parallel: write it as a function of the time and use `run.map`.
- Notebook outputs are stripped before committing (`nbstripout`).
- Commit often with descriptive messages; tag paper versions (e.g. `paper1-submitted`).
