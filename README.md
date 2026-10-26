# plasma-turbulence-analysis

Analysis code for my PhD on current sheets and magnetic reconnection in kinetic plasma
turbulence (CAMELIA hybrid-PIC simulations, DiRAC COSMA).

## Layout

```
projects/              one folder per piece of work (code only)
  cs_identification/   current-sheet detection, thickness methods, beta scans, Luca's runs
  paper1_methods/      Paper 1 (method paper, beta_p = beta_e = 1)
  single_sheet/        single current-sheet analyses
  jpar_jperp/          J_parallel / J_perp analyses and spectra
  quality_checks/      energy conservation, spectra, SNR, resistivity checks
  experiments/         filtering experiments (Gaussian, iterative filtering)
  apj2015/             ApJ 2015 reproduction
  dirac_images/        DiRAC image competition 2026 (finished)
shared/                code used by several projects (import from here, don't copy)
  paths.py             where data, results and caches live (one place to change)
  fields.py            spectral derivatives, radial spectra
  filtering.py         iterative filtering (fifpy), incl. the k-cutoff version
  detection.py         current-sheet detection
  thickness.py         thickness-method helpers (production versions)
  stats.py             statistics helpers
  loading.py           reading CAMELIA output: load_field("Bx", t, run, tag="ApJ"), load_fields([...], t, run)
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
  py_exports/                        .py exports of notebooks that are kept as notebooks
  dirac_superseded/                  older DiRAC script versions
  to_delete/                         safe to delete
MOVES.md               record of the October 2026 reorganisation (old -> new paths)
```

## Using the shared code

Every notebook starts with this block (it finds the repository from the working directory):

```python
import sys
from pathlib import Path
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "shared" / "paths.py").exists())
sys.path.insert(0, str(REPO))
from shared.fields import partial_x_fft, partial_y_fft
from shared.filtering import iterative_filtering1DWithKCutOff
from shared.detection import detect_current_sheets
from shared.paths import DATA_ROOT, results_dir, cache_dir
RESULTS = results_dir("<project>")            # or results_dir("<project>", "<notebook>")
CACHE = cache_dir("<project>")
```

Scripts use the same block with `Path(__file__).resolve().parents` in place of the working-directory list,
so they run from any directory.

Where things live is set in `shared/paths.py` and can be overridden per machine:

| setting | environment variable | default | contents |
|---|---|---|---|
| `NAS_ROOT` | `PTA_NAS` | `/home/UNN/w24021992/NAS` | the workstation's NAS (e.g. `DATA/Brms_0.24`, `DATA/high_res`) |
| `DATA_ROOT` | `PTA_DATA` | `NAS_ROOT/simulations_beta_scan` | the beta scan (`run_dir(beta_p, beta_e)`) |
| `EXTERNAL_DATA_ROOT` | `PTA_EXTERNAL_DATA` | `/DATA` | Luca's QMUL runs, the Nancy runs, `DEVESH/` archives |
| `RESULTS_ROOT` | `PTA_RESULTS` | `results/` in the repository | outputs |
| `CACHE_ROOT` | `PTA_CACHE` | `cache/` in the repository | regenerable intermediate files |

## Rules

- Results live in `results/` (or wherever PTA_RESULTS points), never next to the code or the data; only small
  final data a paper depends on is kept with the code, inside that paper's folder.
- Data paths only through `shared.paths` (no hard-coded `/home/UNN/...` or `/DATA/...`).
- Caches are disposable: anything that rerunning the code recreates can be deleted.
- One copy of each function, in `shared/`; superseded versions go to `archive/` (or git history),
  not `_v2` / `_new` / `-Copy1`. A deliberately different version stays in its notebook, with a note in MOVES.md.
- Notebook outputs are stripped before committing (`nbstripout`).
- Commit often with descriptive messages; tag paper versions (e.g. `paper1-submitted`).
