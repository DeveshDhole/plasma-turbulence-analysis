# Reorganisation record (24 October 2026)

Moves, plus a new `shared/` package. No existing file contents were changed and nothing was deleted.
Use this list to fix paths in notebooks and scripts (old -> new).

## Folders

| Old | New |
|---|---|
| `CSIdentification/` | `projects/cs_identification/` |
| `Paper1/` | `projects/paper1_methods/` |
| `SingleCurrentSheetAnalysis/` | `projects/single_sheet/` |
| `JparJperpAnalysis/` | `projects/jpar_jperp/` |
| `QualityCheck/` | `projects/quality_checks/` |
| `Experiments/` | `projects/experiments/` |
| `ApJ2015/` | `projects/apj2015/` |
| `Dirac Image Competition/` | `projects/dirac_images/` |

## Files from the top level

| Old | New |
|---|---|
| `filters.py` | `shared/filters.py` |
| `current_sheets_id_char_collisionless_turbulence.py` | `shared/current_sheets_id_char_collisionless_turbulence.py` |
| `ApJ2015-Copy1.ipynb` | `projects/apj2015/ApJ2015-Copy1.ipynb` |
| `EnergyPlotsforDifferentBeta.ipynb`, `EnergyPlotsforDifferentPpc.ipynb`, `EnergyfromComponents.ipynb`, `KineticEnergyforDifferentResistivity.ipynb`, `MagneticEnergyforDifferentResistivity.ipynb`, `ResistivityCheck.ipynb`, `SpectrumCompareBeta.ipynb`, `SimulationHeatMap.ipynb` | `projects/quality_checks/` |
| `FilteringCheck.ipynb`, `FilteringCheckEmanuele.ipynb`, `IterativeFilteringIndividualComponents.ipynb` | `projects/experiments/` |

To `archive/cs_detection_development/` (development history of the detection pipeline):
`CSDetectionClean.ipynb`, `CSDetectionClean_HR.ipynb`, `CSDetectionImage.ipynb`, `CodeDump.ipynb`,
`CompareEmanuele.ipynb`, `CompareTwoImages.ipynb`, `CurrentSheetDetection.ipynb`,
`J2_CSDetectionClean.ipynb`, `J2_CSDetectionClean.py`, `J2_CSDetectionCleanGB.ipynb`,
`J2_CSDetectionCleanHR.ipynb`, `J2_CSDetectionCleanHR.py`, `J2_CSDetectionCleanHR_Turbulent.ipynb`,
`J2_CSDetectionCleanMethod5and6.ipynb`, `J2_CSDetectionCleanNR.ipynb`,
`J2_CSDetectionCleanNRMultipleSnapshots.ipynb`, `J2_CSDetectionCleanNRMultipleSnapshotsBetaComparison.ipynb`,
`J2_CSDetectionCleanNR_Circle.ipynb`, `J2_CSDetectionCleanNR_CompareBeta.ipynb`,
`J2_CSDetectionCleanNR_Turbulent.ipynb`, `J2_CSDetectionCleanResistivity(Method3).ipynb`,
`J_par_CSDetectionClean.ipynb`, `J_par_CSDetectionClean_CompareBeta.ipynb`, `Jperp_CSDetectionCleanNR.ipynb`.

## Inside the DiRAC project (`projects/dirac_images/`)

Superseded versions, to `archive/dirac_superseded/`: `video.py`, `event_videos.py`,
`gallery_200.py`, `gallery_new.py`, `final.py`, `iterative_filtering.py` (empty), `Untitled.ipynb`.

Test copies and markers, to `archive/to_delete/` (delete these yourself when happy):
`clock_and_rings_64.py`, `clock_and_rings_128.py`, `ht_test_64/`, `ht_test_128/`,
`events_bp1_be1/videos_zoom48/frames/_done/` -> `videos_zoom48_frames_done/`,
`events_bp1_be1/videos_zoom96/frames/_done/` -> `videos_zoom96_frames_done/`.

Final DiRAC scripts kept in `projects/dirac_images/`: `clock_and_rings.py`, `event_composite.py`,
`event_videos_v2.py`, `find_xpoint.py`, `gallery.py`, `label_new.py`, `neon.py`, `palette.py`,
`particle.py`, `reconnection.py`, `rerender.py`, `scan.py`.

## Results, to `results/` (mirrors the project folders; same names inside)

| Old | New |
|---|---|
| `projects/paper1_methods/`: `Bperp_frames/`, `Bz_frames/`, `Dn1_frames/`, `ETotal_frames/`, `cs_shape_beta1_k10/`, `cs_shape_beta1_k4.706666666666666/`, `cs_shape_m3_beta1_k10/`, `cs_shape_m3_beta1_k4.706666666666666/`, `cs_thickness_pooled_k/` | `results/paper1_methods/` |
| `projects/dirac_images/`: `clock_rings/`, `composites/`, `events_bp1_be1/`, `gallery_bp0.0625/`, `xpoints_t077_near_x067_y244_box96.csv`, `xpoints_t150_near_x080_y230.csv`, `output.log`, `output_particle.log` | `results/dirac_images/` |
| top level: `gaussian_fits/`, `gaussian_fits_J2/`, `gaussian_fits_J2_HR/`, `gaussian_fits_J2_NR/`, `gaussian_fits_J_par/`, five `gaussian_fits*.tar.gz` | `results/archive_cs_detection_development/` |
| `projects/cs_identification/`: `cs_beta_analysis_beta1_t200/`, `cs_beta_pooled_ALL_t200-250/`, `cs_beta_pooled_beta1_t200-250/`, `cs_betae_scan_pooled_bp1_t200-250/`, `cs_betap_scan_pooled_be1_t200-250/`, `cs_debug_visuals/`, `cs_je_vs_aspect_beta1/`, `cs_length_beta0.25/`, `cs_length_beta16/`, `cs_length_ma_vs_geo_beta0.0625/`, `cs_length_ma_vs_geo_beta0.25/`, `cs_length_skel_vs_medax_beta0.25/`, `cs_length_skel_vs_medax_beta16/`, `cs_thickness_pooled/`, `cs_thickness_pooled_1/`, `cs_thickness_results/`, `current_sheets_t40/`, `current_sheets_turbulence_Jz__t200/`, `current_sheets_turbulence_t200/`, `gaussian_fits_J2_NR/`, `gaussian_fits_J2_NR_zhdankin/` | `results/cs_identification/` |
| `projects/cs_identification/`: `JdotE_vs_thickness_M1_beta1_t200.csv`, `cs_count_vs_time_beta0.0625.csv`, `m1_paired_beta1.csv`, `paired_thickness_beta1.csv`, `powerlaw_fits_by_method.csv`, `thickness_vs_intensity_beta1.csv`, `output.log`, `LocalElectronBetaAnalysisFull.log` | `results/cs_identification/` |
| `projects/cs_identification/ElectronBetaScan/`: `gaussian_fits_J2_NR/`, `output.log` | `results/cs_identification/ElectronBetaScan/` |
| `projects/cs_identification/LucaSimulationAnalysis/<1-5>/gaussian_fits_J2_NR/` | `results/cs_identification/LucaSimulationAnalysis/<1-5>/gaussian_fits_J2_NR/` |
| `projects/cs_identification/LucaSimulationAnalysis/1/thicknesses_m1_pooled.csv`, `thicknesses_m2_pooled.csv`, `thicknesses_m3_pooled.csv` | `results/cs_identification/LucaSimulationAnalysis/1/` |

## Caches, to `cache/` (regenerable; safe to delete)

| Old | New |
|---|---|
| `projects/cs_identification/LucaSimulationAnalysis/<1-5>/filtered_cache/` | `cache/cs_identification/LucaSimulationAnalysis/<1-5>/filtered_cache/` |

## New: the `shared/` package

`shared/paths.py` (data, results and cache locations in one place), and `fields.py`, `filtering.py`,
`detection.py`, `thickness.py`, `stats.py`: functions copied verbatim from the Paper 1 pipeline
notebook, checked to give identical results to it. Decisions: detection uses 5 x J2_rms (global) and
0.5 x peak (local); iterative filtering takes its mask from k (the Paper 1 version); Paper 1's thickness
functions are the production methods.

## Not done yet

- Notebooks still define their own copies of the shared functions and use the old paths; they will be
  switched to `shared/` next, after their outputs are stripped (nbstripout).
- Caches (`__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`): left in place, ignored by `.gitignore`.
- The copy of this code on the Northumbria workstation is a separate, older copy and was not changed.

## Code switched to `shared/`

- `projects/cs_identification/PooledStatisticsProtonBeta.ipynb`: its copies of `partial_x_fft`,
  `partial_y_fft`, `iterative_filtering1D`, `iterative_filtering3D`, `detect_current_sheets`, `gaussian`,
  `lognorm_mode` and `lognorm_mode_with_error` replaced by imports from `shared/`; data path from `DATA_ROOT`, cache
  in `cache/cs_identification/filtered_cache/`, outputs in `results/cs_identification/`. Its detection now uses
  5 x / 0.5 x (it had 10 x / 0.1 x).
- `projects/cs_identification/LocalElectronBetaAnalysisFull.ipynb`: the same five detection/filtering functions
  replaced by imports (its copies were identical); data path from `DATA_ROOT`, cache in
  `cache/cs_identification/filtered_cache_Jz/`, outputs in `results/cs_identification/cs_betae_scan_pooled_bp1_t200-250/`
  (where they already were). This notebook is newer than its old .py export (about 300 lines differ).
- `projects/cs_identification/ElectronBetaScan/PooledStatisticsElectronBeta.ipynb`: the same functions as the
  proton-beta notebook replaced by imports; data path from `DATA_ROOT`, cache in
  `cache/cs_identification/ElectronBetaScan/filtered_cache/`, outputs in `results/cs_identification/ElectronBetaScan/`.
  Its detection also had 10 x / 0.1 x and now uses 5 x / 0.5 x. Note: the notebook uses dx = dy = 0.0625 (d_p),
  while its old .py export had dx = dy = 1.0 (pixels).
- The `.py` exports of these three notebooks (already switched to `shared/` earlier) were moved to
  `archive/py_exports/`; the notebooks are the versions kept.
- `CSDetectionMaxTurbulenceFullPipeline.ipynb` moved to `archive/cs_detection_development/`: it was the k-cutoff
  pipeline notebook without the k-cutoff functions, writing to the same output files.
- `LocalBetaAnalysisFull.ipynb` and `CSDetectionAligningDiffMethods.ipynb` switched the same way (identical copies
  -> imports; paths via `DATA_ROOT` / `cache/` / `results/`). `CSDetectionAligningDiffMethods` had detection at
  20 x J2_rms and now uses the shared 5 x / 0.5 x.
- `CSDetectionMaxTurbulenceOneBetaAnimation`, `LocalRhoAnalysis`, `CSDetectionMaxTurbulenceOneBeta`,
  `LocalBetaAnalysis`, `CSDetectionMaxTurbulenceOneBetaLengthAnalysis` and `CSDetectionMaxTurbulence` switched the
  same way (identical copies -> imports; data, caches, figures, movies, CSVs and the Zhdankin diagnostic plots via
  `DATA_ROOT` / `cache/` / `results/`). `CSDetectionMaxTurbulence` had detection at 10 x / 0.1 x and now uses
  5 x / 0.5 x; its per-sheet fits now go to their own folder, `gaussian_fits_J2_NR_max_turbulence/`, instead of
  sharing `gaussian_fits_J2_NR/` with `PooledStatisticsProtonBeta`. `JEAnalysis` switched the same way.
- `Method5CompareBeta.ipynb` moved to `archive/cs_detection_development/`: an older copy of
  `CSDetectionMaxTurbulence` (5 of 7 cells identical, t = 40 only, same output files).
- Per-sheet Gaussian fits: each notebook now has its own folder, so none overwrites another:
  `gaussian_fits_J2_NR/` (`PooledStatisticsProtonBeta`), `gaussian_fits_J2_NR_max_turbulence/`
  (`CSDetectionMaxTurbulence`), `gaussian_fits_J2_NR_one_beta/` (`CSDetectionMaxTurbulenceOneBeta`) and
  `gaussian_fits_J2_NR_aligning_methods/` (`CSDetectionAligningDiffMethods`), all in `results/cs_identification/`.
- `projects/cs_identification/LucaSimulationAnalysis/`: the five per-run notebooks (`1/`-`5/PooledCSAnalysis.ipynb`,
  each with its own 438-line copy of `analyze_simulation`) were replaced by ONE notebook,
  `LucaSimulationAnalysis/PooledCSAnalysis.ipynb`, that runs all five from a `SIMULATIONS` table using run 1's
  `analyze_simulation` (the newest; runs 2-5 had an older one with 20 automatic histogram bins and no mode markers).
  The old folders, with their outputs, are in `archive/luca_simulation_analysis_per_run/`. Data from the new
  `EXTERNAL_DATA_ROOT` setting (`/DATA`, override with PTA_EXTERNAL_DATA). All outputs, including the thickness
  CSVs, sheet coordinates, sheet masks and J^2 files that used to be written to `/DATA/<run>/cs_analysis/`, now go to
  `results/cs_identification/LucaSimulationAnalysis/<run>/`; the cached filtered fields in
  `cache/cs_identification/LucaSimulationAnalysis/<run>/filtered_cache/` are reused. Detection was 10 x / 0.1 x and
  is now the shared 5 x / 0.5 x. Tested end to end (detection, M1/M2/M3, fits, saved files, cache reuse) on
  synthetic fields. Units fixed: thicknesses are converted from pixels to d_i with each run's dx in the histograms
  and the thickness CSVs (the histograms used a hard-coded /16, right only for dx = 0.0625, so for these dx = 0.125
  runs they were a factor 2 too small; the CSVs held pixels under a `thickness_di` header).
- `projects/single_sheet/` (all seven notebooks): identical copies of `partial_x_fft`, `partial_y_fft`,
  `iterative_filtering1D`, `iterative_filtering3D`, `compute_radial_spectrum` and `gaussian` replaced by imports; data
  from `DATA_ROOT`. Each notebook now writes to its own folder, `results/single_sheet/<notebook>/`, because several
  saved figures under the same names (`J2_FWHM.png`, `J2_NR_with_CS_contours.png`, ...) and overwrote each other; the
  files already in `results/single_sheet/` are from before the split. Kept local: the `iterative_filtering3D` of the
  three per-beta notebooks and `MagRecEvent`, which hard-codes Xi = 2.0 (the shared one uses its Xi argument, default
  1.6). Fixed: `IterativeFilteringComparison` saved its spectrum to a file literally named `JB_Spectrum_beta_{beta}`
  (missing f-string).
- `projects/jpar_jperp/`: `Jpar_CSDetectionBeta0.0625`, `Jpar_CSDetectionBeta1`, `Jpar_CSDetectionBeta16`,
  `Jpar_CSDetectionMethod5_Beta0.0625` and `Kdiss` switched the same way (identical copies of `partial_x_fft`,
  `partial_y_fft` and `compute_radial_spectrum` -> imports; data from `DATA_ROOT`), each writing to its own folder
  `results/jpar_jperp/<notebook>/` (`Jpar_CSDetectionBeta0.0625` and the Method5 notebook saved four figures under the
  same names). Kept local: the Method5 notebook's `detect_current_sheets` (global threshold 0.2 x max instead of
  5 x J2_rms: the method that notebook tests) and `Kdiss`'s 2D-FFT derivatives.
- `JparSpectrum`, `JperpSpectrum` and `JperpSpectrum40` (the same notebook for J_par at t = 200, J_perp at t = 200 and
  J_perp at t = 40) merged into `projects/jpar_jperp/JparJperpSpectrum.ipynb` with `COMPONENT` and `TIME` settings;
  the three are in `archive/jpar_jperp_spectra/`. The merged notebook reproduces each original's spectra exactly
  (tested on synthetic fields) and saves the same figure names, in `results/jpar_jperp/JparJperpSpectrum/`. Dropped:
  an unused Gaussian-filter function and unused imports.
- `projects/apj2015/` (`ApJ2015`, `ApJ2015-Copy1`): data from `EXTERNAL_DATA_ROOT` (`/DATA/DEVESH/ApJ2015/`, 107
  path lines) and `DATA_ROOT`; figures to `results/apj2015/`. Its own FFT derivatives were later replaced by the
  shared ones (see the note below); the finite-difference `partial_x_2d` / `partial_y_2d` are kept.
  Fixed: two plot titles with `'$\Delta B_y$'` in non-raw strings (an invalid escape sequence that Python 3.12 warns
  about and a future version will reject) now use raw strings; the titles are unchanged.
- `ApJ2015-Copy1` was not a copy of `ApJ2015`: its imports and helper functions were identical, and its one new cell,
  a 2 x 2 figure of the t = 200 fields (|B_perp|^2, J_par, Delta T_p/T_0, A_p; `visuals.png`), is now a cell in
  `ApJ2015` right after the A_p plot. The notebook itself is in `archive/apj2015/`.
- `ApJ2015` defined `partial_x_fft`/`partial_y_fft` twice: in its second cell (x along axis 0, the same as
  `shared.fields` to ~1e-14) and again in a later cell with x along axis 1. That later version was applied to data read
  with `.T` (x along axis 0), so the cells using it (the two div B checks at t = 200 and J_par at t = 42) swapped the x
  and y derivatives. Both definitions are gone and the notebook imports the shared ones, so those three cells now
  compute div B and J_par with the same convention as every other notebook; the other cells are unchanged.
- The 2D-FFT copies of `partial_x_fft` / `partial_y_fft` kept so far in 17 more notebooks (each matches `shared.fields`
  to ~1e-14) are removed, and those notebooks import the shared ones: `AvgSpectrum`, `Brms`, `DensitySpectrum`,
  `Jrms`, `ReconnectionTime`, `ResistivityCheck`, `Spectrum`, `SpectrumAnimation` (quality_checks); `FilteringCheck`,
  `FilteringCheckEmanuele`, `IterativeFilteringCheck`, `IterativeFilteringIndividualComponents`,
  `IterativeFilteringkcutoff` (experiments); `JBandPass`, `Visuals` (paper1_methods); `Kdiss` (jpar_jperp);
  `IterativeFilteringCheck` (cs_identification). No notebook defines its own `partial_x_fft` / `partial_y_fft` any more.
- More helpers that were copied into many notebooks now live in `shared/` (each shared version gives identical
  results): `partial_x_2d` / `partial_y_2d` (finite differences) in `shared/fields.py`, `gaussian_fft_filter_2d` in
  `shared/filtering.py`, and `index_of_just_smaller` in the new `shared/utils.py`. `shared/filtering.py` now imports
  fifpy inside the iterative-filtering functions, so its other functions work without fifpy. Removed from 28 notebooks
  (quality_checks: `AvgSpectrum`, `Brms`, `DensitySpectrum`, `Jrms`, `ReconnectionTime`, `ResistivityCheck`, `Spectrum`,
  `SpectrumAnimation`; jpar_jperp: the four `Jpar_CSDetection*`, `Kdiss`; single_sheet: the five `SingleCSAnalysis*`,
  `MagRecEvent`; `ApJ2015`; `JBandPass`, `Visuals`; experiments: `FilteringCheck`, `FilteringCheckEmanuele`,
  `IterativeFilteringCheck`, `IterativeFilteringIndividualComponents`, `IterativeFilteringkcutoff`; and
  cs_identification `IterativeFilteringCheck`). Most of those copies were never called and are simply gone; the
  notebooks that call them import the shared versions. Function cells left with nothing in them were removed.
- The `iterative_filtering3D` copies in `SingleCSAnalysisBeta0.0625`, `SingleCSAnalysisBeta1`,
  `SingleCSAnalysisBeta16` and `MagRecEvent` differed from the shared one only by a hard-coded `Xi=2.0` (the `Xi`
  argument was ignored). They are removed; the live call in `SingleCSAnalysisBeta0.0625` and the commented-out calls
  now pass `Xi=2.0` to the shared function, which gives the same result.
- Second round of duplicates moved to `shared/` (each shared version tested identical to every copy it replaced):
  `shared/thickness.py`: `_m3_profile_and_fit`, `_m3_fwhm`, `_walk_to_mask_edge`, `thickness_m1`, `caliper_length`,
  `crop_with_border`; `shared/detection.py`: `get_max_coords`; `shared/stats.py`: `fit_and_plot_pdf`, `avg_beta`;
  `shared/utils.py`: `extract_timestep`; `shared/loading.py`: `load_dataset`; `shared/fields.py`:
  `compute_spectra_at_time` (reads through `load_fields`). Notes on the few behaviours that had to become arguments:
  the M3 settings (`max_len`, `subpix_step`, `margin_frac`, `min_half`) were notebook globals and are now arguments with
  the values every notebook used (60, 0.5, 0.5, 6.0); `caliper_length` takes `connectivity=2` (both notebooks used 2);
  `crop_with_border(region, return_offset=True)` gives the (crop, offset) version; `avg_beta(mask, P_scalar, B2)` takes
  the arrays it used to read from the notebook; the one-value `_walk_m2` callers take `[0]` of the shared
  `(distance, endpoint)`. Notebooks: KCutoff pipeline and its paper1 version, `SingleSheetVisual`,
  `CSDetectionDistributionVsTime`, `CSDetectionMaxTurbulenceOneBeta`, `JEAnalysis`,
  `CSDetectionMaxTurbulenceOneBetaLengthAnalysis`, both `PooledStatistics*`, `LocalBetaAnalysis`, `LocalRhoAnalysis`,
  `EnergyfromComponents`, `KineticEnergyforDifferentResistivity`, `EnergyConservationCheck`, `EnergyPlots`,
  `MagneticEnergyforDifferentResistivity`, `ResistivityCheck`, `AvgSpectrum`, `Spectrum`. Functions defined more than
  once with identical code inside the same notebook (`_init_worker`, `filter_params`, local `load`, `find_sheets`,
  `measure_snapshot`, `crop_box`, `draw_outline`, `draw_centrelines`, `centreline_length`, ...) keep only their first
  definition (checked: no different version was active in between, and every use comes after a definition).
- `ApJ2015` rebuilt (the previous version, with its saved figures of the old ApJ2015 run, is
  `archive/apj2015/ApJ2015_old_run.ipynb`). It now analyses `Brms_0.24/beta_p_1/beta_e_1` (`run_dir(1, 1, 0.24)`);
  every file is read with `load` (one orientation everywhere: before, 81 reads had no `.T` and 29 had one); `dx`, `dy`
  come from `grid_spacing(RUN)` and the times from `output_times(RUN)` (all outputs, 0-300, instead of `range(0, 289)`);
  T_p0 = beta_p/2, T_e = beta_e/2 and B_0 replace the typed 0.25, `/4` and `- 1`; every choice of time and scale is a
  named setting at the top. The five passes over all times are one pass (also saved as `time_series.csv`); the six
  hand-written spectrum loops (whose interpolation weights were swapped) use the shared `compute_radial_spectrum`;
  duplicate cells (two identical div B maps, two B_perp rms plots) are merged; titles that did not match the quantity
  (J_par and div B titled |B_perp|^2, |u| titled |u_perp|^2) are corrected. Every figure (25) and both movies are saved
  to `results/apj2015/ApJ2015/`, figures at 300 dpi with the shared style. Tested end to end on a stand-in run.
- New `shared/plotting.py`: inferno for magnitudes, RdBu_r for signed fields (or fields around a reference value such
  as A_p = 1), a fixed line-colour cycle, `save_figure` (300 dpi) and `show_field` (x horizontal, axes in d_p).
  `shared/loading.py` gained `output_times(run)` and `grid_spacing(run)` (box length 256 d_p / number of cells).
- `projects/experiments/` (`FilteringCheck`, `FilteringCheckEmanuele`, `IterativeFilteringCheck`,
  `IterativeFilteringIndividualComponents`, `IterativeFilteringkcutoff`): identical copies of
  `compute_radial_spectrum`, `iterative_filtering1D`, `iterative_filtering3D` and `k_to_mask_length` replaced by imports;
  data from `DATA_ROOT` / `EXTERNAL_DATA_ROOT`; each notebook writes to `results/experiments/<notebook>/` (several saved
  `JB_Spectrum` and `J2_with(out)_Filter.png` under the same names). `FilteringCheckEmanuele` imported `filters`, which
  now lives in `shared/`: it is imported as `from shared import filters`. Kept local: the 2D-FFT derivatives.
  `IterativeFilteringkcutoff`'s own `iterative_filtering1D(field, Xi, k)` computed a mask length from k but ran with the
  fixed `M=[101]` (the issue fixed in the pipeline); it is replaced by the shared `iterative_filtering1DWithKCutOff`. Its
  calls passed `1` positionally, which set Xi = 1 (k stayed at its default 10, which the fixed mask ignored); they now
  read `iterative_filtering1DWithKCutOff(data_Bx, Xi=1, k=1)`. k = 1 gives exactly the mask 101, so the results are
  unchanged and both settings are now explicit. `Untitled.ipynb` (a scratch check of how M1 thickness values are
  quantised) is renamed `M1Quantisation.ipynb`. Note: `experiments/IterativeFilteringCheck` and
  `cs_identification/IterativeFilteringCheck` are different notebooks that share a name.
- `projects/quality_checks/EnergyPlotsforDifferentBeta` and `EnergyPlotsforDifferentPpc` (identical apart from the file
  list and labels) merged into `EnergyPlots.ipynb` with a `DATASET` setting (`"beta"`: the two extreme-beta runs,
  `"ppc"`: the four particles-per-cell runs); the originals are in `archive/quality_checks_superseded/`. Both used to
  save `EnergyPlotPpc.png`; the figure is now `EnergyPlot_<DATASET>.png` in `results/quality_checks/EnergyPlots/`.
  Tested on synthetic files: the plotted curves are identical to each original's.
- `projects/quality_checks/Jrms.ipynb`: the five cells that were exact copies of `ResistivityCheck` cells (the
  resistivity-run B_rms helper and plot, the `CompareBp_*` figures, the spectra at t = 200/250, J_rms of the
  resistivity runs; they stay in `ResistivityCheck`) and an empty cell were removed; switched to `shared/` paths, figures
  to `results/quality_checks/Jrms/`. The notebook was rewritten for this, so its saved outputs were cleared (its figure
  is saved as a file).
- The other 14 `projects/quality_checks/` notebooks (`AvgSpectrum`, `Brms`, `DensitySpectrum`,
  `EnergyConservationCheck`, `EnergyfromComponents`, `KineticEnergyforDifferentResistivity`,
  `MagneticEnergyforDifferentResistivity`, `ReconnectionTime`, `ResistivityCheck`, `SNR`, `SimulationHeatMap`, `Spectrum`,
  `SpectrumAnimation`, `SpectrumCompareBeta`): identical copies of `compute_radial_spectrum`, `iterative_filtering1D`
  and `iterative_filtering3D` replaced by imports; data from `DATA_ROOT` (beta scan) and `EXTERNAL_DATA_ROOT`
  (`/DATA/DEVESH/ResistivityCheck`, `/DATA/DEVESH/ExtremeBetaSimulations`); each notebook writes to
  `results/quality_checks/<notebook>/` (`Brms.png`, the `CompareBp_*` figures, `JBD_Spectrum` and others were saved under
  the same names by several notebooks). Kept local: the 2D-FFT derivatives and the `compute_radial_spectrum` variants
  of `SNR` and `SpectrumCompareBeta`. `EnergyConservationCheck` saved its figure inside the simulation's data folder
  (`{Base_dir}/plots`); it now goes to `results/quality_checks/EnergyConservationCheck/<beta_p>/<beta_e>/`.
- `ApJ2015` saved two animations (`DivB_movie.mp4`, `Bperp_squared_movie.mp4`) next to the notebook; they now go to
  `results/apj2015/`.
- `projects/dirac_images/` (all 12 scripts): `RUN` (and `find_xpoint.py`'s `--run` default) now use `NAS_ROOT`; every
  output folder now lives in `results/dirac_images/` under its old name, so the chains still connect (`scan` ->
  `rerender` -> `event_videos_v2` through `events_bp1_be1`; `gallery` -> `neon` / `palette` through `gallery_bp0.0625`;
  `clock_and_rings` -> `label_new` through `clock_rings`). `find_xpoint.py` wrote its `xpoints_*.csv` / `.png` to the
  working directory; `particle.py` and `reconnection.py` use `OUT` as a file-name prefix: all of these now land in
  `results/dirac_images/`. The import block finds the repository from the script's own location (`__file__`), so the
  scripts run from any directory; in `find_xpoint.py` it comes before the argument parser. Kept local: the copies of
  `k_to_mask_length` / `iterative_filtering1DWithKCutOff` in `event_composite.py` and `event_videos_v2.py` (already
  mask-from-k; they import `fifpy` only when needed, so the Gaussian mode runs without it, and have an `IF_VERBOSE`
  switch). Checked by importing each script from a copy of the repository, run from another directory.
- `projects/paper1_methods/CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb`: its copies of
  `compute_radial_spectrum`, `partial_x_fft`, `partial_y_fft`, `k_to_mask_length`, the three iterative-filtering
  functions, `detect_current_sheets`, `gaussian`, `_walk_m2`, `bootstrap_lognorm_median` and `bell` replaced by one
  import block in the first cell (these are the functions `shared/` was built from); data from `DATA_ROOT`,
  cache in `cache/paper1_methods/filtered_cache_k/`, outputs in `results/paper1_methods/`.
- `lognorm_mode_with_error` and `lognorm_mode` moved to `shared/stats.py` and imported by both pooled-statistics
  scripts (the copies differed only in comments and spacing; the electron-beta script defined the first one twice).
- `projects/paper1_methods/JBandPass.ipynb`: copies of `k_to_mask_length`, `compute_radial_spectrum`,
  `iterative_filtering1D` and `iterative_filtering3D` replaced by imports (identical); data from `DATA_ROOT`,
  figures to `results/paper1_methods/`. Its own `partial_x_fft` / `partial_y_fft` (2D-FFT version) kept by choice
  (identical to the shared 1D-FFT version to ~4e-16).
- `projects/paper1_methods/Visuals.ipynb`: copy of `iterative_filtering1D` replaced by an import; data from
  `DATA_ROOT` (beta scan) and `NAS_ROOT` (`DATA/Brms_0.24`); figures and the LIC image to `results/paper1_methods/`.
  Its own `partial_x_fft` / `partial_y_fft` (2D-FFT version) kept by choice.
- `projects/paper1_methods/BperpAnimation.ipynb`: paths only (no shared functions). The import block is at the top
  of the first cell, so run that cell first; frames, the animation and figures go to `results/paper1_methods/`.
- `shared/paths.py`: new `NAS_ROOT` setting (the workstation's NAS, override with PTA_NAS) for datasets outside
  the beta scan; `DATA_ROOT` is now `NAS_ROOT/simulations_beta_scan` (same path as before).
- `projects/cs_identification/CSDetectionMaxTurbulenceFullPipelineWithKCutoff.ipynb`: copies of `partial_x_fft`,
  `partial_y_fft`, `k_to_mask_length`, `iterative_filtering1D`, `iterative_filtering3D`, `detect_current_sheets`,
  `gaussian` and `bootstrap_lognorm_median` replaced by imports (identical). Its `iterative_filtering1DWithKCutOff`
  with the fixed mask `M=[101]` replaced by the shared version (mask from k), as decided. The experiment cells keep
  their own variants; `_walk_m2` (cell 3) and `bell` (cell 8) are kept too, although identical to the shared ones,
  because other cells define different versions of the same names and re-running cells would otherwise mix them.
  Data from `DATA_ROOT`, cache in `cache/cs_identification/filtered_cache_Jz/`, all figures, `.npz` and `.csv`
  outputs in `results/cs_identification/`.
- `projects/cs_identification/`: `SingleSheetVisual`, `ThicknessVsIntensity`, `CSDetectionDistributionVsTime`,
  `J2DistributionInsideCS`, `IterativeFilteringCheck` and `ThicknessvsBetaFits` notebooks switched the same way
  (identical copies -> imports; data, caches, figures, videos and CSVs via `DATA_ROOT` / `cache/` / `results/`;
  `ThicknessvsBetaFits` now reads its `pooled_summary.csv` inputs from `results/cs_identification/`).
  `J2DistributionInsideCS` had detection at 10 x / 0.1 x and now uses the shared 5 x / 0.5 x.
  `IterativeFilteringCheck` keeps its own `iterative_filtering3D` (2 IMFs, fixed mask 1000: a filtering experiment)
  and its 2D-FFT derivatives. Experiment variants of `_walk_m2` are kept where they differ from the shared one.
  In `ThicknessvsBetaFits` each cell is a stand-alone script; the import block is at the top of the first cell,
  so run that cell first.

## Duplicate quality-check notebooks: the better version kept

Both versions existed at the top level (June) and in `QualityCheck/` (February). The better one stays in
`projects/quality_checks/` under its normal name; the other went to `archive/quality_checks_superseded/`.

| Notebook | Kept | Archived as |
|---|---|---|
| `Brms.ipynb` | February (marks the B_rms peak, runs in order) | `Brms_Jun2026.ipynb` (same code pointed at ExtremeBetaSimulations/16, peak marking removed) |
| `Jrms.ipynb` | June (14 cells, 5 figures, Gaussian denoising added) | `Jrms_Feb2026.ipynb` |
| `EnergyConservationCheck.ipynb` | February (produced its figure) | `EnergyConservationCheck_Jun2026.ipynb` (identical code, other run; failed with FileNotFoundError) |
| `ReconnectionTime.ipynb` | June (no errors, 4 figures) | `ReconnectionTime_Feb2026.ipynb` (ended in ValueError and RuntimeError) |

## Results to regenerate once the code uses `shared/`

- Outputs of `PooledStatisticsProtonBeta.ipynb` (`results/cs_identification/gaussian_fits_J2_NR/` and
  `J2_NR_M1_M2_M3_pooled.png`): made with 10 x J2_rms / 0.1 x peak detection thresholds.
- Outputs of `ElectronBetaScan/PooledStatisticsElectronBeta.ipynb` (`results/cs_identification/ElectronBetaScan/`):
  also made with 10 x / 0.1 x thresholds.
- `results/cs_identification/cs_thickness_pooled_1/`: made with a fixed filter mask M=[101]
  (about k = 1 rad/d_p) instead of the mask from k.
- Figures of `J2DistributionInsideCS` (`J2_full.png`, `J2_inside.png`, `J2_insideVsoutside_beta*.png`):
  detection was at 10 x / 0.1 x.
- Outputs of `CSDetectionAligningDiffMethods` (`J2_comparison_beta_*.png`, `J2_NR_M1_M2_M3_t40.png`,
  `current_sheets_t40/`): detection was at 20 x J2_rms.
- Outputs of `CSDetectionMaxTurbulence` (`J2_comparison_all_betas.png`, `J2_NR_M1_M2_M3_combined.png`,
  `J2_NR_M1_M2_M3_zhdankin_combined.png`, `J2_NR_M4_diag_beta*.png`, `gaussian_fits_J2_NR_zhdankin/`): detection
  was at 10 x / 0.1 x. Its per-sheet fits will now be written to `gaussian_fits_J2_NR_max_turbulence/`; the
  existing `gaussian_fits_J2_NR/` holds a mixture of its fits and those of `PooledStatisticsProtonBeta`.
- Outputs of Luca's five runs (`results/cs_identification/LucaSimulationAnalysis/*/`, and the older copies in
  `/DATA/<run>/cs_analysis/`): detection was 10 x / 0.1 x; runs 2-5 also used the older histogram code. Their
  thickness histograms were a factor 2 too small and their thickness CSVs were in pixels, not d_i (see above).

## Standards applied project by project (from 27 October 2026)

Every converted notebook:
- reads fields only through `shared.loading.load` (x along axis 0) and history files (`EnTot`, `EnMa`, `EnKi1`, `Time`)
  through `load_history`; no direct `h5py` reads;
- gets its runs from `run_dir` (beta scan, B_rms as `BRMS`), `external_run` (the /DATA disk) or `shared/runs.py`, its
  grid spacing from `grid_spacing(run)` and its times from `output_times(run)`; worker counts follow the machine;
- keeps its choices (snapshot times, plot ranges, thresholds) as named settings near the top;
- saves every figure with `save_figure` (300 dpi) in `results/<project>/<notebook>/`, colours from `shared/plotting.py`
  (inferno for magnitudes, RdBu_r for signed fields, gray for masks, matplotlib's ten line colours).

Added to `shared/`: `loading.load_history`, `paths.external_run`, `fields.current_density`, `fields.summed_spectrum`,
`runs.py` (`RESISTIVITY_SCAN`, `PARTICLES_PER_CELL_SCAN`, `EXTREME_BETA_SCAN`), `plotting.MASK_CMAP` and
`show_field(..., mask=True)`.

### `quality_checks` (all 16 notebooks done)

Corrections beyond the standards above:
- `SNR`: wrote its figures into `plots/` inside each data run on the NAS; now `results/quality_checks/SNR/<run>/`.
  Its last cell overwrote the first cell's figures (now separate names) and labelled a t = 0 spectrum t = 1. Its own
  `compute_radial_spectrum` replaced by the shared one (identical up to the Nyquist wavenumber; same in
  `SpectrumCompareBeta`).
- `ResistivityCheck`: J was differentiated with dx = 0.125 but its spectrum used dx = 0.0625, and the fields were
  read without `.T` (so the x derivative acted along y). Both now use the run's own grid spacing.
- `ReconnectionTime`: a comment said the peak prominence was 10 % while the code used 1 %; the last spectrum was
  titled perpendicular but sums all three J components.
- `EnergyfromComponents`, `KineticEnergyforDifferentResistivity`: 16777216 (4096 squared) replaced by the data's own
  number of cells; the 0.75 added to the energy is now `1.5 * T_E` with `T_E = BETA_E / 2`; six figures that were
  only shown are now saved; two cells saved to the same `TotalEnergyComparison.png` (now separate names). A
  commented-out block copied from the old ApJ2015 notebook (temperature anisotropy, unrelated) was removed from
  `EnergyfromComponents`.
- `AvgSpectrum`: the side-by-side comparison looked for its two PNGs in the notebook folder, where they never were;
  it now reads them from its results folder and says which are missing.
- `SpectrumAnimation`: its own `compute_spectra_at_time` (B and u spectra) had the same name as the shared function
  (B, J and n spectra); renamed `B_and_u_spectra_at`.
- `AvgSpectrum`, `Spectrum`: slope guides are drawn along the spectra (`SLOPE_GUIDES`: -5/3 and -3 on B, -1 on J,
  -2.8 on n) instead of typed index ranges and amplitudes.
- `SimulationHeatMap`: viridis -> inferno, 150 -> 300 dpi. To check: its docstring says the table rows are beta_p,
  but the comments in the tables label the rows beta_e; the plot puts rows on the x-axis (beta_p).
- `DensitySpectrum` saved a density-only figure as `JBD_Spectrum`.

All `quality_checks` figures now have new names (e.g. `J_rms_vs_time.png`, `density_spectra_t1_t200.png`); the old
files in `results/quality_checks/` can be deleted once the notebooks have been rerun.
