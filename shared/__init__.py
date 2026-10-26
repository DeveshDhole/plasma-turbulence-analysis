"""
Shared code for all projects. Import it from any notebook or script:

    import sys; sys.path.insert(0, "<path to plasma-turbulence-analysis>")
    from shared.fields import partial_x_fft, partial_y_fft
    from shared.paths import results_dir, cache_dir, run_dir

Modules
  paths      where simulation data, results and caches live (one place to change)
  fields     spectral derivatives and radial spectra
  filtering  iterative filtering (fifpy), incl. the k-cutoff version used in Paper 1
  detection  current-sheet detection (5 x J2_rms global, 0.5 x peak local threshold)
  thickness  thickness-method helpers (production versions from Paper 1)
  stats      statistics helpers

Functions in fields, filtering, detection, thickness and stats are copied verbatim from
the Paper 1 pipeline notebook and were checked to give identical results to it.

Also here (used by older notebooks): filters.py (Hampel filter) and
current_sheets_id_char_collisionless_turbulence.py (Zhdankin et al. 2013 algorithm as
adapted by Chatraee Azizabadi et al. 2021).
"""
