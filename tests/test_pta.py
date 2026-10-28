"""Tests of the pta package on small synthetic runs. Run with: python -m pytest tests"""
import h5py
import numpy as np
import pytest

import pta
from pta import config
from pta.fields import (current_density, parallel_component, parallel_perpendicular_squared, partial_x_fft,
                        partial_y_fft, resistive_dissipation_wavenumber)
from pta.filtering import gaussian_fft_filter_2d
from pta.parallel import parallel_map
from pta.sheets import (detect_current_sheets, get_max_coords, quantile_mask, region_thicknesses, sheets_to_mask,
                        thickness_m1)
from pta.spectra import compute_radial_spectrum, parallel_perpendicular_spectra, summed_spectrum
from pta.statistics import fit_and_plot_pdf


def write_run(folder, nx=32, ny=24, times=(0, 5, 10), tag="ApJ"):
    """A run folder in CAMELIA's format: <field>_<tag>_t<time>.h5, arrays stored as (y, x) in "DS1"."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    fields = {}
    for time in times:
        for field in ("Bx", "By", "Bz"):
            data = rng.normal(size=(nx, ny)) + time            # (x, y) as the user sees it
            fields[field, time] = data
            with h5py.File(folder / f"{field}_{tag}_t{time}.h5", "w") as f:
                f.create_dataset("DS1", data=data.T)
    with h5py.File(folder / f"EnTot_{tag}_.h5", "w") as f:
        f.create_dataset("DS1", data=np.arange(4.0))
    return fields


def test_run_reads_camelia_output(tmp_path):
    fields = write_run(tmp_path / "run")
    run = pta.Run(tmp_path / "run")
    assert list(run.times) == [0, 5, 10]
    assert run.shape == (32, 24) and run.dx == 256 / 32 and run.dy == 256 / 24
    assert np.array_equal(run.load("By", 5), fields["By", 5])                   # x along axis 0
    assert all(np.array_equal(a, fields[name, 10]) for a, name in zip(run.load_many(["Bx", "Bz"], 10), ["Bx", "Bz"]))
    assert np.array_equal(run.history("EnTot"), np.arange(4.0))
    # the folder-path functions give the same
    assert np.array_equal(pta.run.load("By", 5, tmp_path / "run"), fields["By", 5])
    assert pta.run.grid_spacing(tmp_path / "run") == (run.dx, run.dy)
    assert list(pta.run.output_times(tmp_path / "run")) == [0, 5, 10]
    with pytest.raises(FileNotFoundError):
        run.load("Bx", 7)


def test_run_with_another_file_layout(tmp_path):
    folder = tmp_path / "other"
    folder.mkdir()
    for time in (3, 12):
        with h5py.File(folder / f"Bx_{time:04d}.h5", "w") as f:
            f.create_dataset("data", data=np.full((8, 6), float(time)))           # stored as (x, y)
    run = pta.Run(folder, pattern="{field}_{time:04d}.h5", dataset=None, transpose=False, box_length=(16, 12))
    assert list(run.times) == [3, 12] and run.shape == (8, 6) and (run.dx, run.dy) == (2.0, 2.0)
    assert run.load("Bx", 12)[0, 0] == 12


def test_map_runs_every_time_in_order(tmp_path):
    write_run(tmp_path / "run")
    run = pta.Run(tmp_path / "run")
    means = run.map(lambda time: run.load("Bx", time).mean(), workers=3)
    assert np.allclose(means, [run.load("Bx", time).mean() for time in run.times])
    assert parallel_map(lambda x: x * x, range(9), workers=4) == [x * x for x in range(9)]


def reference_radial_spectrum(field, dx, dy):
    """The original point-by-point loop, to check the vectorised version against."""
    nx, ny = field.shape
    kx, ky = np.fft.fftfreq(nx, d=dx) * 2 * np.pi, np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    K = np.sqrt(kx[:, None]**2 + ky[None, :]**2).ravel()
    P = (np.abs(np.fft.ifft2(field))**2).ravel()
    dk = kx[1] - kx[0]
    k_bins = np.arange(0, K.max() + dk, dk)
    spectrum = np.zeros(len(k_bins))
    i1 = np.floor(K / dk).astype(int)
    w = K / dk - i1
    for i in range(len(K)):
        if i1[i] >= 0 and i1[i] + 1 < len(k_bins):
            spectrum[i1[i]] += P[i] * (1 - w[i])
            spectrum[i1[i] + 1] += P[i] * w[i]
    return k_bins, spectrum / dk


def test_radial_spectrum_matches_the_original_loop():
    field = np.random.default_rng(1).normal(size=(48, 40))
    spectrum = compute_radial_spectrum(field, 0.5, 0.5)
    k, reference = reference_radial_spectrum(field, 0.5, 0.5)
    assert np.array_equal(spectrum.kk, k)
    assert np.allclose(spectrum.sp, reference, rtol=1e-12, atol=0)
    k_sum, total = summed_spectrum((field, 2 * field), 0.5, 0.5)
    assert np.allclose(total, 5 * spectrum.sp)


def test_derivatives_and_current_density():
    nx, ny, dx, dy = 64, 32, 0.25, 0.5
    x, y = np.arange(nx)[:, None] * dx, np.arange(ny)[None, :] * dy
    kx, ky = 2 * np.pi / (nx * dx), 2 * np.pi / (ny * dy)
    f = np.sin(kx * x) * np.cos(ky * y)
    assert np.allclose(partial_x_fft(f, dx), kx * np.cos(kx * x) * np.cos(ky * y))
    assert np.allclose(partial_y_fft(f, dy), -ky * np.sin(kx * x) * np.sin(ky * y))
    Jx, Jy, Jz = current_density(f, 2 * f, 3 * f, dx, dy)
    assert np.allclose(Jz, 2 * partial_x_fft(f, dx) - partial_y_fft(f, dy))
    assert np.allclose(Jx, 3 * partial_y_fft(f, dy)) and np.allclose(Jy, -3 * partial_x_fft(f, dx))


def test_config_paths_runs_and_scans(tmp_path, monkeypatch):
    (tmp_path / "data" / "Brms_0.24" / "beta_p_1" / "beta_e_1").mkdir(parents=True)
    (tmp_path / "pta.toml").write_text('''
[paths]
data = "data"
results = "out/results"
[runs]
beta_scan = "{data}/Brms_{brms}/beta_p_{beta_p}/beta_e_{beta_e}"
[run_defaults.beta_scan]
brms = 0.24
beta_e = 1
[run_options.beta_scan]
tag = "EL4modes"
[scans.two]
layout = "beta_scan"
runs = [{ beta_p = 1, label = "one" }, { beta_p = 4 }]
''')
    monkeypatch.chdir(tmp_path / "data")                       # pta.toml is found in a parent folder
    monkeypatch.delenv("PTA_CONFIG", raising=False)
    monkeypatch.delenv("PTA_DATA", raising=False)
    config.reload()
    assert config.project_root() == tmp_path
    assert config.path("data") == tmp_path / "data"
    assert config.results_dir("p", "n") == tmp_path / "out" / "results" / "p" / "n" and (tmp_path / "out/results/p/n").is_dir()
    run = config.run("beta_scan", beta_p=1)
    assert isinstance(run, pta.Run) and run.path == tmp_path / "data/Brms_0.24/beta_p_1/beta_e_1" and run.tag == "EL4modes"
    assert [label for _, label in config.scan("two")] == ["one", "beta_e_1"]
    monkeypatch.setenv("PTA_DATA", str(tmp_path / "elsewhere"))
    assert config.run_path("beta_scan", beta_p=4, beta_e=16) == tmp_path / "elsewhere/Brms_0.24/beta_p_4/beta_e_16"


def test_gaussian_filter_removes_small_scales():
    field = np.random.default_rng(2).normal(size=(64, 64))
    filtered = gaussian_fft_filter_2d(field, k0=4)
    high = compute_radial_spectrum(field, 1, 1).kk > 1.5
    assert compute_radial_spectrum(filtered, 1, 1).sp[high].sum() < 1e-3 * compute_radial_spectrum(field, 1, 1).sp[high].sum()


def test_iterative_filtering_with_k_cutoff():
    fifpy = pytest.importorskip("fifpy")
    from pta.filtering import iterative_filtering1DWithKCutOff
    nx, dx = 128, 0.25
    x = np.arange(nx)[:, None] * dx
    smooth = np.sin(2 * np.pi * x / (nx * dx)) * np.ones((1, 64))
    noisy = smooth + 0.3 * np.random.default_rng(3).normal(size=(nx, 64))
    filtered = iterative_filtering1DWithKCutOff(noisy, k=2.0, dx=dx)
    assert np.std(filtered - smooth) < np.std(noisy - smooth)


def test_detect_current_sheets_finds_a_sheet():
    J2 = np.random.default_rng(4).random((128, 128)) * 0.1
    J2[60:63, 20:100] = 50.0                                   # one thin intense sheet
    sheets = detect_current_sheets(J2)
    assert len(sheets) >= 1


def test_parallel_and_perpendicular_parts():
    rng = np.random.default_rng(5)
    J, B = tuple(rng.normal(size=(3, 16, 12))), tuple(rng.normal(size=(3, 16, 12)))
    parallel, perpendicular = parallel_perpendicular_squared(J, B)
    total = J[0]**2 + J[1]**2 + J[2]**2
    assert np.allclose(parallel, parallel_component(J, B)**2) and np.allclose(parallel + perpendicular, total)
    assert np.all(perpendicular > -1e-12)
    along = parallel_component(B, B)                                      # B along itself: |B|
    assert np.allclose(along, np.sqrt(B[0]**2 + B[1]**2 + B[2]**2))
    k, total_spectrum, parallel_spectrum, perpendicular_spectrum = parallel_perpendicular_spectra(J, B, 0.5, 0.5)
    assert np.allclose(total_spectrum, summed_spectrum(J, 0.5, 0.5)[1])
    assert np.allclose(parallel_spectrum + perpendicular_spectrum, total_spectrum)
    assert np.isclose(resistive_dissipation_wavenumber(16.0, 1e-4), 2.0 / 1e-2)


def test_quantile_mask_and_sheet_measurements():
    nx = ny = 160
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    field = np.exp(-((x - 50)**2) / 18.0) * (np.abs(y - 80) < 60) + 0.01 * np.random.default_rng(6).random((nx, ny))
    field += np.exp(-((x - 120)**2 + (y - 40)**2) / 30.0)                 # a second, round structure
    mask = quantile_mask(field, 0.9, smoothing_radius=2, min_area=50)
    assert set(np.unique(mask)) <= {0, 1} and mask[50, 80] == 1 and mask[5, 5] == 0
    thicknesses = region_thicknesses(mask, step=0.5)
    assert len(thicknesses) == 2 and all(t > 0 for t in thicknesses)
    labeled = __import__("scipy.ndimage", fromlist=["label"]).label(mask, structure=np.ones((3, 3)))[0]
    assert np.allclose(sorted(thicknesses), sorted(thickness_m1(labeled == i, step=0.5) for i in (1, 2)))
    coords, _ = get_max_coords(mask, field)
    assert (50, 80) in [tuple(c) for c in coords] or any(abs(c[0] - 50) <= 1 for c in coords)
    sheets = detect_current_sheets(field, global_threshold=0.5 * field.max(), local_fraction=0.5)
    assert len(sheets) == 2
    sheet_mask = sheets_to_mask(sheets, field.shape)
    assert sheet_mask.sum() == sum(len(sheet) for sheet in sheets)
    assert detect_current_sheets(field) == detect_current_sheets(field, global_threshold=5 * field.mean())


def test_pdf_panel_options():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = np.random.default_rng(7).lognormal(mean=0.0, sigma=0.4, size=400)
    fig, ax = plt.subplots()
    default = fit_and_plot_pdf(ax, values, "default")
    assert set(default) == {"n", "median_lognorm", "sigma_lognorm", "mean", "median_emp"}
    assert abs(default["median_lognorm"] - 1.0) < 0.1
    with_mode = fit_and_plot_pdf(ax, values, "mode", bins=10, fit_loc=True, mark="mode")
    assert {"mode_lognorm", "loc_lognorm"} <= set(with_mode) and with_mode["mode_lognorm"] < with_mode["median_lognorm"]
    plt.close(fig)
