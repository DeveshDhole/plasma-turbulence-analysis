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


def test_cuts_across_a_sheet_match_the_single_sheet_notebooks():
    from scipy.ndimage import map_coordinates
    from pta.sheets import descent_direction, fwhm_nearest_samples, line_profile, width_along
    nx = ny = 120
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    J2 = np.exp(-((x - 60 + 0.3 * (y - 60))**2) / 20.0) * np.exp(-((y - 60)**2) / 900.0) + 0.002 * np.random.default_rng(9).random((nx, ny))
    cell = (61, 58)
    # direction: the notebooks took np.gradient of the whole field
    gradient_x, gradient_y = np.gradient(J2)
    reference = -np.array([gradient_x[cell], gradient_y[cell]]); reference /= np.linalg.norm(reference)
    direction = descent_direction(J2, cell)
    assert np.array_equal(direction, reference)
    for edge_cell in ((0, 5), (nx - 1, 7), (9, ny - 1)):
        g = -np.array([gradient_x[edge_cell], gradient_y[edge_cell]]); g /= np.linalg.norm(g)
        assert np.array_equal(descent_direction(J2, edge_cell), g)
    # profile along the cut
    s, profile = line_profile(J2, cell, direction, 30)
    assert np.array_equal(profile, map_coordinates(J2, [cell[0] + s * direction[0], cell[1] + s * direction[1]], order=1))
    # a region to walk across: the sheet's own cells
    region = J2 >= 0.1 * J2[cell]
    # width across the region: the notebooks walked forward and backward
    def walk(sign):
        xf, yf = float(cell[0]), float(cell[1])
        for _ in range(1000):
            xn, yn = xf + sign * direction[0], yf + sign * direction[1]
            xi, yi = int(round(xn)), int(round(yn))
            if xi < 0 or yi < 0 or xi >= nx or yi >= ny or not region[xi, yi]: break
            xf, yf = xn, yn
        return np.hypot(xf - cell[0], yf - cell[1])
    width, forward, backward = width_along(region, cell, direction)
    assert (forward, backward) == (walk(1), walk(-1)) and width == forward + backward
    # Gaussian FWHM from whole-cell samples: about 2 sqrt(2 ln 2) times the sheet's sigma across it
    k, values, parameters, fwhm = fwhm_nearest_samples(J2, cell, direction, half_length=40)
    assert len(k) == 81 and 4 < fwhm < 14


def test_harris_fit_recovers_a_harris_sheet():
    from pta.sheets import fit_harris, harris_profile
    s = np.linspace(-40, 40, 400)
    noise = 0.002 * np.random.default_rng(10).normal(size=s.size)
    signed = fit_harris(s, harris_profile(s, 0.3, 1.5, 4.0, 0.1) + noise)
    assert abs(signed["delta"] - 4.0) < 0.1 and abs(signed["s0"] - 1.5) < 0.1
    magnitude = fit_harris(s, np.abs(harris_profile(s, 0.3, -2.0, 6.0, 0.0)) + 0.05 + noise, magnitude=True)
    assert abs(magnitude["delta"] - 6.0) < 0.2 and abs(magnitude["s0"] + 2.0) < 0.2
    centred = fit_harris(s, harris_profile(s, 0.3, 0.0, 3.0, 0.0) + noise, centred=True)
    assert centred["s0"] == 0.0 and abs(centred["delta"] - 3.0) < 0.1


def synthetic_sheet_run(folder, n=192, times=(0, 5), nsheets=3):
    """A run whose By reverses across a few current sheets, so detection finds them."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    x, y = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")

    def profile(coordinate, centres, widths):
        out = np.full_like(coordinate, -1.0)
        for number, (centre, width) in enumerate(zip(centres, widths)):
            sign = 1 if number % 2 == 0 else -1
            for shift in (-n, 0, n):
                out += sign * (np.tanh((coordinate - centre - shift) / width) + 1)
        return out

    for time in times:
        By = 0.3 * profile(x + 5 * np.sin(2 * np.pi * y / n), [40, 110, 160][:nsheets], [4, 6, 5][:nsheets])
        Bx = 0.3 * profile(y, [70], [5])
        for name, data in (("Bx", Bx), ("By", By), ("Bz", 1 + 0 * x)):
            with h5py.File(folder / f"{name}_ApJ_t{time}.h5", "w") as f:
                f.create_dataset("DS1", data=(data + 0.004 * rng.normal(size=data.shape)).T)
    return pta.Run(folder, box_length=n / 16)                      # dx = 1/16 d_p, as in the production runs


def test_pooled_thickness_matches_the_production_pipeline(tmp_path, monkeypatch):
    from scipy import ndimage as ndi
    from skimage import measure, morphology
    from pta.fields import partial_x_fft, partial_y_fft
    from pta.sheets import detect_current_sheets, pooled_thickness, snapshot_thickness
    from pta.sheets.thickness import _m3_profile_and_fit, _walk_m2
    import matplotlib
    matplotlib.use("Agg")

    monkeypatch.setenv("PTA_RESULTS", str(tmp_path / "results"))
    monkeypatch.setenv("PTA_CACHE", str(tmp_path / "cache"))
    config.reload()
    run = synthetic_sheet_run(tmp_path / "run")

    # the production pipeline, inline, for one snapshot
    Bx, By = run.load_many(["Bx", "By"], 0)
    J = np.abs(partial_x_fft(By, run.dx) - partial_y_fft(Bx, run.dy))
    mask = np.zeros_like(J)
    for sheet in detect_current_sheets(J):
        for i, j in sheet:
            mask[i, j] = 1
    disk = morphology.disk(1)
    mask = morphology.closing(morphology.opening(mask, disk), disk)
    try:
        cleaned = morphology.remove_small_objects(mask.astype(bool), max_size=499)
    except TypeError:
        cleaned = morphology.remove_small_objects(mask.astype(bool), min_size=500)
    labelled = measure.label(cleaned, connectivity=2)
    gx, gy = np.gradient(ndi.gaussian_filter(J, sigma=3))
    expected = {"m1": [], "m2": [], "m3": []}
    for index in range(1, int(labelled.max()) + 1):
        region = labelled == index
        rows, columns = np.where(region)
        peak = int(np.argmax(J[region]))
        i, j = int(rows[peak]), int(columns[peak])
        expected["m1"].append(2 * float(ndi.distance_transform_edt(region).max()))
        gradient = np.array([gx[i, j], gy[i, j]])
        direction = (0.0, 0.0) if np.allclose(gradient, 0) else tuple(map(float, -gradient / np.linalg.norm(gradient)))
        expected["m2"].append(_walk_m2(cleaned, i, j, *direction)[0]
                              + _walk_m2(cleaned, i, j, -direction[0], -direction[1])[0])
        expected["m3"].append(_m3_profile_and_fit(J, cleaned, i, j, *direction)[3])
    assert int(labelled.max()) >= 2

    got = snapshot_thickness(run, 0, k=None)
    assert got["n_sheets"] == int(labelled.max())
    for method, values in expected.items():
        values = np.asarray(values, dtype=float)
        if method == "m2":
            values = values[values > 0]
        if method == "m3":
            values = values[np.isfinite(values) & (np.asarray(values) > 0)]
        assert np.allclose(np.sort(values * run.dx), np.sort(got[method])), method

    # pooling over several snapshots: one figure, one panel per method, and a lognormal fit per method
    run = synthetic_sheet_run(tmp_path / "run", times=range(0, 25, 5))       # enough sheets to fit
    pooled = pooled_thickness(run=run, times=list(range(0, 25, 5)), k=None, workers=2, bootstrap=50)
    assert pooled.thickness["m1"].size == sum(len(s) for s in pooled.per_time["m1"].values())
    assert len(pooled.figure.axes) >= 3
    for row in pooled.statistics:
        assert row["n"] >= 5                                      # fewer than 5 sheets: fit_and_plot_pdf returns NaN
        assert row["median_lognorm"] > 0 and row["ci_lo"] <= row["median_lognorm"] <= row["ci_hi"]
        assert 0.5 < row["median_lognorm"] / np.median(pooled.thickness[row["method"]]) < 2
    figure_path, array_path = pooled.save(config.results_dir("pooling_test"))
    assert figure_path.exists() and array_path.exists()
    stored = np.load(array_path)
    assert np.array_equal(stored["m3_d_p"], pooled.thickness["m3"])


def test_animation_writes_a_movie_of_every_frame(tmp_path, monkeypatch):
    import shutil
    import matplotlib
    matplotlib.use("Agg")
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from pta import animation

    monkeypatch.setenv("PTA_RESULTS", str(tmp_path / "results"))
    config.reload()
    run = synthetic_sheet_run(tmp_path / "run", n=64, times=(0, 5, 10))
    field_movie = animation.animate_field(run, lambda time: run.load("Bx", time), tmp_path / "field.mp4",
                                          centre=0, workers=2)
    line_movie = animation.animate_lines(np.arange(5), lambda time: np.arange(5) * (1 + time),
                                         tmp_path / "lines.mp4", frames=[0, 5, 10], labels=["a"])
    assert field_movie.exists() and field_movie.stat().st_size > 0
    assert line_movie.exists() and line_movie.stat().st_size > 0
