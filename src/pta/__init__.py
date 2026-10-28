"""pta: analysis of 2D kinetic plasma-turbulence simulations (CAMELIA output by default).

    import pta
    run = pta.Run("/path/to/run")                          # a folder of <field>_ApJ_t<time>.h5 files
    Bx, By = run.load_many(["Bx", "By"], 200)              # 2D arrays, x along axis 0
    k, spectrum = pta.spectra.summed_spectrum((Bx, By), run.dx, run.dy)
    B_rms = run.map(lambda time: run.load("Bx", time).std())   # the same analysis at every output time, in parallel

Modules: run (simulation outputs), config (where data and results are: pta.toml), fields (derivatives,
current density), spectra, filtering, sheets (current-sheet detection, measurement and pooled thickness
statistics), statistics, parallel, plotting and animation.
"""
__version__ = "0.1.0"

from .run import Run
from . import animation, config, fields, filtering, parallel, plotting, sheets, spectra, statistics, utils
