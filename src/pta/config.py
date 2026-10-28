"""Where a project's simulation data, results and caches are: read from a pta.toml file.

pta.toml is the file named by the environment variable PTA_CONFIG, or else the first pta.toml found in the
working directory or one of its parents. Without one, results and caches go to ./results and ./cache.

    [paths]                      # folders; relative ones are relative to the folder holding pta.toml
    data = "/path/to/simulations"
    results = "results"
    cache = "cache"
    scans = "{data}/scans"       # an entry can use another one as {name}

    [runs]                       # run folders by name, with {parameters} filled in by run()
    beta_scan = "{data}/Brms_{brms}/beta_p_{beta_p}/beta_e_{beta_e}"

    [run_defaults.beta_scan]     # default parameters of a layout
    brms = 0.24

    [run_options.beta_scan]      # options for pta.Run (tag, pattern, box_length, ...)
    tag = "ApJ"

    [scans.resistivity]          # named sets of runs, for scan()
    layout = "resistivity"
    runs = [{ name = "5e-4", label = "eta = 5e-4" }]

An environment variable PTA_<NAME> overrides the [paths] entry <name> (PTA_DATA, PTA_RESULTS, ...).

    from pta import config
    run = config.run("beta_scan", beta_p=1)              # a pta.Run
    for run, label in config.scan("resistivity"): ...
    RESULTS = config.results_dir("my_project", "my_notebook")
"""
import os
import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:                # Python < 3.11
    import tomli as tomllib

from .run import Run

CONFIG_NAME = "pta.toml"
_parsed = {}


def config_file():
    """The pta.toml in use, or None."""
    if os.environ.get("PTA_CONFIG"):
        path = Path(os.environ["PTA_CONFIG"]).expanduser()
        return path / CONFIG_NAME if path.is_dir() else path
    for folder in (Path.cwd(), *Path.cwd().parents):
        if (folder / CONFIG_NAME).is_file():
            return folder / CONFIG_NAME
    return None


def project_root():
    """The folder holding pta.toml (the working directory without one)."""
    file = config_file()
    return file.parent if file else Path.cwd()


def settings():
    """The contents of pta.toml as a dict ({} without one). Read once; reload() reads it again."""
    file = config_file()
    if file is None:
        return {}
    key = str(file.resolve())
    if key not in _parsed:
        with open(file, "rb") as handle:
            _parsed[key] = tomllib.load(handle)
    return _parsed[key]


def reload():
    """Forget the parsed pta.toml, so the next call reads the file again."""
    _parsed.clear()


def paths():
    """Every [paths] entry as an absolute Path, after the PTA_<NAME> overrides and the {name} substitutions."""
    raw = {"results": "results", "cache": "cache", **settings().get("paths", {})}
    raw.update({name: os.environ[f"PTA_{name.upper()}"] for name in raw if os.environ.get(f"PTA_{name.upper()}")})
    resolved = {}

    def resolve(name, chain=()):
        if name in resolved:
            return resolved[name]
        if name in chain:
            raise ValueError("[paths] entries refer to each other in a loop: " + " -> ".join(chain + (name,)))
        value = str(raw[name])
        for other in re.findall(r"\{(\w+)\}", value):
            if other not in raw:
                raise KeyError(f"[paths] {name} uses {{{other}}}, which is not a [paths] entry")
            value = value.replace("{" + other + "}", str(resolve(other, chain + (name,))))
        path = Path(os.path.expandvars(value)).expanduser()
        resolved[name] = path if path.is_absolute() else project_root() / path
        return resolved[name]

    for name in raw:
        resolve(name)
    return resolved


def path(name):
    """One [paths] entry, e.g. path("data")."""
    entries = paths()
    if name not in entries:
        raise KeyError(f"no [paths] entry {name!r} (entries: {', '.join(entries)}); see {config_file()}")
    return entries[name]


def _folder(name, parts, create):
    folder = path(name).joinpath(*map(str, parts))
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def results_dir(*parts, create=True):
    """results_dir("project", "notebook") -> <results>/project/notebook (created if missing)."""
    return _folder("results", parts, create)


def cache_dir(*parts, create=True):
    """cache_dir("project", "filtered") -> <cache>/project/filtered (created if missing)."""
    return _folder("cache", parts, create)


def run_path(layout, **parameters):
    """The folder of a [runs] layout, with its {parameters}, [run_defaults] and [paths] entries filled in."""
    layouts = settings().get("runs", {})
    if layout not in layouts:
        raise KeyError(f"no run layout {layout!r} in [runs] of {config_file()} (layouts: {', '.join(layouts)})")
    values = {**{name: str(folder) for name, folder in paths().items()},
              **settings().get("run_defaults", {}).get(layout, {}), **parameters}
    folder = Path(layouts[layout].format(**values)).expanduser()
    return folder if folder.is_absolute() else project_root() / folder


def run(layout, **parameters):
    """A pta.Run for a [runs] layout, with the layout's [run_options]."""
    return Run(run_path(layout, **parameters), **settings().get("run_options", {}).get(layout, {}))


def scan(name):
    """[(Run, label), ...] for the runs of a [scans.<name>] entry."""
    entry = settings().get("scans", {}).get(name)
    if entry is None:
        raise KeyError(f"no [scans.{name}] in {config_file()}")
    runs = []
    for item in entry["runs"]:
        parameters = {key: value for key, value in item.items() if key != "label"}
        folder = run(entry["layout"], **parameters)
        runs.append((folder, item.get("label", folder.name)))
    return runs
