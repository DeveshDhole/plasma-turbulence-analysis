"""Movies of a field over time: one place for every animation in the repository.

    from pta import animation
    animation.animate_field(run, lambda time: run.load("Bx", time), RESULTS / "Bx.mp4")

    # any frames, not just fields:
    animation.animate(draw, frames, RESULTS / "spectra.mp4")

The frames are computed in parallel (pta.parallel), then drawn one after another, so a slow analysis per
frame costs one pass over the snapshots. Writing needs ffmpeg.
"""
import numpy as np

from .parallel import parallel_map
from .plotting import colour_limits, save_figure, show_field

DEFAULT_FPS = 12
DEFAULT_BITRATE = 1800


def _writer(fps, bitrate):
    from matplotlib.animation import FFMpegWriter
    return FFMpegWriter(fps=fps, bitrate=bitrate)


def animate(draw, frames, path, setup=None, fps=DEFAULT_FPS, bitrate=DEFAULT_BITRATE, figure=None, blit=False):
    """Write an mp4 from `frames` (anything iterable): setup(first frame) -> (figure, state), then
    draw(state, frame) for each. Without setup, draw(frame) is called on a figure you pass in.
    Returns the path."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    frames = list(frames)
    if not frames:
        raise ValueError("no frames to animate")
    state = None
    if setup is not None:
        figure, state = setup(frames[0])
    if figure is None:
        raise ValueError("pass a figure, or a setup that returns one")

    def update(frame):
        drawn = draw(frame) if state is None else draw(state, frame)
        return drawn if isinstance(drawn, (list, tuple)) else []

    movie = FuncAnimation(figure, update, frames=frames, interval=1000 / fps, blit=blit)
    movie.save(str(path), writer=_writer(fps, bitrate))
    plt.close(figure)
    print(f"saved {path}")
    return path


def animate_field(run, field_at, path, times=None, dx=None, dy=None, title=None, label=None, centre=None,
                  limits=None, limits_from=None, mask=False, corner=(0.0, 0.0), figsize=(8, 6), workers=None,
                  fps=DEFAULT_FPS, bitrate=DEFAULT_BITRATE, **show_options):
    """A movie of a 2D field over the output times of a run (x along axis 0, axes in d_p).

    field_at    : time -> the field at that time (computed in parallel over the times)
    times       : which output times (default: all of them)
    limits      : fixed colour range; by default from the snapshot at limits_from (default: the first time)
    title       : time -> the title of that frame, or a fixed string (default: "t = <time>")
    centre, label, mask, corner, **show_options : passed to pta.plotting.show_field
    """
    import matplotlib.pyplot as plt

    times = list(run.times if times is None else times)
    dx = run.dx if dx is None else dx
    dy = run.dy if dy is None else dy
    if limits is None:
        limits = colour_limits(field_at(limits_from if limits_from is not None else times[0]), centre)
    fields = parallel_map(field_at, times, workers=workers)

    def frame_title(time):
        if title is None:
            return f"t = {time}"
        return title(time) if callable(title) else title

    figure, ax = plt.subplots(figsize=figsize)
    image = show_field(ax, fields[0], dx, dy, centre=centre, limits=limits, label=label, mask=mask,
                       corner=corner, **show_options)
    heading = ax.set_title(frame_title(times[0]))
    figure.tight_layout()

    def draw(index):
        image.set_data(fields[index].T)
        heading.set_text(frame_title(times[index]))
        return [image, heading]

    return animate(draw, range(len(times)), path, figure=figure, fps=fps, bitrate=bitrate)


def animate_lines(x_values, series_at, path, frames=None, labels=None, xlabel=None, ylabel=None, title=None,
                  xlim=None, ylim=None, log=False, figsize=(7, 5), workers=None, fps=DEFAULT_FPS,
                  bitrate=DEFAULT_BITRATE):
    """A movie of one or more curves over time, e.g. spectra: series_at(frame) -> one array, or a tuple of
    arrays, one per curve. x_values is the common x axis (or frame -> x values)."""
    import matplotlib.pyplot as plt

    frames = list(frames)
    computed = parallel_map(series_at, frames, workers=workers)
    first = computed[0] if isinstance(computed[0], (list, tuple)) else (computed[0],)
    figure, ax = plt.subplots(figsize=figsize)
    plot = ax.loglog if log else ax.plot
    lines = [plot([], [], label=(labels[number] if labels else None))[0] for number in range(len(first))]
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if labels:
        ax.legend()
    heading = ax.set_title("")
    figure.tight_layout()

    def draw(index):
        values = computed[index]
        values = values if isinstance(values, (list, tuple)) else (values,)
        x = x_values(frames[index]) if callable(x_values) else x_values
        for line, series in zip(lines, values):
            line.set_data(x, series)
        heading.set_text(title(frames[index]) if callable(title) else (title or f"t = {frames[index]}"))
        return lines + [heading]

    return animate(draw, range(len(frames)), path, figure=figure, fps=fps, bitrate=bitrate)
