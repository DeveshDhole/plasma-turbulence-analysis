"""Length of a current sheet: the longest path along its middle line.

    length = sheet_length(region, step=run.dx)      # in d_p, or None if the sheet should be skipped

The middle line comes from skeletonize; its longest end-to-end path is found with two shortest-path passes
(walk to the farthest pixel, then from there to the farthest again). Sheets that have closed into a ring
around a plasmoid, and blobs whose middle line is a single pixel, return None: a ring's middle line measures
its circumference, not an end-to-end span. From the Paper 1 shape pipeline.
"""
import numpy as np
from scipy import ndimage as ndi

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def centreline_length(centreline, step=1.0):
    """(length, number of pixels on the line) of the longest path along a one-pixel-wide line."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    beads = np.argwhere(centreline)
    if len(beads) < 2:
        return 0.0, 0

    number_at = {(i, j): n for n, (i, j) in enumerate(beads)}
    starts, ends, costs = [], [], []
    for n, (i, j) in enumerate(beads):
        for di, dj in NEIGHBOURS:
            m = number_at.get((i + di, j + dj))
            if m is not None:
                starts.append(n)
                ends.append(m)
                costs.append(np.hypot(di, dj) * step)
    chain = csr_matrix((costs, (starts, ends)), shape=(len(beads), len(beads)))

    far = dijkstra(chain, indices=0, directed=False)
    one_end = int(np.argmax(np.where(np.isinf(far), -1, far)))
    far = dijkstra(chain, indices=one_end, directed=False)
    far = np.where(np.isinf(far), -1, far)
    return float(far.max()), int(len(beads))


def sheet_length(region, step=1.0, max_hole_area=20):
    """Length of one sheet in the units of step, or None if it is a ring round a plasmoid (a hole bigger
    than max_hole_area cells) or a blob whose middle line is a single pixel."""
    from skimage.morphology import skeletonize    # imported here so the rest of pta works without scikit-image

    rows, columns = np.where(region)
    if rows.size == 0:
        return None
    crop = region[rows.min():rows.max() + 1, columns.min():columns.max() + 1]
    crop = np.pad(crop, 2, constant_values=False)

    filled = ndi.binary_fill_holes(crop)
    if filled.sum() - crop.sum() > max_hole_area:
        return None
    crop = filled                                 # tiny pinholes from thresholding: fill them

    length, n_pixels = centreline_length(skeletonize(crop), step)
    if n_pixels < 2 or length <= 0:
        return None
    return length
