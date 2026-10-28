"""Current-sheet detection.

detect_current_sheets: peak flooding (from the Paper 1 pipeline notebook,
CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb). By default a sheet is grown from each peak above
5 x the field's mean, down to 0.5 x the peak; both thresholds can be changed.
quantile_mask: the field above one of its quantiles, cleaned morphologically (the jpar_jperp method).
"""
import numpy as np
from scipy.ndimage import label


def detect_current_sheets(mod_Jz, global_threshold=None, local_fraction=0.5):
    """Sheets of a non-negative field (|J|, J^2, ...) by peak flooding: from the highest remaining peak, the
    4-connected region above local_fraction x the peak is one sheet; repeat until the highest remaining peak is
    below global_threshold (default 5 x the field's mean). Returns a list of sheets, each a list of (i, j)."""

    J2_rms = np.mean(mod_Jz)
    
    # size of grid
    nx, ny = mod_Jz.shape
    
    # global maximum |Jz|
    max_global = np.max(mod_Jz)
    
    # threshold to stop searching for new sheets
    if global_threshold is None:
        global_threshold = 5 * J2_rms
    
    # keep track of points already assigned to sheets
    visited = np.zeros_like(mod_Jz, dtype=bool)
    
    # list to store all current sheets
    current_sheets = []

    while True:
        
        # mask out already visited cells
        masked_field = np.where(visited, -np.inf, mod_Jz)
        
        # location of largest remaining |Jz|
        max_index = np.unravel_index(np.argmax(masked_field), mod_Jz.shape)
        peak_value = mod_Jz[max_index]
        
        # stop condition
        if peak_value < global_threshold:
            break
        
        # threshold for this sheet
        local_threshold = local_fraction * peak_value
        
        # stack for depth-first search
        stack = [max_index]
        
        # store indices belonging to this sheet
        sheet_points = []
        
        while stack:
            
            i, j = stack.pop()
            
            if visited[i, j]:
                continue
            
            if mod_Jz[i, j] <= local_threshold:
                continue
            
            visited[i, j] = True
            sheet_points.append((i, j))
            
            # check 4 neighbors
            neighbors = [
                (i+1, j),
                (i-1, j),
                (i, j+1),
                (i, j-1)
            ]
            
            for ni, nj in neighbors:
                if 0 <= ni < nx and 0 <= nj < ny:
                    if not visited[ni, nj]:
                        stack.append((ni, nj))
        
        current_sheets.append(sheet_points)

    return current_sheets


def get_max_coords(cleaned_mask, J2):
    """Per-region argmax of |J2|."""
    labeled_mask, num_features = label(cleaned_mask)
    max_coords = []
    for i in range(1, num_features + 1):
        region = (labeled_mask == i)
        region_vals = J2[region]
        if region_vals.size == 0:
            continue
        max_pos = np.argmax(np.abs(region_vals))
        xs_r, ys_r = np.where(region)
        max_coords.append((xs_r[max_pos], ys_r[max_pos])) 
    return max_coords, labeled_mask


def quantile_mask(field, quantile, smoothing_radius=5, min_area=300):
    """1 where |field| is above its quantile, after opening and closing with a disk of smoothing_radius cells and
    removing regions smaller than min_area cells; 0 elsewhere (needs scikit-image)."""
    from skimage import morphology               # imported here so the rest of pta works without scikit-image

    binary_mask = np.abs(field) > np.quantile(field, quantile)
    struct = morphology.disk(smoothing_radius)
    smoothed_mask = morphology.opening(binary_mask, struct)          # the same as binary_opening on a binary mask
    smoothed_mask = morphology.closing(smoothed_mask, struct)
    try:                                         # scikit-image >= 0.26: remove regions of at most max_size cells
        cleaned_mask = morphology.remove_small_objects(smoothed_mask.astype(bool), max_size=min_area - 1)
    except TypeError:                            # older scikit-image: remove regions of fewer than min_size cells
        cleaned_mask = morphology.remove_small_objects(smoothed_mask.astype(bool), min_size=min_area)
    return cleaned_mask.astype(int)


def sheets_to_mask(sheets, shape):
    """The mask (1 on a sheet, 0 elsewhere) of the sheets returned by detect_current_sheets."""
    mask = np.zeros(shape, dtype=int)
    for sheet in sheets:
        if sheet:
            rows, columns = zip(*sheet)
            mask[list(rows), list(columns)] = 1
    return mask
