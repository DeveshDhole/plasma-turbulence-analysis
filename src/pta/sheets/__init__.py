"""Current sheets: detection (detection.py), thickness and length of one sheet (thickness.py, length.py), cuts
across a sheet (profiles.py), and thickness and shape pooled over many snapshots (pooling.py)."""
from .detection import detect_current_sheets, get_max_coords, quantile_mask, sheets_to_mask
from .length import centreline_length, sheet_length
from .pooling import (PooledShape, PooledThickness, filtered_field, label_sheets, parallel_current, pooled_shape,
                      pooled_thickness, snapshot_shape, snapshot_thickness)
from .profiles import (descent_direction, fit_harris, fwhm_nearest_samples, harris_profile, line_profile,
                       width_along)
from .thickness import bell, caliper_length, crop_with_border, gaussian, region_thicknesses, thickness_m1
