"""Current sheets: detection (detection.py), measurement of their thickness and length (thickness.py), and cuts
across a sheet (profiles.py)."""
from .detection import detect_current_sheets, get_max_coords, quantile_mask, sheets_to_mask
from .profiles import (descent_direction, fit_harris, fwhm_nearest_samples, grow_region, harris_profile,
                       line_profile, width_along)
from .thickness import bell, caliper_length, crop_with_border, gaussian, region_thicknesses, thickness_m1
