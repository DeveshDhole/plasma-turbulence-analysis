"""Current-sheet detection (global threshold 5 x J2_rms, local threshold 0.5 x peak).

Function copied verbatim from the Paper 1 pipeline notebook
(CSDetectionMaxTurbulenceFullPipelineWithKCutoff_Beta1.ipynb); behaviour unchanged.
"""
import numpy as np


def detect_current_sheets(mod_Jz):

    J2_rms = np.mean(mod_Jz)
    
    # size of grid
    nx, ny = mod_Jz.shape
    
    # global maximum |Jz|
    max_global = np.max(mod_Jz)
    
    # threshold to stop searching for new sheets
    # global_threshold = 0.1 * max_global
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
        local_threshold = 0.5 * peak_value
        
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
