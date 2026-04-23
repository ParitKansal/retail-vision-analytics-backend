
import numpy as np
import cv2
from typing import List, Dict, Any

def region_points(region: Dict[str, Any]) -> np.ndarray:
    """Extracts points from a region dictionary into a numpy array of shape (N, 2)."""
    return np.array([[p['x'], p['y']] for p in region['points']])

def make_eroded_mask(shape: tuple, pts_int: np.ndarray, erode_px: int = 1) -> np.ndarray:
    """Create a filled polygon mask and optionally erode it to exclude boundary pixels."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [pts_int.reshape(-1, 1, 2)], 255)
    # Note: Traditional erosion excluded for now based on experiment 4.0 usage, 
    # but the parameter is kept for compatibility if needed in future.
    # If erosion logic is uncommented in source, we can add it here.
    # The source code had the erosion lines commented out.
    return mask

def snip_points(rel_pts: np.ndarray, scale: float, length_axis: np.ndarray, width_axis: np.ndarray) -> np.ndarray:
    """
    Scale points only along the length axis (principal direction).
    Used for 'snipping' or shortening the strip without changing its width.
    """
    if abs(scale - 1.0) < 0.001:
        return rel_pts.copy()
    
    result = np.zeros_like(rel_pts)
    for i, pt in enumerate(rel_pts):
        along_length = np.dot(pt, length_axis)
        along_width = np.dot(pt, width_axis)
        # Scale only the length component
        result[i] = (along_length * scale) * length_axis + along_width * width_axis
    return result
