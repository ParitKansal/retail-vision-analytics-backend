
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from config import Config
from geometry_utils import region_points, make_eroded_mask, snip_points
from regions import REGIONS, CHECKING_REGIONS, get_region_by_name

class DrawerDetector:
    """
    Detects drawer status (Front Pass/Fail, Back Pass/Fail) by fitting rigid pairs
    of colored strips into checking regions.
    """

    def __init__(self):
        # Load regions
        self.cyan = get_region_by_name(REGIONS, 'Cyan')
        self.yellow = get_region_by_name(REGIONS, 'Yellow')
        self.pink = get_region_by_name(REGIONS, 'Pink')
        self.green = get_region_by_name(REGIONS, 'Green')
        
        # Load checking regions
        self.red_check = get_region_by_name(CHECKING_REGIONS, 'red')
        self.blue_check = get_region_by_name(CHECKING_REGIONS, 'blue')

        if not all([self.cyan, self.yellow, self.pink, self.green, self.red_check, self.blue_check]):
            raise ValueError("Failed to load one or more required regions from definitions.")

    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image frame to detect drawer markers.
        
        Returns:
            Dict containing:
            - 'front_fit': Result dict for Front (Cyan+Yellow) or None
            - 'back_fit': Result dict for Back (Pink+Green) or None
        """
        front_fit = self._try_fit_pair(
            image, 
            strip_bright=self.yellow, 
            strip_dark=self.cyan, 
            checking_region=self.red_check,
            bright_min_thresh=Config.BRIGHT_MIN_THRESHOLD,
            dark_max_thresh=Config.DARK_MAX_THRESHOLD
        )

        back_fit = self._try_fit_pair(
            image, 
            strip_bright=self.green, 
            strip_dark=self.pink, 
            checking_region=self.blue_check,
            bright_min_thresh=Config.BRIGHT_MIN_THRESHOLD,
            dark_max_thresh=Config.DARK_MAX_THRESHOLD
        )

        return {
            'front_fit': front_fit,
            'back_fit': back_fit
        }
    
    def _try_fit_pair(self, 
                      image: np.ndarray, 
                      strip_bright: Dict, 
                      strip_dark: Dict, 
                      checking_region: Dict,
                      bright_min_thresh: int,
                      dark_max_thresh: int) -> Optional[Dict[str, Any]]:
        """
        Fit a rigid pair of strips inside a checking region.
        The pair (bright + dark) is treated as a rigid unit.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h_img, w_img = gray.shape

        # Build rigid pair geometry relative to combined centroid
        pts_bright = region_points(strip_bright)
        pts_dark = region_points(strip_dark)
        all_pts = np.vstack([pts_bright, pts_dark])
        pair_centroid = np.mean(all_pts, axis=0)
        rel_bright = pts_bright - pair_centroid
        rel_dark = pts_dark - pair_centroid

        # Compute length axis via PCA (principal direction = length, secondary = width)
        cov = np.cov(all_pts.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # eigh returns sorted ascending, so last eigenvector = largest eigenvalue = length axis
        length_axis = eigenvectors[:, -1]
        width_axis = eigenvectors[:, 0]

        # Checking region mask and bounding box
        check_pts = region_points(checking_region).astype(np.int32).reshape(-1, 1, 2)
        check_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        cv2.fillPoly(check_mask, [check_pts], 255)
        bx, by, bw, bh = cv2.boundingRect(check_pts)

        # Crop for efficiency
        crop_gray = gray[by:by+bh, bx:bx+bw]
        crop_check = check_mask[by:by+bh, bx:bx+bw]

        # Iteration parameters from Config
        scales = Config.FIT_SCALES
        angle_max = Config.FIT_ANGLE_MAX
        angle_step = Config.FIT_ANGLE_STEP

        for scale in scales:
            # Angles: 0, 1, -1, 2, -2, ... up to max
            angles_to_try = [0]
            for a in range(angle_step, angle_max + 1, angle_step):
                angles_to_try.append(a)
                angles_to_try.append(-a)

            for angle in angles_to_try:
                angle_rad = np.radians(angle)
                cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

                # Snip along length only, then rotate
                t_bright = snip_points(rel_bright, scale, length_axis, width_axis) @ rot.T
                t_dark = snip_points(rel_dark, scale, length_axis, width_axis) @ rot.T

                # Scan the cropped checking region
                # Optimizable: Only scan pixels that are inside the checking region
                # We can iterate over all pixels in crop, but check mask first.
                
                # Note: Python loops are slow. In a production C++ implementation, this would be optimized.
                # Here we stick to logic provided in experiment 3.0/4.0.
                for scan_y in range(bh):
                    for scan_x in range(bw):
                        if crop_check[scan_y, scan_x] == 0:
                            continue

                        center = np.array([scan_x, scan_y], dtype=np.float64)
                        abs_b = (t_bright + center).astype(np.int32)
                        abs_d = (t_dark + center).astype(np.int32)

                        # Quick bounds check in crop space
                        # Check if any point is outside [0, bw] or [0, bh]
                        b_min = np.min(abs_b, axis=0)
                        b_max = np.max(abs_b, axis=0)
                        d_min = np.min(abs_d, axis=0)
                        d_max = np.max(abs_d, axis=0)
                        
                        if (b_min[0] < 0 or b_max[0] >= bw or b_min[1] < 0 or b_max[1] >= bh or
                            d_min[0] < 0 or d_max[0] >= bw or d_min[1] < 0 or d_max[1] >= bh):
                            continue

                        # All vertices must be inside checking region mask
                        # (Checking vertices is a heuristic; technically should check all pixels, 
                        # but vertices usually sufficient for convex-ish shapes)
                        all_inside = True
                        for pt in np.vstack([abs_b, abs_d]):
                            if crop_check[pt[1], pt[0]] == 0:
                                all_inside = False
                                break
                        if not all_inside:
                            continue

                        # Create eroded masks (inner ROI)
                        erode_px = Config.ERODE_PIXELS
                        mask_b = make_eroded_mask((bh, bw), abs_b, erode_px=erode_px)
                        mask_d = make_eroded_mask((bh, bw), abs_d, erode_px=erode_px)

                        pix_b = crop_gray[mask_b == 255]
                        pix_d = crop_gray[mask_d == 255]

                        if pix_b.size == 0 or pix_d.size == 0:
                            continue

                        max_b = int(np.max(pix_b))
                        max_d = int(np.max(pix_d))
                        avg_b = float(np.mean(pix_b))
                        avg_d = float(np.mean(pix_d))

                        # Check Thresholds
                        # Bright Strip: Must be bright enough
                        pass_bright = (max_b >= bright_min_thresh and avg_b >= Config.BRIGHT_AVG_MIN_THRESHOLD)
                        # Dark Strip: Must be dark enough
                        pass_dark = (max_d <= dark_max_thresh and avg_d < Config.DARK_AVG_MAX_THRESHOLD)

                        if pass_bright and pass_dark:
                            # Transform points back to full image coordinates
                            img_bright = abs_b + np.array([bx, by])
                            img_dark = abs_d + np.array([bx, by])
                            
                            return {
                                'center': (float(scan_x + bx), float(scan_y + by)),
                                'angle': angle,
                                'scale': scale,
                                'max_bright': max_b,
                                'max_dark': max_d,
                                'min_bright': int(np.min(pix_b)),
                                'min_dark': int(np.min(pix_d)),
                                'avg_bright': avg_b,
                                'avg_dark': avg_d,
                                'points_bright': img_bright,
                                'points_dark': img_dark
                            }

        return None

        return None
