"""
Total Occupancy Patch Module

This module provides functionality to count person bounding boxes within a defined region.
A person is considered "inside" the region if:
1. The bottom line check: corners (bottom-left, bottom-right) AND mid-point of the bottom
   line of the person bbox are inside the region.
   OR
2. The overlap check: 60% or more of the person bbox area is inside the region.
"""

import json
import os
import logging
from typing import List, Dict, Tuple, Any

logger = logging.getLogger(__name__)


def load_region_config(config_path: str = None, region_key: str = "back") -> List[Dict]:
    """
    Load the region polygon configuration from occupancy_patch.json.
    
    Args:
        config_path: Path to the occupancy_patch.json file. 
                    If None, uses the artifacts folder relative to this file.
        region_key: The key in the JSON to load (e.g., "back", "left", "kiosk", "cafe")
    
    Returns:
        List of region dictionaries, each containing 'points' and other metadata.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "artifacts", "occupancy_patch.json"
        )
    
    with open(config_path, "r") as f:
        data = json.load(f)
    
    return data.get(region_key, [])


def point_in_polygon(x: float, y: float, polygon: List[Dict[str, float]]) -> bool:
    """
    Check if a point (x, y) is inside a polygon using the ray casting algorithm.
    
    Args:
        x: X coordinate of the point
        y: Y coordinate of the point
        polygon: List of dicts with 'x' and 'y' keys representing polygon vertices
    
    Returns:
        True if point is inside the polygon, False otherwise
    """
    n = len(polygon)
    if n < 3:
        return False
    
    inside = False
    p1x, p1y = polygon[0]['x'], polygon[0]['y']
    
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
        
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        
        p1x, p1y = p2x, p2y
    
    return inside


def get_bbox_bottom_points(bbox: List[float]) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    Get the three key points of the bottom line of a bounding box:
    - Bottom-left corner
    - Bottom-right corner  
    - Mid-point of the bottom line
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
    
    Returns:
        Tuple of (bottom_left, bottom_right, bottom_mid) where each is (x, y)
    """
    x1, y1, x2, y2 = bbox
    
    bottom_left = (x1, y2)
    bottom_right = (x2, y2)
    bottom_mid = ((x1 + x2) / 2, y2)
    
    return bottom_left, bottom_right, bottom_mid


def calculate_bbox_region_overlap_ratio(bbox: List[float], polygon: List[Dict[str, float]], grid_resolution: int = 20) -> float:
    """
    Calculate what percentage of the bounding box area is inside the polygon.
    Uses a grid sampling approach for approximation.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        polygon: List of dicts with 'x' and 'y' keys representing polygon vertices
        grid_resolution: Number of sample points along each axis (higher = more accurate)
    
    Returns:
        Overlap ratio between 0.0 and 1.0
    """
    x1, y1, x2, y2 = bbox
    
    # Generate grid of sample points within the bbox
    total_points = 0
    inside_points = 0
    
    x_step = (x2 - x1) / grid_resolution if grid_resolution > 0 else 0
    y_step = (y2 - y1) / grid_resolution if grid_resolution > 0 else 0
    
    if x_step == 0 or y_step == 0:
        return 0.0
    
    for i in range(grid_resolution + 1):
        for j in range(grid_resolution + 1):
            px = x1 + i * x_step
            py = y1 + j * y_step
            total_points += 1
            
            if point_in_polygon(px, py, polygon):
                inside_points += 1
    
    return inside_points / total_points if total_points > 0 else 0.0


def is_person_in_region(
    person_bbox: List[float], 
    region_polygon: List[Dict[str, float]],
    overlap_threshold: float = 0.6
) -> Tuple[bool, str]:
    """
    Determine if a person bounding box is considered "inside" a region.
    
    Criteria:
    1. Bottom line check: All three points (bottom-left, bottom-right, bottom-mid) 
       are inside the region polygon.
    2. Overlap check: 60% or more of the bbox area overlaps with the region.
    
    Person is considered inside if EITHER criteria is met.
    
    Args:
        person_bbox: Bounding box as [x1, y1, x2, y2]
        region_polygon: List of dicts with 'x' and 'y' keys representing polygon vertices
        overlap_threshold: Minimum overlap ratio for criteria 2 (default 0.6 = 60%)
    
    Returns:
        Tuple of (is_inside, reason) where reason explains which criteria matched
    """
    # Criteria 1: Bottom line check
    bottom_left, bottom_right, bottom_mid = get_bbox_bottom_points(person_bbox)
    
    bl_inside = point_in_polygon(bottom_left[0], bottom_left[1], region_polygon)
    br_inside = point_in_polygon(bottom_right[0], bottom_right[1], region_polygon)
    bm_inside = point_in_polygon(bottom_mid[0], bottom_mid[1], region_polygon)
    
    if bl_inside and br_inside and bm_inside:
        return True, "bottom_line"
    
    # Criteria 2: Overlap check (60% of bbox inside region)
    overlap_ratio = calculate_bbox_region_overlap_ratio(person_bbox, region_polygon)
    
    if overlap_ratio >= overlap_threshold:
        return True, f"overlap_{overlap_ratio:.2f}"
    
    return False, "outside"


def count_persons_in_region(
    persons: List[Dict[str, Any]], 
    regions: List[Dict[str, Any]] = None,
    region_key: str = "back",
    config_path: str = None,
    overlap_threshold: float = 0.6
) -> Dict[str, Any]:
    """
    Count how many person bounding boxes are inside the defined region(s).
    
    Args:
        persons: List of person detections, each with a 'bbox' key [x1, y1, x2, y2]
        regions: Pre-loaded region configs. If None, loads from config file.
        region_key: Key to use when loading from config file (default "back")
        config_path: Path to occupancy_patch.json config file
        overlap_threshold: Minimum overlap ratio for the 60% overlap criteria
    
    Returns:
        Dictionary with:
        - 'total_count': Total number of persons inside the region(s)
        - 'persons_inside': List of person indices that are inside
        - 'details': Detailed results for each person (inside/outside, reason)
    """
    if regions is None:
        regions = load_region_config(config_path, region_key)
    
    if not regions:
        logger.warning(f"No regions found for key '{region_key}'")
        return {
            "total_count": 0,
            "persons_inside": [],
            "details": []
        }
    
    total_count = 0
    persons_inside = []
    details = []
    
    for idx, person in enumerate(persons):
        bbox = person.get("bbox", [])
        if len(bbox) != 4:
            details.append({
                "person_index": idx,
                "is_inside": False,
                "reason": "invalid_bbox"
            })
            continue
        
        # Check against all regions (person is inside if inside ANY region)
        is_inside = False
        matched_reason = "outside"
        matched_region_id = None
        
        for region in regions:
            polygon = region.get("points", [])
            region_id = region.get("id", "unknown")
            
            if not polygon:
                continue
            
            inside, reason = is_person_in_region(bbox, polygon, overlap_threshold)
            
            if inside:
                is_inside = True
                matched_reason = reason
                matched_region_id = region_id
                break  # Person only needs to be inside one region
        
        if is_inside:
            total_count += 1
            persons_inside.append(idx)
        
        details.append({
            "person_index": idx,
            "is_inside": is_inside,
            "reason": matched_reason,
            "region_id": matched_region_id,
            "bbox": bbox
        })
    
    return {
        "total_count": total_count,
        "persons_inside": persons_inside,
        "details": details
    }


def get_total_occupancy(
    persons: List[Dict[str, Any]],
    region_key: str = "back",
    config_path: str = None,
    overlap_threshold: float = 0.6
) -> int:
    """
    Convenience function to just get the count of persons in region.
    
    Args:
        persons: List of person detections, each with a 'bbox' key [x1, y1, x2, y2]
        region_key: Key to use when loading from config file (default "back")
        config_path: Path to occupancy_patch.json config file
        overlap_threshold: Minimum overlap ratio for the 60% overlap criteria
    
    Returns:
        Integer count of persons inside the region
    """
    result = count_persons_in_region(
        persons=persons,
        region_key=region_key,
        config_path=config_path,
        overlap_threshold=overlap_threshold
    )
    return result["total_count"]


# Example usage / Testing
if __name__ == "__main__":
    # Sample test with dummy data
    test_persons = [
        {"bbox": [300, 400, 400, 600]},  # Should be inside based on config
        {"bbox": [50, 50, 100, 100]},    # Likely outside
        {"bbox": [500, 300, 600, 500]},  # Depends on region
    ]
    
    result = count_persons_in_region(test_persons, region_key="back")
    
    print(f"Total persons in region: {result['total_count']}")
    print(f"Person indices inside: {result['persons_inside']}")
    print("\nDetails:")
    for detail in result['details']:
        print(f"  Person {detail['person_index']}: {detail['is_inside']} ({detail['reason']})")
