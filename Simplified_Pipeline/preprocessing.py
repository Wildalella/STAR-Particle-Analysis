"""
Image preprocessing and analysis utilities.
Handles calibration, exclusion zones, and contour detection.
"""

import cv2
import numpy as np
from scipy.spatial import KDTree


def calibrate_scale(image, roi_coords, known_length_mm):
    """
    Calibrate pixel-to-mm scale from scale bar.
    
    Args:
        image: Input image (BGR)
        roi_coords: (y_start, y_end, x_start, x_end) of scale bar
        known_length_mm: Physical length of scale bar in mm
        
    Returns:
        float: pixels per mm, or None if calibration fails
    """
    y_start, y_end, x_start, x_end = roi_coords
    
    # Validate ROI bounds
    if not all([0 <= y_start < y_end <= image.shape[0],
                0 <= x_start < x_end <= image.shape[1]]):
        print("Error: Scale bar ROI out of bounds")
        return None
    
    roi = image[y_start:y_end, x_start:x_end]
    if roi.size == 0:
        print("Error: Scale bar ROI is empty")
        return None
    
    # Find scale bar contour
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Error: No contours in scale bar")
        return None
    
    # Get largest contour (the scale bar)
    scale_contour = max(contours, key=cv2.contourArea)
    _, _, w, _ = cv2.boundingRect(scale_contour)
    
    if w == 0:
        print("Error: Scale bar has zero width")
        return None
    
    pixels_per_mm = w / known_length_mm
    print(f"Calibration: {pixels_per_mm:.2f} pixels/mm")
    return pixels_per_mm


def create_exclusion_mask(image_shape, exclusion_zones):
    """
    Create binary mask with excluded zones set to 0.
    
    Args:
        image_shape: Shape of image (H, W, C)
        exclusion_zones: List of (y_start, y_end, x_start, x_end) tuples
        
    Returns:
        np.ndarray: Binary mask (255 where included, 0 where excluded)
    """
    mask = np.ones(image_shape[:2], dtype=np.uint8) * 255
    
    for y_start, y_end, x_start, x_end in exclusion_zones:
        # Clamp to image bounds
        y_start = max(0, min(y_start, image_shape[0]))
        y_end = max(0, min(y_end, image_shape[0]))
        x_start = max(0, min(x_start, image_shape[1]))
        x_end = max(0, min(x_end, image_shape[1]))
        
        mask[y_start:y_end, x_start:x_end] = 0
    
    return mask


def apply_exclusion_mask(detection_mask, exclusion_mask):
    """Apply exclusion mask to detection mask"""
    return cv2.bitwise_and(detection_mask, exclusion_mask)


def detect_contours(mask, image=None, min_area=1, max_area_percentile=99, max_area_multiplier=3):
    """
    Detect contours and compute centroids with flexible filtering.
    
    Args:
        mask: Binary detection mask
        image: Optional input image (not required)
        min_area: Minimum contour area
        max_area_percentile: Percentile for max area calculation
        max_area_multiplier: Multiplier for max area threshold
        
    Returns:
        tuple: (centroids_list, contours_list) where centroids are (x, y) tuples
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [], []
    
    # Determine area thresholds
    areas = [cv2.contourArea(cnt) for cnt in contours]
    max_area_threshold = max(np.percentile(areas, max_area_percentile) * max_area_multiplier, 5000)
    
    centroids = []
    valid_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area_threshold:
            # Compute centroid
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                centroids.append((cx, cy))
                valid_contours.append(contour)
    
    return centroids, valid_contours


def compute_nearest_neighbor_distances(centroids, pixels_per_mm):
    """
    Compute nearest neighbor distances for all centroids.
    
    Args:
        centroids: List of (x, y) tuples
        pixels_per_mm: Scale conversion factor
        
    Returns:
        list: Distances in mm to nearest neighbor for each centroid
    """
    if len(centroids) < 2:
        return []
    
    tree = KDTree(centroids)
    distances_mm = []
    
    for point in centroids:
        # k=2 because first neighbor is the point itself
        dists, _ = tree.query(point, k=2)
        distances_mm.append(dists[1] / pixels_per_mm)
    
    return distances_mm
