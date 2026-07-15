"""
Consolidated detection methods for STAR particle analysis.
Combines HSV, LAB, and filtering approaches into a single module.
"""

import cv2
import numpy as np


class DetectionMethods:
    """Container for all detection methods with consistent interface"""
    
    @staticmethod
    def method_1_expanded_hsv(image):
        """Expand HSV range for purple detection"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower, upper = np.array([120, 80, 5]), np.array([170, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = np.ones((2, 2), np.uint8)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    @staticmethod
    def method_2_multiple_ranges(image):
        """Multiple overlapping HSV ranges"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ranges = [
            ([120, 100, 10], [140, 255, 255]),
            ([140, 100, 10], [160, 255, 255]),
            ([160, 100, 10], [170, 255, 255]),
            ([125, 80, 5], [165, 255, 255]),
        ]
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            combined |= cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    @staticmethod
    def method_3_saturation_based(image):
        """Saturation and hue filtering"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat_mask = cv2.threshold(hsv[:, :, 1], 100, 255, cv2.THRESH_BINARY)[1]
        hue_mask = cv2.inRange(hsv[:, :, 0], 120, 170)
        return cv2.bitwise_and(sat_mask, hue_mask)
    
    @staticmethod
    def method_4_adaptive_threshold(image):
        """Adaptive threshold on saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        adaptive = cv2.adaptiveThreshold(
            sat, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
        )
        hue_mask = cv2.inRange(hsv[:, :, 0], 120, 170)
        combined = cv2.bitwise_and(adaptive, hue_mask)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(combined, kernel, iterations=1)
    
    @staticmethod
    def method_5_statistical(image):
        """Statistical outlier detection based on saturation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        s_mean, s_std = np.mean(sat), np.std(sat)
        high_sat = sat > (s_mean + 2 * s_std)
        hue_mask = cv2.inRange(hsv[:, :, 0], 120, 170)
        return (np.logical_and(high_sat, hue_mask).astype(np.uint8) * 255)
    
    @staticmethod
    def method_6_lab_color(image):
        """LAB color space detection"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return (np.logical_and(lab[:, :, 1] > 135, lab[:, :, 2] < 120).astype(np.uint8) * 255)
    
    @staticmethod
    def optimized_combined(image, sat_threshold=60, hue_range=(100, 225), 
                          purple_lower=(120, 90, 5), purple_upper=(170, 255, 255)):
        """Optimized detection combining multiple approaches"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Saturation + Hue approach
        sat_mask = cv2.threshold(hsv[:, :, 1], sat_threshold, 255, cv2.THRESH_BINARY)[1]
        hue_mask = cv2.inRange(hsv[:, :, 0], *hue_range)
        mask1 = cv2.bitwise_and(sat_mask, hue_mask)
        
        # Purple range approach
        mask2 = cv2.inRange(hsv, np.array(purple_lower), np.array(purple_upper))
        
        # Combine and clean
        combined = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((1, 1), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    
    @classmethod
    def get_all_methods(cls):
        """Return list of (method_function, name) tuples"""
        return [
            (cls.method_1_expanded_hsv, "Expanded HSV Range"),
            (cls.method_2_multiple_ranges, "Multiple HSV Ranges"),
            (cls.method_3_saturation_based, "Saturation-Based"),
            (cls.method_4_adaptive_threshold, "Adaptive Threshold"),
            (cls.method_5_statistical, "Statistical Outlier"),
            (cls.method_6_lab_color, "LAB Color Space"),
        ]
