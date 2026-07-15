"""
Visualization utilities for STAR analysis results.
Includes histograms, heatmaps, overlays, and method comparisons.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_figure(filepath, figsize=(10, 6), dpi=150):
    """Context manager for consistent figure saving"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {filepath}")
    plt.close()


def plot_distance_histogram(distances_mm, image_name, output_folder="outputs"):
    """
    Create and save distance histogram.
    
    Args:
        distances_mm: List of nearest-neighbor distances in mm
        image_name: Base name for output file
        output_folder: Output directory
    """
    plt.figure(figsize=(10, 6))
    plt.hist(distances_mm, bins=30, color='mediumslateblue', edgecolor='black', alpha=0.7)
    
    avg = np.mean(distances_mm)
    plt.axvline(avg, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg:.3f} mm')
    
    plt.title(f'Nearest-Neighbor Distance Distribution\n{image_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Distance (mm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_folder) / f"{image_name}_distance_histogram.png"
    save_figure(str(output_path))


def plot_heatmap(centroids, image_shape, image_name, output_folder="outputs"):
    """
    Create density heatmap of dot locations.
    
    Args:
        centroids: List of (x, y) centroid coordinates
        image_shape: Shape of original image (H, W, C)
        image_name: Base name for output file
        output_folder: Output directory
    """
    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    
    for cx, cy in centroids:
        cv2.circle(heatmap, (cx, cy), 20, 1, -1)
    
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8) if heatmap.max() > 0 else heatmap
    
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title(f'Dot Density Heatmap\n{image_name}', fontsize=14, fontweight='bold')
    plt.axis('off')
    
    output_path = Path(output_folder) / f"{image_name}_heatmap.png"
    save_figure(str(output_path))


def visualize_detection_result(image, contours, centroids, stats, base_name, 
                               exclusion_zones=None, output_folder="outputs"):
    """
    Create final detection visualization with statistics overlay.
    
    Args:
        image: Input image (BGR)
        contours: List of detected contours
        centroids: List of (x, y) centroids
        stats: Dict with keys: dot_count, avg_distance, min_distance, max_distance, std_distance, unit
        base_name: Base name for output file
        exclusion_zones: Optional list of (y_start, y_end, x_start, x_end) exclusion zones
        output_folder: Output directory
    """
    result_img = image.copy()
    
    # Draw exclusion zones
    if exclusion_zones:
        for y_start, y_end, x_start, x_end in exclusion_zones:
            cv2.rectangle(result_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.putText(result_img, 'EXCLUDED', (x_start + 5, y_start + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Draw contours and centroids
    cv2.drawContours(result_img, contours, -1, (0, 255, 0), 2)
    for cx, cy in centroids:
        cv2.circle(result_img, (cx, cy), 5, (0, 0, 255), 2)
    
    # Add statistics text
    unit = stats.get('unit', 'pixels')
    stats_text = [
        f"Dots: {stats['dot_count']}",
        f"Avg Dist: {stats['avg_distance']:.3f} {unit}" if stats.get('avg_distance') else "Avg Dist: N/A",
        f"Min: {stats['min_distance']:.3f} {unit}" if stats.get('min_distance') else None,
        f"Max: {stats['max_distance']:.3f} {unit}" if stats.get('max_distance') else None,
        f"Std: {stats['std_distance']:.3f} {unit}" if stats.get('std_distance') else None,
    ]
    
    y_offset = 30
    for text in filter(None, stats_text):
        # Black outline for contrast
        cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(result_img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
    
    output_path = Path(output_folder) / f"{base_name}_detected.png"
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result_img)
    print(f"Detection overlay saved: {output_path}")


def compare_detection_methods(image, detection_methods_list, exclusion_mask, 
                             image_name, output_folder="outputs"):
    """
    Compare multiple detection methods side-by-side.
    
    Args:
        image: Input image (BGR)
        detection_methods_list: List of (method_func, method_name) tuples
        exclusion_mask: Binary exclusion mask
        image_name: Base name for output file
        output_folder: Output directory
        
    Returns:
        dict: Results with format {method_name: {'count': int, 'centroids': list, ...}}
    """
    from preprocessing import detect_contours, apply_exclusion_mask
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (method_func, method_name) in enumerate(detection_methods_list):
        # Apply method
        mask = method_func(image)
        mask = apply_exclusion_mask(mask, exclusion_mask)
        centroids, contours = detect_contours(mask)
        
        results[method_name] = {
            'centroids': centroids,
            'contours': contours,
            'count': len(centroids),
            'mask': mask
        }
        
        # Visualize
        viz_img = image.copy()
        cv2.drawContours(viz_img, contours, -1, (0, 255, 0), 1)
        for cx, cy in centroids:
            cv2.circle(viz_img, (cx, cy), 3, (0, 0, 255), -1)
        
        viz_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(viz_rgb)
        axes[i].set_title(f'{method_name}\n{len(centroids)} dots detected')
        axes[i].axis('off')
    
    output_path = Path(output_folder) / f"{image_name}_method_comparison.png"
    save_figure(str(output_path), figsize=(18, 12))
    
    # Print summary
    print("\n" + "="*50)
    print("Detection Method Comparison")
    print("="*50)
    for name, data in results.items():
        print(f"  {name:30s}: {data['count']:4d} dots")
    print("="*50)
    
    return results
