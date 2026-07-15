"""
Main image processing pipeline for STAR particle analysis.
Simplified orchestration of detection, analysis, and visualization.
Includes both classical and ML detection methods.
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from detection_methods import DetectionMethods
from preprocessing import (
    calibrate_scale, create_exclusion_mask, apply_exclusion_mask,
    detect_contours, compute_nearest_neighbor_distances
)
from visualization import (
    plot_distance_histogram, plot_heatmap, visualize_detection_result,
    compare_detection_methods
)
from ml_detection import run_ml_detection, convert_predictions_to_centroids


def process_image(image_path, output_folder="outputs", config=None):
    """
    Process a single image with full analysis pipeline.
    Includes both classical and ML detection methods.
    
    Args:
        image_path: Path to image file
        output_folder: Where to save results
        config: Config dict with keys:
                - scale_bar_roi_coords: (y_start, y_end, x_start, x_end)
                - scale_bar_mm: Physical length of scale bar
                - exclusion_zones: List of exclusion zone tuples
                - roboflow_api_key: (optional) Roboflow API key
                - roboflow_workspace: (optional) Roboflow workspace name
                - roboflow_workflow: (optional) Roboflow workflow ID
                
    Returns:
        dict: Analysis results including both classical and ML detection
    """
    if config is None:
        config = {}
    
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    base_name = image_path.stem
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print(f"{'='*60}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Cannot read {image_path}")
        return None
    
    # Extract config with defaults
    scale_roi = config.get('scale_bar_roi_coords', (315, 330, 945, 1150))
    scale_mm = config.get('scale_bar_mm', 2.0)
    exclusion_zones = config.get('exclusion_zones', [])
    
    # ===== STEP 1: Calibration =====
    pixels_per_mm = calibrate_scale(image, scale_roi, scale_mm)
    if pixels_per_mm is None:
        pixels_per_mm = 1.0
        unit = "pixels"
    else:
        unit = "mm"
    
    # ===== STEP 2: Create exclusion mask =====
    exclusion_mask = create_exclusion_mask(image.shape, exclusion_zones)
    
    # ===== STEP 3: ML Detection (if enabled) =====
    ml_centroids = []
    ml_count = 0
    print("\nRunning ML detection...")
    
    ml_api_key = config.get('roboflow_api_key', 'rhzcr5MSI8GCPCHm1OYv')
    ml_workspace = config.get('roboflow_workspace', 'payton-q3ubd')
    ml_workflow = config.get('roboflow_workflow', 'detect-count-and-visualize')
    
    ml_predictions = run_ml_detection(
        str(image_path),
        api_key=ml_api_key,
        workspace_name=ml_workspace,
        workflow_id=ml_workflow
    )
    
    if ml_predictions:
        ml_centroids = convert_predictions_to_centroids(ml_predictions)
        ml_count = len(ml_centroids)
        print(f"ML Detection: Detected {ml_count} dots")
    else:
        print(f"ML Detection: Failed or no objects detected")
    
    # ===== STEP 4: Compare all classical detection methods =====
    print("\nComparing classical detection methods...")
    method_results = compare_detection_methods(
        image,
        DetectionMethods.get_all_methods(),
        exclusion_mask,
        base_name,
        str(output_folder)
    )
    
    # ===== STEP 5: Optimized classical detection =====
    print("\nRunning optimized classical detection...")
    optimized_mask = DetectionMethods.optimized_combined(image)
    optimized_mask = apply_exclusion_mask(optimized_mask, exclusion_mask)
    optimized_centroids, optimized_contours = detect_contours(optimized_mask)
    
    classical_count = len(optimized_centroids)
    print(f"Classical Optimized: Detected {classical_count} dots")
    
    # ===== STEP 6: Distance analysis (Classical) =====
    distances_mm = []
    dot_details = []
    
    if len(optimized_centroids) >= 2:
        distances_mm = compute_nearest_neighbor_distances(optimized_centroids, pixels_per_mm)
        
        for i, ((cx, cy), dist_mm) in enumerate(zip(optimized_centroids, distances_mm)):
            dot_details.append({
                'dot_index': i,
                'centroid_x': cx,
                'centroid_y': cy,
                'nearest_distance': dist_mm
            })
    
    # ===== STEP 7: Distance analysis (ML) =====
    ml_distances_mm = []
    
    if len(ml_centroids) >= 2:
        ml_distances_mm = compute_nearest_neighbor_distances(ml_centroids, pixels_per_mm)
    
    # ===== STEP 8: Compile statistics =====
    stats = {
        'image_name': image_path.name,
        'processing_timestamp': datetime.now().isoformat(),
        
        # Classical detection results
        'classical_count': classical_count,
        'classical_avg_distance': np.mean(distances_mm) if distances_mm else None,
        'classical_min_distance': np.min(distances_mm) if distances_mm else None,
        'classical_max_distance': np.max(distances_mm) if distances_mm else None,
        'classical_std_distance': np.std(distances_mm) if distances_mm else None,
        
        # ML detection results
        'ml_count': ml_count,
        'ml_avg_distance': np.mean(ml_distances_mm) if ml_distances_mm else None,
        'ml_min_distance': np.min(ml_distances_mm) if ml_distances_mm else None,
        'ml_max_distance': np.max(ml_distances_mm) if ml_distances_mm else None,
        'ml_std_distance': np.std(ml_distances_mm) if ml_distances_mm else None,
        
        # Comparison
        'ml_vs_classical_diff': ml_count - classical_count if (ml_count and classical_count) else None,
        'ml_vs_classical_percent_diff': ((ml_count - classical_count) / classical_count * 100) if classical_count and ml_count else None,
        
        # Metadata
        'unit': unit,
        'pixels_per_mm': pixels_per_mm if unit == 'mm' else None,
        'excluded_zones_count': len(exclusion_zones),
    }
    
    # Add method comparison summary
    stats['method_comparison'] = json.dumps({
        name: data['count'] for name, data in method_results.items()
    })
    
    # ===== STEP 9: Visualizations =====
    print("\nGenerating visualizations...")
    
    # Distance histograms
    if distances_mm:
        plot_distance_histogram(distances_mm, f"{base_name}_classical", str(output_folder))
    if ml_distances_mm:
        plot_distance_histogram(ml_distances_mm, f"{base_name}_ml", str(output_folder))
    
    # Heatmaps
    plot_heatmap(optimized_centroids, image.shape, f"{base_name}_classical", str(output_folder))
    if ml_centroids:
        plot_heatmap(ml_centroids, image.shape, f"{base_name}_ml", str(output_folder))
    
    # Detection overlays
    classical_stats = {
        'dot_count': classical_count,
        'avg_distance': stats['classical_avg_distance'],
        'min_distance': stats['classical_min_distance'],
        'max_distance': stats['classical_max_distance'],
        'std_distance': stats['classical_std_distance'],
        'unit': unit,
    }
    visualize_detection_result(
        image, optimized_contours, optimized_centroids, classical_stats,
        f"{base_name}_classical", exclusion_zones, str(output_folder)
    )
    
    if ml_centroids:
        ml_contours, _ = detect_contours(np.zeros_like(image[:, :, 0]))  # Dummy for visualization
        ml_stats = {
            'dot_count': ml_count,
            'avg_distance': stats['ml_avg_distance'],
            'min_distance': stats['ml_min_distance'],
            'max_distance': stats['ml_max_distance'],
            'std_distance': stats['ml_std_distance'],
            'unit': unit,
        }
        # Create ML visualization
        ml_viz = image.copy()
        for cx, cy in ml_centroids:
            cv2.circle(ml_viz, (cx, cy), 5, (255, 0, 255), 2)  # Magenta for ML
        
        # Save ML detection overlay
        ml_output_path = Path(output_folder) / f"{base_name}_ml_detected.png"
        cv2.imwrite(str(ml_output_path), ml_viz)
        print(f"ML detection overlay saved: {ml_output_path}")
    
    # ===== STEP 10: Print summary =====
    print(f"\n✓ Completed: {image_path.name}")
    print(f"  Classical dots: {classical_count}")
    if stats['classical_avg_distance']:
        print(f"    Avg distance: {stats['classical_avg_distance']:.3f} {unit}")
    print(f"  ML dots: {ml_count}")
    if stats['ml_avg_distance']:
        print(f"    Avg distance: {stats['ml_avg_distance']:.3f} {unit}")
    if stats['ml_vs_classical_percent_diff']:
        print(f"  Difference: {stats['ml_vs_classical_percent_diff']:+.1f}%")
    
    return stats, dot_details


def batch_process(image_folder, output_folder="outputs", config=None):
    """
    Process all images in a folder.
    
    Args:
        image_folder: Folder containing images
        output_folder: Where to save results
        config: Configuration dict
        
    Returns:
        list: Results dicts for each image
    """
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    
    if not image_folder.exists():
        print(f"Error: Folder not found: {image_folder}")
        return []
    
    # Find all images
    patterns = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp')
    image_files = sorted([f for p in patterns for f in image_folder.glob(p)])
    
    if not image_files:
        print(f"No images found in {image_folder}")
        return []
    
    print(f"\nFound {len(image_files)} image(s)")
    print("="*60)
    
    all_results = []
    
    for i, image_path in enumerate(image_files, start=1):
        print(f"\n[{i}/{len(image_files)}]", end=" ")
        result = process_image(str(image_path), str(output_folder), config)
        
        if result:
            stats, dot_details = result
            all_results.append(stats)
    
    return all_results
