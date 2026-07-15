"""
STAR Particle Analysis - Main Entry Point (FIXED VERSION)
Simplified orchestration with better output feedback
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
from datetime import datetime

# Show where we're running from
print(f"\n{'='*70}")
print(f"Running from: {os.getcwd()}")
print(f"{'='*70}\n")

from pipeline import batch_process

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'image_folder': r"C:\Users\wildabrick\Documents\STAR-Project\images",
    'output_folder': "outputs",  # Creates in current directory
    
    # Calibration
    'scale_bar_roi_coords': (315, 330, 945, 1150),
    'scale_bar_mm': 2.0,
    
    # Exclusion zones: (y_start, y_end, x_start, x_end)
    'exclusion_zones': [
        (250, 330, 950, 1150),  # 2 mm text area
        (0, 100, 0, 100),       # f symbol area
    ],
    
    # Roboflow ML Configuration
    'roboflow_api_key': 'rhzcr5MSI8GCPCHm1OYv',
    'roboflow_workspace': 'payton-q3ubd',
    'roboflow_workflow': 'detect-count-and-visualize',
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main pipeline execution"""
    print("="*70)
    print("STAR PARTICLE ANALYSIS - SIMPLIFIED PIPELINE")
    print("="*70)
    
    # Validate config
    image_folder = Path(CONFIG['image_folder'])
    output_folder = Path(CONFIG['output_folder'])
    
    print(f"\n📁 Configuration:")
    print(f"   Image folder: {image_folder}")
    print(f"   Output folder: {output_folder}")
    print(f"   Image folder exists: {image_folder.exists()}")
    
    if not image_folder.exists():
        print(f"\n✗ ERROR: Image folder not found!")
        print(f"   Expected: {image_folder}")
        return
    
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"   Output folder ready: {output_folder.absolute()}")
    
    # ===== RUN PIPELINE =====
    print(f"\n{'='*70}")
    print("PROCESSING IMAGES")
    print(f"{'='*70}\n")
    
    results = batch_process(
        str(image_folder),
        str(output_folder),
        config={
            'scale_bar_roi_coords': CONFIG['scale_bar_roi_coords'],
            'scale_bar_mm': CONFIG['scale_bar_mm'],
            'exclusion_zones': CONFIG['exclusion_zones'],
            'roboflow_api_key': CONFIG['roboflow_api_key'],
            'roboflow_workspace': CONFIG['roboflow_workspace'],
            'roboflow_workflow': CONFIG['roboflow_workflow'],
        }
    )
    
    if not results:
        print("\n✗ No images were processed.")
        return
    
    # ===== EXPORT RESULTS =====
    print("\n" + "="*70)
    print("EXPORTING RESULTS")
    print("="*70)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save Excel with multiple sheets
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_file = output_folder / f"detection_results_{timestamp}.xlsx"
    
    print(f"\n📊 Creating Excel file...")
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Main results - keep all columns
            df_display = df.copy()
            df_display.to_excel(writer, sheet_name='Results', index=False)
            print(f"   ✓ Results sheet (with {len(df_display)} rows)")
            print(f"     Columns: Classical & ML detection results")
            
            # Summary statistics
            summary = pd.DataFrame([{
                'Total Images': len(results),
                'Total Classical Dots': int(df['classical_count'].sum()),
                'Total ML Dots': int(df['ml_count'].sum()),
                'Avg Classical Dots/Image': float(df['classical_count'].mean()),
                'Avg ML Dots/Image': float(df['ml_count'].mean()),
                'Classical Avg Distance (mm)': float(df['classical_avg_distance'].mean()) if df['classical_avg_distance'].notna().any() else None,
                'ML Avg Distance (mm)': float(df['ml_avg_distance'].mean()) if df['ml_avg_distance'].notna().any() else None,
            }])
            summary.to_excel(writer, sheet_name='Summary', index=False)
            print(f"   ✓ Summary sheet")
            
            # Comparison sheet
            comparison = df[['image_name', 'classical_count', 'ml_count', 'ml_vs_classical_diff', 'ml_vs_classical_percent_diff']].copy()
            comparison.columns = ['Image', 'Classical Count', 'ML Count', 'Difference', 'Percent Difference (%)']
            comparison.to_excel(writer, sheet_name='ML vs Classical', index=False)
            print(f"   ✓ ML vs Classical comparison sheet")
        
        print(f"\n✓ Excel file saved:")
        print(f"   {excel_file.absolute()}")
    
    except Exception as e:
        print(f"\n✗ ERROR saving Excel file: {e}")
        print(f"   Make sure openpyxl is installed: pip install openpyxl")
        return
    
    # ===== PRINT SUMMARY =====
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Total images processed: {len(results)}")
    print(f"\n📊 CLASSICAL DETECTION:")
    print(f"   Total dots: {int(df['classical_count'].sum())}")
    print(f"   Average dots per image: {df['classical_count'].mean():.1f}")
    if df['classical_avg_distance'].notna().any():
        print(f"   Average distance: {df['classical_avg_distance'].mean():.3f} mm")
    
    print(f"\n🤖 ML DETECTION:")
    print(f"   Total dots: {int(df['ml_count'].sum())}")
    print(f"   Average dots per image: {df['ml_count'].mean():.1f}")
    if df['ml_avg_distance'].notna().any():
        print(f"   Average distance: {df['ml_avg_distance'].mean():.3f} mm")
    
    print(f"\n📈 COMPARISON:")
    avg_diff = df['ml_vs_classical_diff'].mean()
    avg_percent_diff = df['ml_vs_classical_percent_diff'].mean()
    print(f"   Average difference: {avg_diff:+.1f} dots ({avg_percent_diff:+.1f}%)")
    print(f"   Classical detects more: {(df['classical_count'] > df['ml_count']).sum()} images")
    print(f"   ML detects more: {(df['ml_count'] > df['classical_count']).sum()} images")
    print(f"   Equal detection: {(df['ml_count'] == df['classical_count']).sum()} images")
    
    print(f"\n📁 Output folder: {output_folder.absolute()}")
    print(f"\n   Generated files:")
    
    # List output files
    output_files = sorted(output_folder.glob("*"))
    for f in output_files:
        if f.is_file():
            size_mb = f.stat().st_size / (1024*1024)
            if size_mb > 0.01:
                print(f"   ✓ {f.name} ({size_mb:.2f} MB)")
            else:
                print(f"   ✓ {f.name}")
    
    print("\n" + "="*70)
    print("✓ PROCESSING COMPLETE!")
    print("="*70)
    print(f"\nCheck this folder for all results:")
    print(f"  {output_folder.absolute()}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
