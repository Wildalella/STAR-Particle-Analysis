"""
STAR Particle Analysis - Web Interface v2 (CLEAN)
Folder-based batch processing with result images displayed
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image
from glob import glob

# Import pipeline
from pipeline import batch_process

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="STAR Particle Analysis",
    page_icon="",
    layout="wide"
)

st.markdown("""
<style>
    .header-container {
        background: linear-gradient(135deg, #1f77b4 0%, #0055a4 100%);
        padding: 2rem;
        color: white;
        border-radius: 0;
    }
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_folder' not in st.session_state:
    st.session_state.uploaded_folder = None

ANALYSIS_CONFIG = {
    'scale_bar_roi_coords': (315, 330, 945, 1150),
    'scale_bar_mm': 2.0,
    'exclusion_zones': [
        (250, 330, 950, 1150),
        (0, 100, 0, 100),
    ],
    'roboflow_api_key': 'rhzcr5MSI8GCPCHm1OYv',
    'roboflow_workspace': 'payton-q3ubd',
    'roboflow_workflow': 'detect-count-and-visualize',
}

# ============================================================================
# HEADER
# ============================================================================

st.markdown("""
<div class="header-container">
    <h1 class="header-title"> STAR Particle Analysis</h1>
    <p style="color: white; margin: 0;">Batch Processing with Result Images</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("Upload Folder")
    
    folder_path = st.text_input(
        "Enter folder path",
        placeholder="C:/path/to/images"
    )
    
    if folder_path and Path(folder_path).exists():
        st.success("✓ Folder found!")
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
            image_files.extend(Path(folder_path).glob(ext))
        
        if image_files:
            st.info(f"Found {len(image_files)} images")
            st.session_state.uploaded_folder = folder_path
        else:
            st.warning("No images in folder")
    elif folder_path:
        st.error("Folder not found")
    else:
        st.info("Enter folder path")
    
    st.divider()
    
    st.subheader(" Settings")
    scale_mm = st.number_input("Scale bar (mm)", value=2.0, min_value=0.1)
    ANALYSIS_CONFIG['scale_bar_mm'] = scale_mm
    
    enable_exclusion = st.checkbox("Enable exclusion zones", value=True)
    if not enable_exclusion:
        ANALYSIS_CONFIG['exclusion_zones'] = []

# ============================================================================
# MAIN AREA
# ============================================================================

if st.session_state.uploaded_folder is None:
    st.subheader("Getting Started")
    st.markdown("""
    **Steps:**
    1. Enter folder path in sidebar
    2. Click "Run Analysis"
    3. View results in tabs
    """)

else:
    st.subheader("Analysis Results")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary",
        "Overlays",
        "Statistics",
        "Heatmaps",
        "Export"
    ])
    
    # ===== TAB 1: SUMMARY =====
    with tab1:
        if st.session_state.results is None:
            if st.button("▶️ Run Analysis", use_container_width=True):
                with st.spinner("Processing... (1-2 min per image)"):
                    try:
                        results = batch_process(
                            str(st.session_state.uploaded_folder),
                            output_folder="streamlit_outputs",
                            config=ANALYSIS_CONFIG
                        )
                        
                        if results:
                            st.session_state.results = results
                            st.success("✓ Complete!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Images", len(results))
            col2.metric("Total Classical", int(df['classical_count'].sum()))
            col3.metric("Total ML", int(df['ml_count'].sum()))
            col4.metric("Avg Diff %", f"{df['ml_vs_classical_percent_diff'].mean():+.1f}%")
            
            st.divider()
            st.markdown("### Results Table")
            
            display_df = df[['image_name', 'classical_count', 'ml_count', 
                            'ml_vs_classical_percent_diff']].copy()
            display_df.columns = ['Image', 'Classical', 'ML', 'Diff %']
            
            st.dataframe(display_df)
    
    # ===== TAB 2: OVERLAYS =====
    with tab2:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Detection Overlays")
            
            results = st.session_state.results
            output_folder = Path("streamlit_outputs")
            
            image_names = [r['image_name'] for r in results]
            selected_image = st.selectbox("Select image", image_names)
            
            base_name = Path(selected_image).stem
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical**")
                overlay_path = output_folder / f"{base_name}_classical_detected.png"
                if overlay_path.exists():
                    st.image(str(overlay_path))
                else:
                    st.info("Not found")
            
            with col2:
                st.markdown("**ML**")
                overlay_path = output_folder / f"{base_name}_ml_detected.png"
                if overlay_path.exists():
                    st.image(str(overlay_path))
                else:
                    st.info("Not found")
            
            st.divider()
            st.markdown("### All Overlays")
            
            png_files = sorted(output_folder.glob("*_detected.png"))
            
            if png_files:
                cols = st.columns(2)
                for idx, img_file in enumerate(png_files):
                    with cols[idx % 2]:
                        try:
                            img = Image.open(img_file)
                            st.image(img, caption=img_file.name)
                        except:
                            st.error(f"Could not load {img_file.name}")
    
    # ===== TAB 3: STATISTICS =====
    with tab3:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical**")
                st.metric("Mean Dist", f"{df['classical_avg_distance'].mean():.4f} mm")
                st.metric("Std Dev", f"{df['classical_std_distance'].mean():.4f} mm")
            
            with col2:
                st.markdown("**ML**")
                st.metric("Mean Dist", f"{df['ml_avg_distance'].mean():.4f} mm")
                st.metric("Std Dev", f"{df['ml_std_distance'].mean():.4f} mm")
            
            st.divider()
            st.markdown("### Distribution")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=df['classical_avg_distance'].dropna(),
                name='Classical',
                marker_color='#1f77b4',
                opacity=0.7
            ))
            fig.add_trace(go.Histogram(
                x=df['ml_avg_distance'].dropna(),
                name='ML',
                marker_color='#ff7f0e',
                opacity=0.7
            ))
            fig.update_layout(barmode='overlay', height=400)
            st.plotly_chart(fig)
    
    # ===== TAB 4: HEATMAPS =====
    with tab4:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Heatmaps")
            
            results = st.session_state.results
            output_folder = Path("streamlit_outputs")
            
            image_names = [r['image_name'] for r in results]
            selected_image = st.selectbox("Select image", image_names, key="heatmap")
            
            base_name = Path(selected_image).stem
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical**")
                heatmap_path = output_folder / f"{base_name}_classical_heatmap.png"
                if heatmap_path.exists():
                    st.image(str(heatmap_path))
                else:
                    st.info("Not found")
            
            with col2:
                st.markdown("**ML**")
                heatmap_path = output_folder / f"{base_name}_ml_heatmap.png"
                if heatmap_path.exists():
                    st.image(str(heatmap_path))
                else:
                    st.info("Not found")
            
            st.divider()
            st.markdown("### All Heatmaps")
            
            heatmap_files = sorted(output_folder.glob("*_heatmap.png"))
            
            if heatmap_files:
                cols = st.columns(2)
                for idx, img_file in enumerate(heatmap_files):
                    with cols[idx % 2]:
                        try:
                            img = Image.open(img_file)
                            st.image(img, caption=img_file.name)
                        except:
                            st.error(f"Could not load {img_file.name}")
    
    # ===== TAB 5: EXPORT =====
    with tab5:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Download Results")
            
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            export_df = df[['image_name', 'classical_count', 'ml_count',
                           'ml_vs_classical_percent_diff', 'classical_avg_distance',
                           'ml_avg_distance']].copy()
            export_df.columns = ['Image', 'Classical', 'ML', 'Diff %', 'Class Mean', 'ML Mean']
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            
            st.divider()
            st.markdown("### Summary")
            st.dataframe(export_df)

st.divider()
st.markdown("**STAR Particle Analysis v2.0**")
