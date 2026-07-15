"""
STAR Particle Analysis - Web Interface with Automatic Confidence Scores
Folder-based batch processing with confidence scoring and result images displayed
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
from PIL import Image

# Import pipeline
from pipeline import batch_process

# ============================================================================
# CONFIDENCE SCORE CALCULATOR
# ============================================================================

class ConfidenceScoreCalculator:
    """Calculate confidence scores automatically"""
    
    @staticmethod
    def calculate(classical_count, ml_count):
        """
        Calculate confidence score based on agreement between methods.
        
        Returns: 0-100 confidence score
        """
        if classical_count == 0 and ml_count == 0:
            return 0
        
        if classical_count == 0 or ml_count == 0:
            return 20
        
        # Calculate percent difference
        diff = abs(classical_count - ml_count)
        avg = (classical_count + ml_count) / 2
        percent_diff = (diff / avg) * 100
        
        # Convert to confidence (0% diff = 100% conf, 100% diff = 0% conf)
        confidence = max(0, 100 - percent_diff)
        
        return round(confidence, 1)
    
    @staticmethod
    def get_grade(confidence):
        """Convert confidence score to letter grade"""
        if confidence >= 90:
            return "A+ (Excellent)", "🟢"
        elif confidence >= 85:
            return "A (Very Good)", "🟢"
        elif confidence >= 75:
            return "B (Good)", "🟡"
        elif confidence >= 65:
            return "C (Moderate)", "🟡"
        elif confidence >= 50:
            return "D (Fair)", "🔴"
        else:
            return "F (Poor)", "🔴"
    
    @staticmethod
    def get_recommendation(confidence):
        """Get action recommendation based on confidence"""
        if confidence >= 85:
            return "✅ Use with high confidence"
        elif confidence >= 70:
            return "⚠️ Use with caution"
        elif confidence >= 50:
            return "⚠️ Verify manually recommended"
        else:
            return "❌ Must verify before use"

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
    .confidence-high {
        color: green;
        font-weight: bold;
    }
    .confidence-medium {
        color: orange;
        font-weight: bold;
    }
    .confidence-low {
        color: red;
        font-weight: bold;
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
    <h1 class="header-title">🔬 STAR Particle Analysis</h1>
    <p style="color: white; margin: 0;">Batch Processing with Confidence Scoring</p>
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
    
    st.subheader("Settings")
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
    3. View results with automatic confidence scores
    """)

else:
    st.subheader("Analysis Results with Confidence Scoring")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Summary",
        "Overlays",
        "Statistics",
        "Heatmaps",
        "Export"
    ])
    
    # ===== TAB 1: SUMMARY WITH CONFIDENCE =====
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
                            # Add confidence scores to results
                            for result in results:
                                conf = ConfidenceScoreCalculator.calculate(
                                    result['classical_count'],
                                    result['ml_count']
                                )
                                result['confidence_score'] = conf
                                grade, emoji = ConfidenceScoreCalculator.get_grade(conf)
                                result['confidence_grade'] = grade
                                result['confidence_emoji'] = emoji
                            
                            st.session_state.results = results
                            st.success("✓ Complete!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            # Overall metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            col1.metric("Images", len(results))
            col2.metric("Classical Total", int(df['classical_count'].sum()))
            col3.metric("ML Total", int(df['ml_count'].sum()))
            col4.metric("Avg Diff %", f"{df['ml_vs_classical_percent_diff'].mean():+.1f}%")
            col5.metric("Avg Confidence", f"{df['confidence_score'].mean():.1f}%")
            
            st.divider()
            
            # Confidence distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Confidence Score Distribution")
                conf_scores = df['confidence_score'].values
                high = sum(1 for c in conf_scores if c >= 85)
                medium = sum(1 for c in conf_scores if 70 <= c < 85)
                low = sum(1 for c in conf_scores if c < 70)
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=['High\n(85%+)', 'Medium\n(70-84%)', 'Low\n(<70%)'],
                        y=[high, medium, low],
                        marker_color=['green', 'orange', 'red']
                    )
                ])
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Confidence Score Details")
                st.metric("Average Confidence", f"{df['confidence_score'].mean():.1f}%")
                st.metric("Min Confidence", f"{df['confidence_score'].min():.1f}%")
                st.metric("Max Confidence", f"{df['confidence_score'].max():.1f}%")
                
                avg_conf = df['confidence_score'].mean()
                if avg_conf >= 85:
                    st.success("✅ Overall: Methods agree well!")
                elif avg_conf >= 70:
                    st.warning("⚠️ Overall: Moderate agreement")
                else:
                    st.error("❌ Overall: Methods disagree strongly")
            
            st.divider()
            st.markdown("### Per-Image Results with Confidence")
            
            # Create detailed results table
            display_df = df[['image_name', 'classical_count', 'ml_count', 
                            'ml_vs_classical_percent_diff', 'confidence_score',
                            'confidence_grade']].copy()
            display_df.columns = ['Image', 'Classical', 'ML', 'Diff %', 
                                  'Confidence %', 'Grade']
            
            # Format for display
            display_df['Confidence %'] = display_df['Confidence %'].apply(lambda x: f"{x:.1f}%")
            
            st.dataframe(display_df)
            
            # Highlight low confidence images
            low_conf = df[df['confidence_score'] < 70]
            if len(low_conf) > 0:
                st.divider()
                st.warning(f"⚠️ {len(low_conf)} image(s) with low confidence (<70%)")
                st.write("**Recommended actions for these images:**")
                st.write("1. Manually verify detections")
                st.write("2. Check if image has unusual characteristics")
                st.write("3. Consider which method is more accurate")
                
                low_conf_display = low_conf[['image_name', 'classical_count', 
                                             'ml_count', 'confidence_score']].copy()
                low_conf_display.columns = ['Image', 'Classical', 'ML', 'Confidence']
                st.dataframe(low_conf_display)
    
    # ===== TAB 2: OVERLAYS =====
    with tab2:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Detection Overlays with Confidence")
            
            results = st.session_state.results
            output_folder = Path("streamlit_outputs")
            
            # Create confidence-based selector
            image_options = [
                f"{r['image_name']} (Confidence: {r['confidence_score']:.1f}%)" 
                for r in results
            ]
            selected = st.selectbox("Select image", image_options)
            
            # Extract image name
            selected_image = selected.split(" (")[0]
            result = [r for r in results if r['image_name'] == selected_image][0]
            
            # Show confidence info
            col1, col2, col3 = st.columns(3)
            col1.metric("Confidence", f"{result['confidence_score']:.1f}%")
            col2.metric("Grade", result['confidence_grade'])
            col3.write(f"**{ConfidenceScoreCalculator.get_recommendation(result['confidence_score'])}**")
            
            st.divider()
            
            base_name = Path(selected_image).stem
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical Detection**")
                overlay_path = output_folder / f"{base_name}_classical_detected.png"
                if overlay_path.exists():
                    st.image(str(overlay_path))
            
            with col2:
                st.markdown("**ML Detection**")
                overlay_path = output_folder / f"{base_name}_ml_detected.png"
                if overlay_path.exists():
                    st.image(str(overlay_path))
            
            st.divider()
            st.markdown("### All Overlays Sorted by Confidence")
            
            # Sort by confidence
            sorted_results = sorted(results, key=lambda x: x['confidence_score'], reverse=True)
            png_files = []
            
            cols = st.columns(3)
            for idx, result in enumerate(sorted_results):
                with cols[idx % 3]:
                    st.markdown(f"**{result['image_name'][:30]}**")
                    st.caption(f"Confidence: {result['confidence_score']:.1f}%")
                    
                    base = Path(result['image_name']).stem
                    overlay = output_folder / f"{base}_classical_detected.png"
                    if overlay.exists():
                        st.image(str(overlay))
    
    # ===== TAB 3: STATISTICS =====
    with tab3:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical Method**")
                st.metric("Mean Count", f"{df['classical_count'].mean():.0f}")
                st.metric("Median Count", f"{df['classical_count'].median():.0f}")
                st.metric("Std Dev", f"{df['classical_count'].std():.1f}")
            
            with col2:
                st.markdown("**ML Method**")
                st.metric("Mean Count", f"{df['ml_count'].mean():.0f}")
                st.metric("Median Count", f"{df['ml_count'].median():.0f}")
                st.metric("Std Dev", f"{df['ml_count'].std():.1f}")
            
            st.divider()
            st.markdown("### Confidence vs Agreement")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['ml_vs_classical_percent_diff'],
                y=df['confidence_score'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=df['confidence_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Confidence %")
                ),
                text=df['image_name'],
                hovertemplate='<b>%{text}</b><br>Difference: %{x:.1f}%<br>Confidence: %{y:.1f}%'
            ))
            
            fig.update_layout(
                title="Confidence Score vs Method Disagreement",
                xaxis_title="% Difference (ML vs Classical)",
                yaxis_title="Confidence Score (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== TAB 4: HEATMAPS =====
    with tab4:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Heatmaps")
            
            results = st.session_state.results
            output_folder = Path("streamlit_outputs")
            
            image_options = [
                f"{r['image_name']} (Confidence: {r['confidence_score']:.1f}%)" 
                for r in results
            ]
            selected = st.selectbox("Select image", image_options, key="heatmap")
            
            selected_image = selected.split(" (")[0]
            base_name = Path(selected_image).stem
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Classical Heatmap**")
                heatmap_path = output_folder / f"{base_name}_classical_heatmap.png"
                if heatmap_path.exists():
                    st.image(str(heatmap_path))
            
            with col2:
                st.markdown("**ML Heatmap**")
                heatmap_path = output_folder / f"{base_name}_ml_heatmap.png"
                if heatmap_path.exists():
                    st.image(str(heatmap_path))
    
    # ===== TAB 5: EXPORT =====
    with tab5:
        if st.session_state.results is None:
            st.info("Run analysis first")
        else:
            st.markdown("### Download Results with Confidence Scores")
            
            results = st.session_state.results
            df = pd.DataFrame(results)
            
            export_df = df[['image_name', 'classical_count', 'ml_count',
                           'ml_vs_classical_percent_diff', 'confidence_score',
                           'confidence_grade']].copy()
            export_df.columns = ['Image', 'Classical', 'ML', 'Diff %', 
                                'Confidence %', 'Grade']
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            st.divider()
            st.markdown("### Results Summary")
            st.dataframe(export_df)

st.divider()
st.markdown("**STAR Particle Analysis v3.0 - With Automatic Confidence Scoring**")
