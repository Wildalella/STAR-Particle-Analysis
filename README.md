# STAR Particle Analysis Pipeline

## Automated Computer Vision and Machine Learning Framework for Quantifying STAR Microparticles

---

## Overview

The **STAR Particle Analysis Pipeline** is an automated computer vision and machine learning framework for detecting, quantifying, and analyzing **Skin Targeted Active Release (STAR)** microparticles in microscopy images.

STAR (Skin Targeted Active Release) particles are microscopic, star-shaped structures designed to create temporary micropores in the skin, allowing topical drugs and vaccines to penetrate more effectively. After treatment, the skin is stained with **gentian violet**, which highlights the micropores created by the STAR particles.

This project automates the analysis of these microscopy images using both **classical computer vision** and **machine learning (YOLO)** techniques. Instead of manually counting thousands of micropores, the pipeline automatically detects features, measures their spatial distribution, compares multiple detection methods, and generates publication-ready visualizations and reports.

The software also provides an interactive **Streamlit** web application for batch image analysis, visualization, and result comparison.

---

## Features

| Feature | Description |
|----------|-------------|
| Classical Computer Vision | Detects STAR micropores using multiple image processing algorithms. |
| Six Detection Methods | Expanded HSV, Multiple HSV Ranges, Saturation-Based, Adaptive Threshold, Statistical Outlier, and LAB Color Space detection. |
| Hybrid Detection Pipeline | Combines multiple classical approaches for improved detection accuracy. |
| Machine Learning Detection | Uses a custom YOLO model trained with Roboflow for automated object detection. |
| Method Comparison | Compares classical computer vision results with machine learning predictions. |
| Scale Calibration | Converts pixel measurements into real-world millimeter measurements using the image scale bar. |
| Contour Detection | Identifies particle boundaries and computes centroid locations. |
| Distance Analysis | Calculates nearest-neighbor distances between detected micropores using KDTree. |
| Statistical Analysis | Computes count, minimum, maximum, average, and standard deviation of particle spacing. |
| Heatmaps | Generates spatial density heatmaps of detected micropores. |
| Histograms | Creates nearest-neighbor distance distribution plots. |
| Detection Overlays | Produces annotated images showing detected micropores and analysis results. |
| Batch Processing | Processes entire folders of microscopy images automatically. |
| Streamlit Dashboard | Interactive web interface for uploading images, viewing overlays, comparing methods, and exporting results. |
| Excel Export | Automatically generates Excel reports with summary statistics and comparisons. |
| CSV Export | Exports processed results for further statistical analysis. |
| SQLite Integration | Stores processed image data in a local database. |
| Google Sheets Integration | Uploads analysis results for cloud-based collaboration. |
| Error Handling | Handles missing images, failed detections, and processing errors gracefully. |

---

## Workflow

```
Microscopy Images
        │
        ▼
 Image Preprocessing
        │
        ├── Scale Calibration
        ├── Exclusion Zones
        └── Image Cleaning
        │
        ▼
 ┌──────────────────────────────┐
 │ Classical Computer Vision    │
 │                              │
 │ • Expanded HSV               │
 │ • Multiple HSV Ranges        │
 │ • Saturation-Based           │
 │ • Adaptive Threshold         │
 │ • Statistical Outlier        │
 │ • LAB Color Space            │
 └──────────────────────────────┘
        │
        ▼
 Optimized Hybrid Detection
        │
        ▼
 Contour Detection & Centroids
        │
        ├──────────────┐
        │              │
        ▼              ▼
 Distance Analysis   YOLO Detection
        │              │
        └──────┬───────┘
               ▼
     Classical vs ML Comparison
               │
               ▼
 Visualization & Reporting
               │
        ├── Detection Overlays
        ├── Heatmaps
        ├── Histograms
        ├── Excel Reports
        ├── CSV Files
        ├── SQLite Database
        └── Google Sheets
```

---

## Outputs

For every processed image, the pipeline automatically generates:

- Detection overlay images
- Classical detection statistics
- Machine learning detection statistics
- Classical vs. ML comparison
- Distance distribution histograms
- Spatial density heatmaps
- Excel summary reports
- CSV exports
- SQLite database records
- Google Sheets reports

---

## Technologies Used

- Python
- OpenCV
- NumPy
- SciPy
- Matplotlib
- Pandas
- Plotly
- Streamlit
- Roboflow
- YOLO
- SQLite
- OpenPyXL

---

## Applications

This project can be used for:

- Biomedical image analysis
- Drug delivery research
- Microscopy image processing
- Computer vision research
- Machine learning model evaluation
- Automated laboratory workflows
- Spatial distribution analysis
- Research data visualization