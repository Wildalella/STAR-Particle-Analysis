---

# Current Limitations & Future Development

The STAR Particle Analysis Pipeline is under active development. The following features are planned for future releases:

## Docker Support

- Containerize the complete application using **Docker** to simplify installation and ensure reproducible execution across Windows, macOS, and Linux.
- Bundle all required dependencies, including Python packages, OpenCV, Streamlit, and the inference SDK.
- Provide Docker Compose support for one-command deployment.

**Goal:**
- `docker build`
- `docker run`
- Launch the complete Streamlit application without manual environment setup.

---

## Automatic Annotation Removal

Current images contain non-biological artifacts such as:

- Scale bars
- Measurement labels (e.g., "2 mm")
- Figure labels
- Text annotations
- Symbols
- Image borders

Although exclusion zones are currently used to ignore these regions, future versions will automatically detect and remove these artifacts before analysis.

Planned improvements include:

- Automatic scale bar detection
- OCR-based text detection and masking
- Figure annotation removal
- Adaptive region masking
- Automatic exclusion zone generation

This will allow the pipeline to process images with minimal user configuration.

---

## Planned Improvements

- Improve YOLO detection accuracy using larger annotated datasets.
- Support additional microscopy image formats.
- Add confidence score visualization for machine learning predictions.
- Generate PDF summary reports.
- Export results to additional database formats.
- Enable parallel batch processing for large datasets.
- Add configurable analysis parameters through the Streamlit interface.
- Package the project as an installable Python package.

---