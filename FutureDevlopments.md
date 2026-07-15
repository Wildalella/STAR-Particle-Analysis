## Limitations and Future Development

The STAR Particle Analysis Pipeline is actively being improved. Future updates will focus on increasing automation, portability, and detection accuracy.

### Planned Improvements

- **Docker Containerization**
  - Package the complete pipeline into a Docker container for reproducible deployment across different systems.
  - Simplify installation by including all required dependencies, models, and application requirements.

- **Automatic Artifact Removal**
  - Improve preprocessing to automatically detect and remove non-biological features such as:
    - Scale bars
    - Measurement labels
    - Figure annotations
    - Text and symbols
  - Replace manually defined exclusion zones with automated masking methods using OCR and computer vision.

- **Machine Learning Improvements**
  - Expand training datasets to improve YOLO detection accuracy.
  - Add confidence score visualization and improved model evaluation metrics.

- **Software Improvements**
  - Add automated report generation.
  - Improve batch processing performance.
  - Support additional microscopy image formats.
  - Continue improving the Streamlit interface for easier analysis.

The goal is to create a fully automated, reproducible, and user-friendly platform for microscopy image analysis in biomedical research.
