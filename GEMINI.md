# Defacer: Medical Image De-identification Project

## Project Overview
This project provides a tool for de-identifying (defacing) identifiable facial features (eyes, nose, ears, and mouth) in medical images (CT and MRI scans). It supports both **DICOM** and **NIfTI** formats.

The core algorithm uses an **Attention-gated 3D U-Net** and **SENet** to detect facial features and apply local masking/blurring, minimizing information loss in regions of interest like the brain while protecting personal health information (PHI).

### Main Technologies
- **Python:** 3.7
- **Deep Learning:** TensorFlow 1.14.0, Keras 2.2.4
- **Imaging Libraries:** `pydicom`, `nibabel`, `scipy`, `skimage`
- **Visualization:** `matplotlib`

## Building and Running

### Prerequisites
1.  **Environment:** Ensure you have Python 3.7 installed.
2.  **Dependencies:** Install the required libraries:
    ```bash
    # Note: TensorFlow 1.x is required as per current code (tf.get_default_graph())
    pip install tensorflow==1.14.0 Keras==2.2.4 pydicom nibabel scipy scikit-image matplotlib
    ```

### Execution
The project provides separate scripts for processing DICOM and NIfTI data:

#### DICOM Anonymization
1.  Place your DICOM files in a directory (e.g., `test_exam`).
2.  Configure paths in `run_anonymization.py`.
3.  Run:
    ```bash
    python run_anonymization.py
    ```

#### NIfTI Anonymization
1.  Configure the `nifti_folder` in `run_nifti.py`.
2.  Run:
    ```bash
    python run_nifti.py
    ```

### Output and Verification
-   **Anonymized Files:** Stored in `anonymized_output2/` or `anonymized_nifti_output/`.
-   **Visual Verification:** PNG plots of the predicted masks are saved in `verification_images/` and `verification_images_nii/`.

## Development Conventions

### Architecture
-   **`Defacer` Class:** Located in `model_distribution/defacer.py` (or the root `defacer.py`), it contains the main logic for loading scans, predicting masks, and applying blurring.
-   **Model Distribution:** Weights (`.h5`) and model definitions are found in `model_distribution/model/`.
-   **Anonymization Mask:** Controlled by a tuple/list `(eyes, nose, ears, mouth)` where `1` indicates removal and `0` indicates keeping the feature.

### Adding New Features
-   New model architectures should be placed in `model_distribution/model/`.
-   Training scripts and research notebooks are located in `model_training/`.

### Metadata De-identification
The `Defacer` class includes a `header_deidentification` method for DICOM files, which removes sensitive fields such as Patient Name, ID, Birth Date, etc., according to HIPAA/GDPR standards.
