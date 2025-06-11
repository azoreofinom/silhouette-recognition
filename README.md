#  Body and Head Shape-Based Biometric Recognition System

A biometric recognition and verification system that relies solely on **body and head shape**, without using facial recognition. This approach is designed for scenarios where the subject is not facing the camera or where face recognition is not feasible.

---

##  Summary

This system:
- Recognizes and verifies individuals using silhouette and shape features.
- Avoids facial recognition to enhance privacy and utility in non-frontal views.
- Performs well despite limited training data.
- Highlights both the potential and the limitations of shape-based biometrics.

---

##  Methodology

###  Silhouette Extraction
- Crops input images to a green background region.
- Uses background subtraction with keypoint alignment (ORB + RANSAC).
- Converts aligned image to binary silhouette mask via thresholding.

###  Feature Extraction

#### Baseline (Full Body)
- **Hu Moments**
- **Fourier Descriptors** (via PyEFD)
- **Pose Estimation** using MediaPipe (torso, leg lengths)
- **Silhouette Height** from bounding box

#### Improved (Head Shape)
- Extracts head from silhouette using a fixed-size bounding box
- Features include:
  - Hu Moments
  - Fourier Descriptors
  - Head height, width, area, and perimeter

###  Classification
- Features reduced using **PCA**
- Classification via **SVM** (RBF kernel, grid-searched hyperparameters)

---

##  Results

| Task                | Metric                     | Score       |
|---------------------|----------------------------|-------------|
| **Recognition**     | Rank-1 Accuracy (CCR)      | 68%         |
|                     | Precision                  | 0.73        |
|                     | Recall                     | 0.68        |
|                     | F1 Score                   | 0.68        |
| **Verification**    | Accuracy at Equal Error Rate | 91%       |

- Head-based features outperform full-body baseline, especially when clothing varies.
- Verification is more reliable than recognition due to being a binary task.

---

##  Limitations

- Sensitive to:
  - Clothing and posture variations
  - Viewpoint changes
- Head detection uses a fixed box size, not generalizable
- Small dataset limits generalization

---

##  Future Improvements

- Replace head detection with **YOLOv8** or another object detection network.
- Improve pose estimation to be more robust to viewpoint changes.
- Extend the system for use with **outdoor or unconstrained environments**.

---

##  Dataset

- Based on a subset of the **Large Southampton Gait Database**
- Dataset not included in this repository

---


