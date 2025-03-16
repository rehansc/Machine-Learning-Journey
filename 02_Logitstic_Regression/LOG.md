# Overview of the Breast Cancer Dataset

The Breast Cancer Wisconsin (Diagnostic) Dataset is a well-known dataset used for binary classification tasks. It helps in predicting whether a tumor is benign (non-cancerous) or malignant (cancerous) based on 30 numerical features extracted from cell nuclei in digitized images.

# Key Details of the Dataset

Source: UCI Machine Learning Repository
Number of Samples: 569
Number of Features: 30
Target Classes:
0 → Malignant (Cancerous)
1 → Benign (Non-Cancerous)

# Features in the Dataset
Each tumor is described using 10 real-valued features, calculated for:

Mean value
Standard error
Worst (largest) value
So, there are 30 features in total (10 × 3).

# Main Feature Groups
Radius – Mean of distances from center to perimeter
Texture – Variation in gray levels
Perimeter – Length of tumor boundary
Area – Size of the tumor
Smoothness – Local variation in radius lengths
Compactness – (Perimeter² / Area - 1.0)
Concavity – Severity of indentations in the tumor
Concave Points – Number of concave portions of the contour
Symmetry – How symmetrical the tumor is
Fractal Dimension – Measurement of tumor complexity
Class Distribution
Benign (1) → 357 samples (63%)
Malignant (0) → 212 samples (37%)
Imbalance? → Yes, more benign cases than malignant.
