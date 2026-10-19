# Gender Classification from Facial Images Using PCA, LDA and k-NN

A biometric system that classifies facial images as **male** or **female** using classical pattern recognition techniques, implemented from scratch in **MATLAB**: Principal Component Analysis (PCA) and Linear Discriminant Analysis (LDA) for feature extraction, and a custom **k-Nearest Neighbors** classifier with cosine similarity.

Built as a project for **CSCI 158 (Spring 2025)**.



## Why Classical Methods?

Deep learning models like CNNs reach high accuracy but need large datasets and heavy computation. This project explores how far lightweight, interpretable techniques can go for gender classification, which suits small datasets, low-power devices and situations where you need to explain how a decision was made.



## How It Works

```
Face image ─► Grayscale ─► Resize 128×128 ─► CLAHE contrast ─► Flatten
          ─► PCA (dimensionality reduction) ─► [optional] LDA (class separation)
          ─► k-NN with cosine similarity ─► Male / Female
```

1. **Preprocessing:** each image is converted to grayscale, resized to 128 × 128 pixels and enhanced with Contrast-Limited Adaptive Histogram Equalization (CLAHE), then flattened into a vector.
2. **PCA:** finds the directions of greatest variance ("eigenfaces") and projects images onto a lower-dimensional space.
3. **LDA:** applied after PCA to find the projection that best separates the two classes.
4. **k-NN classifier:** a custom implementation that uses **cosine distance** instead of Euclidean distance, which works better for high-dimensional image features.
5. **Evaluation:** 80/20 train-test split, accuracy, confusion matrices and ROC curves with AUC.

Two pipelines are compared:
- **PCA + k-NN**
- **PCA → LDA + k-NN**



## Results

| Method | 100 images<br>(50 F / 50 M) | 200 images<br>(100 F / 100 M) | 1000 images<br>(500 F / 500 M) |
|---|---:|---:|---:|
| PCA + k-NN | **95.0%** | 82.5% | 81.5% |
| PCA → LDA + k-NN | 90.0% | **92.5%** | 81.5% |

**Settings used**

| Dataset size | PCA components (PCA only) | PCA components before LDA | k |
|---|---:|---:|---:|
| 100 images | 100 | 30 | 3 |
| 200 images | 75 | 15 | 3 |
| 1000 images | 80 | 30 | 5 |

**Observations**
- LDA improved class separation and generally gave higher AUC on the ROC curves.
- Cosine similarity outperformed Euclidean distance in the high-dimensional feature space.
- Confusion matrices showed balanced performance across both classes, with no strong bias toward either gender.
- Accuracy depends heavily on the number of PCA components and the value of k, so these were tuned for each dataset size.



## Repository Contents

| File | Description |
|---|---|
| `Code.m` | MATLAB implementation: preprocessing, PCA, LDA, k-NN, evaluation and ROC curves |
| `report (1).pdf` | Project report with methodology, confusion matrices and ROC curves |
| `Literature Review.pdf` | Review of classical and deep learning approaches to gender classification |
| `Facial Recognition - CSCI 158 .pptx` | Presentation slides |
| `Datasets/` | Face image datasets (100, 200 and large set), each split into `male/` and `female/` folders |



## How to Run

**Requirements:** MATLAB with the **Image Processing Toolbox** and **Statistics and Machine Learning Toolbox** (for `adapthisteq`, `confusionmat` and `perfcurve`).

1. Download or clone this repository.
2. Open `Code.m` and update these two lines to point to the dataset on your computer:
   ```matlab
   dataset_zip  = "path/to/bigdataset_158.zip";
   unzip_folder = "path/to/bigdataset_158";
   ```
3. Optionally adjust the settings at the top of the file (`k_pca_main`, `k_pca_lda`, `k`, `num_male`, `num_female`) to match the table above.
4. Run the script. It prints the accuracy and confusion matrix for both pipelines and plots the ROC curves.



## Future Work

- Compare against texture features such as LBP and LPQ
- Test on larger, more diverse benchmark datasets
- Improve robustness to lighting, pose and occlusion
- Compare with a lightweight CNN baseline
