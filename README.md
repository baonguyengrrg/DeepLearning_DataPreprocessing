# Data Preprocessing for Deep Learning models
## Batch Normalization Demonstration

A practical PyTorch implementation to visualize how **Batch Normalization (BN)** stabilizes internal covariate shift in deep learning models.

### 📌 Project Overview
This project simulates a real-world financial dataset (Credit Scoring) to demonstrate the mathematical impact of BN. We process a mini-batch of 5 samples with 3 distinct features:
- **Feature 1:** Annual Income
- **Feature 2:** Credit History Length
- **Feature 3:** Debt Ratio

### 🚀 Installation & Usage

1. **Prerequisites**: Ensure you have Python installed. After that, pulling *Batchnorm.py*.
2. **Install PyTorch**:
   ```bash
   pip install torch
3. **Run the script**
   ```bash
   python Batchnorm.py

## Group Normalization Demonstration

A practical implementation to demonstrate how Group Normalization (GN) provides a robust alternative to Batch Normalization, especially when working with limited memory or micro-batch sizes.

### 📌 Project Overview
While Batch Normalization depends on the batch dimension, this project shows how GN processes each data instance independently by partitioning feature channels into groups. We use the same financial dataset structure to visualize the shift in the computation axis:

- **Independent Scaling**: Unlike BN, GN stability is entirely invariant to batch size.

- **Channel Grouping**: Features are divided into predefined groups to compute local mean and variance.

- **Dimension Shift**: The statistical moments are derived row-wise (dim=1), ensuring consistent normalization even for a single sample.

### 🚀 Installation & Usage

1. **Prerequisites**: Ensure you have Python installed. After that, pulling *Groupnorm.py*.
2. **Install PyTorch**:
   ```bash
   pip install torch
3. **Run the script**
   ```bash
   python Groupnorm.py

## Z-score Normalization (Standardization)
A NumPy-based implementation to demonstrate how Standardization transforms raw data into dimensionless scale with a mean of 0 and a standard deviation of 1.

### 📌 Project Overview
This script simulates a feature with varying magnitudes-specifically, the resistance load (kg) in a strength training regimen-to show how Z-score normalization neutralizes magnitude dominance.
- **Input Data**: A sequence of loads $[70, 80, 90, 100, 110]$.
- **The Baseline ($Z=0$)**: Shows how the mean value ($90$ kg) becomes the central reference point.
- **Statistical Narrative**: Each data point is re-expressed by standard deviation units.

### 🚀 Installation & Usage

1. **Prerequisites**: Ensure you have Python installed. After that, pulling Standardization.py.
2. **Install NumPy**:
```bash
pip install numpy
```
2. **Run the script**
```
python Standardization.py
```

## 🛠 Full Preprocessing Pipeline

A complete end-to-end workflow using Pandas and Scikit-learn to transform raw, messy data into optimized features for Machine Learning.

### 🔍 Pipeline Stages:

**Data Cleaning**: Imputing missing values via Mean/Median logic.

**Data Integration**: Merging disparate datasets (Demographics + Sales).

**Data Discretization**: Binning continuous variables into categorical labels.

**Data Transformation**: One-hot encoding and Min-Max scaling.

**Data Reduction**: Reducing dimensions using Principal Component Analysis (PCA).

### 🚀 Installation & Usage

1. **Install Scikit-learn & Pandas**:
```
pip install pandas scikit-learn
```
2. **Run the script**:
```
python Full_Pipeline.py
```\

