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
