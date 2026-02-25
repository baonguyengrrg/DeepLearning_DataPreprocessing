import numpy as np
def construct():
    data = np.array([70, 80, 90, 100, 110], dtype=float)
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    print("Original data:", data)
    print("Mean:", mean)
    print("Standard Deviation:", std_dev)
    print("Z-scores:", z_scores)
if __name__ == "__main__":
    construct()
