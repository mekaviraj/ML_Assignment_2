# Improved K-Means Clustering Using Evidence Distance

This project reproduces and extends the research paper  
"An Improved K-Means Algorithm Based on Evidence Distance" (Entropy, 2021) by Zhu et al.

The original paper proposed replacing the traditional Euclidean distance in K-Means with an Evidence Distance derived from Dempster–Shafer theory, improving the quality and stability of clustering.  
In this project, I reproduced their approach and then introduced additional enhancements such as K-Means++ initialization and feature preprocessing (scaling and PCA) to further improve clustering accuracy and convergence.

## 📘 Project Overview
This repository was created as part of Assignment 2 for the Machine Learning course.  
The goal was to select a research paper in ML, reproduce its results, identify research gaps, and demonstrate improvements through experiments and analysis.

### Core idea:
Evaluate how initialization methods and preprocessing affect the performance of Evidence Distance–based K-Means on benchmark data.

## ⚙️ Methodology Summary
Three main models were implemented and compared:

| Method                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Baseline K-Means      | Standard scikit-learn K-Means using Euclidean distance.                     |
| ED-KMeans (Paper Model)| Implementation of the algorithm proposed in the paper using Evidence Distance and random initialization. |
| Improved ED-KMeans    | My extended version of ED-KMeans using K-Means++ initialization and preprocessing (StandardScaler and optional PCA). |

All methods were tested on the Iris dataset from `sklearn.datasets`.

## 🧩 Implementation Details
- **Programming Language**: Python 3
- **Libraries Used**:
  - NumPy
  - scikit-learn
  - matplotlib

### Project Structure:
src/
├── baseline.py # Standard K-Means implementation
├── ed_kmeans.py # Evidence Distance K-Means implementation
├── final_ml_2.ipynb # Notebook with full experiments, results & plots
├── Report.pdf # Detailed project report
└── README.md # This file

## 🧪 Experimental Setup
- **Dataset**: Iris (150 samples, 4 features, 3 classes)
- **Cluster count (k)**: 3
- **Metrics Used**:
  - Adjusted Rand Index (ARI)
  - Silhouette Score
  - Number of Iterations (for convergence)

### Procedure:
1. Run baseline K-Means and record ARI, Silhouette, iterations.
2. Run ED-KMeans with random initialization (reproduction of the paper).
3. Run improved ED-KMeans with K-Means++ and preprocessing.
4. Compare results, visualize clusters using PCA, and analyze improvements.

## 📊 Results Summary

| Method                | ARI   | Silhouette | Iterations |
|-----------------------|-------|------------|------------|
| Baseline K-Means      | 0.73  | 0.55       | 9          |
| ED-KMeans (random)    | 0.78  | 0.58       | 11         |
| Improved ED-KMeans    | 0.82  | 0.61       | 7          |

- **Evidence Distance** improved both clustering quality and stability.
- **K-Means++ initialization** reduced iteration count and improved consistency.
- **Preprocessing** (scaling and PCA) gave slightly better separation and compact clusters in PCA visualization.

## 📈 Visualizations
The notebook includes:
- PCA 2D Cluster Plots for all three methods.
- Elbow Method plot for determining optimal k.
- Silhouette Score plot to validate cluster quality.

## 📄 Report
A detailed **Report.pdf** is included in the repository, containing:
- Introduction to clustering and K-Means
- Summary of the reference paper
- Identified research gaps
- Methodology and experimental setup
- Results, discussion, and conclusion

## 💡 Key Takeaways
- Replacing Euclidean distance with **Evidence Distance** improves clustering performance.
- **Initialization** plays a crucial role in stability and convergence speed.
- Simple **preprocessing** steps like scaling and PCA can make a measurable difference.
- Even small algorithmic refinements can lead to meaningful improvements in clustering outcomes.

## 🚀 How to Run
Clone this repository and run the scripts:

```bash
# Baseline K-Means
python src/baseline.py

# Evidence Distance K-Means
python src/ed_kmeans.py
Or open the Jupyter Notebook for full results:

jupyter notebook src/final_ml_2.ipynb

References

Zhu, A., Hua, Z., Shi, Y., Tang, Y., & Miao, L. (2021).
An Improved K-Means Algorithm Based on Evidence Distance.
Entropy, 23(11), 1550. https://doi.org/10.3390/e23111550