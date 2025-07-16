
# 🧠 Customer Segmentation and Clustering Dashboard

An interactive Streamlit dashboard for performing customer segmentation using clustering techniques such as KMeans, Hierarchical Clustering, and DBSCAN. The application includes end-to-end data preprocessing, RFM feature generation, outlier handling, and insightful visualizations.

---

## 📂 Project Structure

```
project/
├── app.py                  # Main Streamlit application
├── main.py                 # Backend utility functions
├── data/                   # Stores uploaded and processed datasets
├── models/                 # Stores saved models
├── plots/                  # Stores generated plots
```

---

## ⚙️ Features

- Upload your own dataset (CSV format)
- Clean and preprocess data
- Generate RFM (Recency, Frequency, Monetary) features
- Handle outliers and visualize distributions
- Scale and visualize RFM features in 3D
- Apply and visualize clustering techniques:
  - KMeans Clustering
  - Hierarchical Clustering (Agglomerative)
  - DBSCAN Clustering
- View t-SNE, PCA, and 3D visualizations

---

## 🔧 Setup Instructions

### 🔗 Prerequisites

Install required Python packages:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn plotly
```

### ▶️ Running the App

```bash
streamlit run app.py
```

---

## 🧮 RFM Explained

- **Recency**: Time since the last purchase
- **Frequency**: Number of purchases in a given time frame
- **Monetary**: Total amount spent by the customer

These features form the basis for clustering customers based on behavior.

---

## 📊 Clustering Visualizations

Each clustering algorithm generates:

- Cluster assignments (colored scatter plots)
- Dimensionality reduction (t-SNE, PCA)
- Elbow and dendrogram (as applicable)
- 3D segmentation plots

---

## 🧠 Backend Functions (in `main.py`)

Expected utility functions:

- `get_preProcess(path)`
- `get_rfm(df)`
- `get_distributions(df)`
- `get_scaled(df)`
- `scale_data(df, columns)`
- `get_kmeans(scaled_data, df)`
- `get_hierarchical(scaled_data, df)`
- `get_dbscan(scaled_data, df)`

---

## ✅ Output

- Cleaned data views
- RFM table preview
- Filtered distributions (before/after outliers)
- Scaled 3D plots
- Clustering results for each algorithm

---

## 📌 Notes

- Modular and visual-first design for ease of understanding
- Suitable for business use cases in customer segmentation
- All models and plots are saved for reproducibility
# ML_projects
