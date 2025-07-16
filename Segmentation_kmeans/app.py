import streamlit as st
import pandas as pd
import os
from main import *

# Ensure required folders exist - -- ---
os.makedirs('plots', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

st.set_page_config(layout="wide")
st.title("🧠 Customer Segmentation and Clustering Dashboard")

# Sidebar - File Upload
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    data_path = os.path.join("data", "data.csv")
    with open(data_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success("✅ Data uploaded successfully.")

# Preprocessing
if st.sidebar.button("🧹 Run Preprocessing"):
    cleaned_df = get_preProcess(data_path="data/data.csv")
    st.dataframe(cleaned_df.head())
    st.success("✅ Preprocessing completed.")

# RFM Feature Engineering
if st.sidebar.button("🧮 Generate RFM Features"):
    cleaned_df = pd.read_csv("data/cleaned_data.csv")
    rfm_df = get_rfm(cleaned_df)
    st.dataframe(rfm_df.head())
    st.success("✅ RFM features generated.")

# Distribution & Outliers
if st.sidebar.button("📊 Handle Outliers & Visualize"):
    rfm_df = pd.read_csv("data/rfm_data.csv")
    filtered_rfm_df = get_distributions(rfm_df)
    col1, col2 = st.columns(2)
    with col1:
        st.image("plots/Frequency_distribution_Before.png", caption="Before")
    with col2:
        st.image("plots/Frequency_distribution_After.png", caption="After")

    with col1:
        st.image("plots/Monetary_distribution_Before.png", caption="Before")
    with col2:
        st.image("plots/Monetary_distribution_After.png", caption="After")

    with col1:
        st.image("plots/Recency_distribution_Before.png", caption="Before")
    with col2:
        st.image("plots/Recency_distribution_After.png", caption="After")

    # st.image("plots/Monetary_distribution_Before.png")
    # st.image("plots/Monetary_distribution_After.png")
    # st.image("plots/Recency_distribution_Before.png")
    # st.image("plots/Recency_distribution_After.png")

# Scaling & 3D Scatter
if st.sidebar.button("📈 Scale & Plot 3D"):
    filtered_rfm_df = pd.read_csv("data/rfm_data_filtered.csv")
    scaled_data, scaler = get_scaled(filtered_rfm_df)
    st.image("plots/3d_rfm_scatter.png")

# KMeans
if st.sidebar.button("🔀 Run KMeans Clustering"):
    scaled_data = pd.read_csv("data/rfm_data_filtered.csv")[['Recency', 'Frequency', 'Monetary']]
    scaled_data, _ = scale_data(scaled_data, ['Recency', 'Frequency', 'Monetary'])
    filtered_rfm_df = pd.read_csv("data/rfm_data_filtered.csv")
    
    get_kmeans(scaled_data, filtered_rfm_df)
    st.image("plots/kmeans_elbow_auto.png")
    st.image("plots/kmeans_clusters.png")
    st.image("plots/tsne_kmeans_clusters.png")
    st.image("plots/pca_kmeans.png")

    # 3D Scatter Plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(filtered_rfm_df['Recency'], filtered_rfm_df['Frequency'], filtered_rfm_df['Monetary'], c=filtered_rfm_df['KMeans_Labels'], cmap='viridis', s=60)
    ax.set_xlabel("Recency")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Monetary")
    ax.set_title("3D Customer Segmentation (KMeans)")
    st.pyplot(fig)


# Hierarchical Clustering
if st.sidebar.button("🌿 Run Hierarchical Clustering"):
    scaled_data = pd.read_csv("data/rfm_data_filtered.csv")[['Recency', 'Frequency', 'Monetary']]
    scaled_data, _ = scale_data(scaled_data, ['Recency', 'Frequency', 'Monetary'])
    filtered_rfm_df = pd.read_csv("data/rfm_data_filtered.csv")

    get_hierarchical(scaled_data, filtered_rfm_df)
    st.image("plots/hierarchical_dendrogram.png")
    # st.image("plots/hierarchical_clusters.png")
    st.image("plots/tsne_hierarchical_clusters.png")
    st.image("plots/pca_hierarchical.png")

# DBSCAN Clustering
if st.sidebar.button("🧬 Run DBSCAN Clustering"):
    scaled_data = pd.read_csv("data/rfm_data_filtered.csv")[['Recency', 'Frequency', 'Monetary']]
    scaled_data, _ = scale_data(scaled_data, ['Recency', 'Frequency', 'Monetary'])
    filtered_rfm_df = pd.read_csv("data/rfm_data_filtered.csv")
    
    get_dbscan(scaled_data, filtered_rfm_df)
    # st.image("plots/dbscan_eps_silhouette.png")
    st.image("plots/dbscan_clusters.png")
    st.image("plots/tsne_dbscan_clusters.png")
    st.image("plots/pca_dbscan.png")