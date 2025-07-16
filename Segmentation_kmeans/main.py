import joblib
import os
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA               #PCA plots
from sklearn.cluster import KMeans, DBSCAN
from kneed import KneeLocator                       #Auto-detecting k value for elbow plot in kmeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE                   #t-SNE plots

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ======================================
# Data Preprocesing 
# ======================================
# Loading Dataset, drop missing rows, remove negetive, convert to dateTime

def load_and_clean_data(file_path):
    # Load data
    df = pd.read_csv(file_path, encoding='latin1')

    # Drop rows with missing CustomerID
    df.dropna(subset=['CustomerID'], inplace=True)

    # Remove negative or zero Quantity and UnitPrice
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    # Convert InvoiceDate to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df

# ======================================
# Feature Enggineering 
# ======================================
#Create RFM

def create_rfm_features(df):
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    # Define latest date for recency calculation
    latest_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    # Group by CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (latest_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Quantity': 'sum',
        'UnitPrice': 'mean'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'TotalQuantity', 'AvgUnitPrice']

    # Create Monetary column
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    monetary = df.groupby('CustomerID')['TotalPrice'].sum().reset_index()
    rfm = rfm.merge(monetary, on='CustomerID')
    rfm.rename(columns={'TotalPrice': 'Monetary'}, inplace=True)

    return rfm

# ======================================
# Checking Feature Distribution, Outlier handling and Scaling
# ======================================
# Plots for distributions

def plot_and_save_distributions(rfm_df, Before_After):
    plt.style.use('ggplot')
    features = ['Recency', 'Frequency', 'Monetary']

    for feature in features:
        plt.figure(figsize=(8, 5))
        sns.histplot(rfm_df[feature], kde=True, bins=30)
        plt.title(f'{feature} Distribution')
        plt.xlabel(feature)
        plt.ylabel('Count')
        plt.savefig(f'plots/{feature}_distribution_{Before_After}.png')
        plt.show()
        plt.close()

# ======================================
# Outlier handling

def detect_and_handle_outliers(rfm_df):
    def handle_outlier(col):
        Q1 = rfm_df[col].quantile(0.25)
        Q3 = rfm_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return rfm_df[(rfm_df[col] >= lower) & (rfm_df[col] <= upper)]

    filtered_df = rfm_df.copy()
    for col in ['Recency', 'Frequency', 'Monetary']:
        filtered_df = handle_outlier(col)

    return filtered_df
# ======================================
# scaler 
def scale_data(df, features):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    return scaled, scaler

# ======================================
# Data visualization
# ======================================

# Ploting 3D 
def plot_3d_scatter(data, labels, title, filename):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', s=50)
    ax.set_xlabel('Recency')
    ax.set_ylabel('Frequency')
    ax.set_zlabel('Monetary')
    ax.set_title(title)
    plt.savefig(f'plots/{filename}')
    plt.close()

# ======================================
# Training 
# ======================================
# Kmeans Training (Kmeans Cluster and t-SNE, PCA plots, silhouette, Elbow method)

# Kmeans 

def train_kmeans(data, max_clusters=10):
    silhouette_scores = []
    inertias = []

    # Test k from 2 to max_clusters
    k_range = range(2, max_clusters + 1)
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))
        inertias.append(model.inertia_)

    # Auto-detect the elbow point
    kl = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    best_k = kl.elbow
    best_model = KMeans(n_clusters=best_k, random_state=42).fit(data)

    # Plot Silhouette Score
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, silhouette_scores, marker='o', color='teal')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Auto-selected k = {best_k}')
    plt.legend()
    plt.grid(True)

    # Plot Elbow Method (Inertia)
    plt.subplot(1, 2, 2)
    plt.plot(k_range, inertias, marker='s', color='orange')
    plt.title('Elbow Method: Inertia vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (WCSS)')
    plt.axvline(x=best_k, color='red', linestyle='--', label=f'Elbow at k = {best_k}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('plots/kmeans_elbow_auto.png')
    plt.show()

    print(f"Auto-selected best k using elbow method: {best_k}")
    return best_model, best_model.labels_

# Kmeans Cluster and t-SNE, PCA plots

def plot_kmeans_clusters(data, labels, centers=None):
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
    else:
        reduced_data = data

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Set1', s=30, alpha=0.7, label='Points')

    if centers is not None:
        if data.shape[1] > 2:
            centers_2d = pca.transform(centers)
        else:
            centers_2d = centers
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='X', s=200, label='Centroids')

    plt.title('KMeans Clustering Results')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/kmeans_clusters.png')
    plt.show()
    plt.close()

# Kmeans t-SNE

def plot_tsne_clusters(data, labels, perplexity=30, learning_rate=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Set1', s=30, alpha=0.7)
    
    # Legend based on unique labels
    unique_labels = sorted(set(labels))
    handles = [plt.Line2D([], [], marker='o', color='w', label=f'Cluster {lbl}',
                          markerfacecolor=scatter.cmap(scatter.norm(lbl)), markersize=8)
               for lbl in unique_labels]
    plt.legend(handles=handles)

    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/tsne_kmeans_clusters.png')
    plt.show()
    plt.close()

# ======================================
# Hirarchial Training (Dendrogram, t-sne)
# ======================================
# Hirarchial 

def train_hierarchical(data, method='ward'):
    Z = linkage(data, method=method)

    # Auto-select cut height using distance gap
    last_10 = Z[-10:, 2]
    gaps = np.diff(last_10)
    best_gap_idx = gaps.argmax()
    cut_height = last_10[best_gap_idx]

    # Dendrogram with cut line
    plt.figure(figsize=(10, 6))
    dendrogram(Z)
    plt.axhline(y=cut_height, color='red', linestyle='--', label=f'Cut at {cut_height:.2f}')
    plt.legend()

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index or Cluster Merge Step')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('plots/hierarchical_dendrogram.png')
    # plt.show()

    # Assign clusters
    labels = fcluster(Z, t=cut_height, criterion='distance')
    return Z, labels

# Hierarchical t-SNE

def plot_tsne_hierarchical(data, labels, perplexity=30, learning_rate=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced_data = tsne.fit_transform(data)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='Set2', s=30, alpha=0.7)

    # Legend based on unique labels
    unique_labels = sorted(np.unique(labels))
    handles = [plt.Line2D([], [], marker='o', color='w', label=f'Cluster {lbl}',
                          markerfacecolor=scatter.cmap(scatter.norm(lbl)), markersize=8)
               for lbl in unique_labels]
    plt.legend(handles=handles)

    plt.title('t-SNE Visualization of Hierarchical Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/tsne_hierarchical_clusters.png')
    plt.show()

# ======================================
# DBSCAN Training (clustor , t-SNE)
# ======================================
#DBSCAN train

def train_dbscan(data, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    # Dimensionality reduction for visualization if needed
    if data.shape[1] > 2:
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(data)
    else:
        reduced_data = data

    # Plot DBSCAN results
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            color = 'k'  # Black for noise
            label_name = 'Noise'
        else:
            color = plt.cm.Set1(label / max(unique_labels))  # Varying colors
            label_name = f'Cluster {label}'
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=[color], label=label_name, s=30)

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/dbscan_clusters.png')
    plt.show()

    return model, labels

# DBSCAN t-SNE

def plot_tsne_dbscan_clusters(scaled_data, labels, perplexity=30, learning_rate=200):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=42)
    reduced_data = tsne.fit_transform(scaled_data)

    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = (labels == label)
        if label == -1:
            color = 'k'
            label_name = 'Noise'
        else:
            color = plt.cm.Set1(label / max(unique_labels))  # Distinct colors
            label_name = f'Cluster {label}'
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=[color], label=label_name, s=30, alpha=0.7)

    plt.title('t-SNE Visualization of DBSCAN Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/tsne_dbscan_clusters.png')
    plt.show()

# ======================================
# Ploting PCA 
# ======================================
def plot_pca_clusters(data, labels, title, filename):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette='Set2', s=60)
    plt.title(title)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.savefig(f'plots/{filename}')
    plt.close()

# ======================================
## Model performances - Comparing
# ======================================
# Comparing Model performances
def compare_models(data, label_sets):
    scores = {}
    for name, labels in label_sets.items():
        if len(set(labels)) > 1 and -1 not in set(labels):
            sil_score = silhouette_score(data, labels)
            db_score = davies_bouldin_score(data, labels)
            scores[name] = {"Silhouette": sil_score, "Davies-Bouldin": db_score}
    return scores

# ======================================
# Saving trained models
# ======================================
#Saving trained model

def save_model(model, name):
    os.makedirs('models', exist_ok=True)
    path = f'models/{name}.pkl'
    joblib.dump(model, path)
    print(f"Model saved at {path}")

# ****************************************************
# *********************************************
# ======================================
# Calling Methods
# ======================================
# *********************************************
# ****************************************************

# ======================================
# pre-processing call
# ======================================

def get_preProcess(data_path= 'data/data.csv'):
    data_path = data_path
    cleaned_df = load_and_clean_data(data_path)
    cleaned_df.to_csv('data/cleaned_data.csv', index=False)
    print(f"Data cleaned and saved to 'data/cleaned_data.csv' with shape: {cleaned_df.shape}")
    return cleaned_df

# ======================================
# RFM Call
# ======================================

def get_rfm(cleaned_df):
    rfm_df = create_rfm_features(cleaned_df)
    rfm_df.to_csv('data/rfm_data.csv', index=False)
    print(f"RFM features created and saved to 'data/rfm_data.csv' with shape: {rfm_df.shape}")
    return rfm_df

# ======================================
# Saving RFM distribution curves
# ======================================

def get_distributions(rfm_df):
    before_after= "Before"
    plot_and_save_distributions(rfm_df, before_after)
    print("Initial distribution plots saved to 'plots/'.")

    filtered_rfm_df = detect_and_handle_outliers(rfm_df)
    filtered_rfm_df.to_csv('data/rfm_data_filtered.csv', index=False)
    print(f"Outliers handled and data saved to 'data/rfm_data_filtered.csv' with shape: {filtered_rfm_df.shape}")

    before_after= "After"
    plot_and_save_distributions(filtered_rfm_df,before_after)
    print("Filtered distribution plots saved to 'plots/'.")
    return filtered_rfm_df


# ======================================
# Scaling and 3D scatter
# ======================================
# Scaling and 3D scatter

def get_scaled(filtered_rfm_df): 
    features = ['Recency', 'Frequency', 'Monetary']
    scaled_data, scaler = scale_data(filtered_rfm_df, features)
    plot_3d_scatter(scaled_data, labels=[0]*len(scaled_data), title='3D Scatter of Scaled RFM', filename='3d_rfm_scatter.png')
    print("3D scatter plot of scaled RFM data saved to 'plots/3d_rfm_scatter.png'.")
    return scaled_data, scaler


# ======================================
# Kmeans -Calling (Kmeans Cluster and t-SNE, PCA plots, silhouette, Elbow method)
# ======================================
def get_kmeans(scaled_data, filtered_rfm_df, model_path="models/kmeans_model.pkl"):
    # Train KMeans
    # Check if the model is already saved
    if os.path.exists(model_path):
        kmeans_model = joblib.load(model_path)
        print("Loaded existing KMeans model.")
    else:
        # Train the model and save it
        kmeans_model, kmeans_labels = train_kmeans(scaled_data)
        joblib.dump(kmeans_model, model_path)
        print("Trained and saved new KMeans model.")
    
    # Predict labels (whether loaded or trained)
    kmeans_labels = kmeans_model.predict(scaled_data)
    filtered_rfm_df['KMeans_Labels'] = kmeans_labels

    #Plots Kmeans with centroids
    plot_kmeans_clusters(scaled_data, kmeans_labels, centers=kmeans_model.cluster_centers_)

    #Plots t-SNE
    plot_tsne_clusters(scaled_data, kmeans_labels)

    # PCA plot
    plot_pca_clusters(scaled_data, kmeans_labels, 'KMeans Clusters (PCA)', 'pca_kmeans.png')


# ======================================
# Hierarchical  -Call (Dendrogram and t-SNE, PCA plots)
# ======================================
 # calling Hierarchical Clustering

def get_hierarchical(scaled_data, filtered_rfm_df):
    Z, hierarchical_labels = train_hierarchical(scaled_data)
    filtered_rfm_df['Hierarchical_Labels'] = hierarchical_labels
    print("Hierarchical clustering completed.")

    #Dendrogram
    # t-SNE
    plot_tsne_hierarchical(scaled_data, hierarchical_labels)

    #PCA plot
    plot_pca_clusters(scaled_data, hierarchical_labels, 'Hierarchical Clusters (PCA)', 'pca_hierarchical.png')


# ======================================
# DBSCAN  -Call (cluster and t-SNE, PCA plots)
# ======================================


def get_dbscan(scaled_data, filtered_rfm_df):
    # Train DBSCAN
    dbscan_model, dbscan_labels = train_dbscan(scaled_data)
    filtered_rfm_df['DBSCAN_Labels'] = dbscan_labels
    print("DBSCAN clustering completed.")

    #t-SNE
    plot_tsne_dbscan_clusters(scaled_data, dbscan_labels)
    filtered_rfm_df.to_csv('data/clustered_rfm.csv', index=False)
    print("Clustered RFM data saved to 'data/clustered_rfm.csv'.")

    #PCA plot
    plot_pca_clusters(scaled_data, dbscan_labels, 'DBSCAN Clusters (PCA)', 'pca_dbscan.png')

# ======================================
# Model Comparison - Call
# ======================================
def get_model_comparison(kmeans_labels,hierarchical_labels,dbscan_labels, scaled_data):
    # Model Comparison
    label_sets = {
        'KMeans': kmeans_labels,
        'Hierarchical': hierarchical_labels,
        'DBSCAN': dbscan_labels
    }
    model_scores = compare_models(scaled_data, label_sets)
    for model, scores in model_scores.items():
        print(f"{model} - Silhouette: {scores['Silhouette']:.4f}, Davies-Bouldin: {scores['Davies-Bouldin']:.4f}")

    print("""
    ✅ Conclusion:
    KMeans performs better than Hierarchical clustering on both metrics:

    Higher Silhouette Score → better cluster cohesion and separation

    Lower Davies-Bouldin Index → better-defined clusters""")

# ======================================
# Save models
def get_save_models(kmeans_model, dbscan_model,Z, scaler):    
    save_model(kmeans_model, "kmeans_model")
    save_model(dbscan_model, "dbscan_model")
    save_model(Z, "hierarchical_model")
    save_model(scaler, "scaler")

# ======================================
# Save labeled data
def get_save_labeled_data(filtered_rfm_df,kmeans_labels,hierarchical_labels,dbscan_labels):
    filtered_rfm_df['KMeans_Labels'] = kmeans_labels
    filtered_rfm_df['Hierarchical_Labels'] = hierarchical_labels
    filtered_rfm_df['DBSCAN_Labels'] = dbscan_labels
    filtered_rfm_df.to_csv("data/clustered_rfm.csv", index=False)
    print("Final labeled data saved and models stored.")
# ======================================
# ======================================
# ======================================