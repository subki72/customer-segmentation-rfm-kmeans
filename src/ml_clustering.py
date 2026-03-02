import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class CustomerClustering:
    """
    A machine learning pipeline for clustering customers using K-Means.
    Includes data normalization and model fitting.
    """
    
    def __init__(self, rfm_df: pd.DataFrame):
        self.rfm_df = rfm_df.copy()
        self.scaled_data = None
        
    def preprocess_for_ml(self):
        """
        Applies Log Transformation and Standardization to the RFM features.
        """
        print("[*] Preprocessing RFM data for Machine Learning...")
        log_data = self.rfm_df[['recency', 'frequency', 'monetary']].copy()
        for col in log_data.columns:
            log_data[col] = np.log1p(log_data[col])
            
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(log_data)
        print("[+] Data log-transformed and scaled successfully.")
        
    def fit_kmeans(self, n_clusters: int = 4) -> pd.DataFrame:
        """
        Fits the K-Means algorithm and assigns clusters to the dataframe.
        """
        print(f"[*] Fitting K-Means model with {n_clusters} clusters...")
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(self.scaled_data)
        
        self.rfm_df['ml_cluster'] = kmeans_model.labels_
        print("[+] K-Means clustering completed.\n")
        return self.rfm_df