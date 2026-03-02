import pandas as pd
import numpy as np

class DataPreprocessor:
    """
    A modular pipeline class to handle data ingestion, cleaning, 
    and preprocessing for RFM Customer Segmentation.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df = None

    def process_data(self) -> pd.DataFrame:
        """
        Executes the end-to-end loading and cleaning process.
        """
        print(f"[*] Loading dataset from: {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path, encoding="utf-8")
            print("[+] Dataset loaded successfully.")
        except Exception as e:
            print(f"[-] Error loading dataset: {e}")
            return None

        print("[*] Initiating data cleaning protocol...")
        self.df.dropna(subset=['customer_id'], inplace=True)
        
        self.df['order_id'] = self.df['order_id'].astype(str)
        self.df = self.df[~self.df['order_id'].str.startswith('C')]
        
        self.df = self.df[(self.df['quantity'] > 0) & (self.df['price'] > 0)]
        self.df['total_revenue'] = self.df['quantity'] * self.df['price']
        self.df['order_date'] = pd.to_datetime(self.df['order_date'])
        self.df['customer_id'] = self.df['customer_id'].astype(int)
        
        print("[+] Data cleaning completed.\n")
        return self.df