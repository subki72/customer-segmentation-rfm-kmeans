import pandas as pd
import numpy as np
import datetime as dt

class RFMEngine:
    """
    A class to calculate RFM metrics, compute quintile scores, 
    and assign advanced actionable business segments.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.rfm = None

    def generate_rfm(self) -> pd.DataFrame:
        """
        Executes the full RFM calculation and segmentation pipeline.
        """
        print("[*] Initializing RFM calculations...")
        snapshot_date = self.df['order_date'].max() + dt.timedelta(days=1)
        
        self.rfm = self.df.groupby('customer_id').agg({
            'order_date': lambda x: (snapshot_date - x.max()).days,
            'order_id': 'nunique',
            'total_revenue': 'sum'
        }).reset_index()
        
        self.rfm.rename(columns={
            'order_date': 'recency',
            'order_id': 'frequency',
            'total_revenue': 'monetary'
        }, inplace=True)
        
        self.rfm['r_score'] = pd.qcut(self.rfm['recency'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1])
        self.rfm['f_score'] = pd.qcut(self.rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        self.rfm['m_score'] = pd.qcut(self.rfm['monetary'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
        
        self.rfm['rf_score'] = self.rfm['r_score'].astype(str) + self.rfm['f_score'].astype(str)
        
        seg_map = {
            r'[1-2][1-2]': 'Hibernating', r'[1-2][3-4]': 'At Risk', r'[1-2]5': 'Cannot Lose Them',
            r'3[1-2]': 'About To Sleep', r'33': 'Need Attention', r'[3-4][4-5]': 'Loyal Customers',
            r'41': 'Promising', r'51': 'New Customers', r'[4-5][2-3]': 'Potential Loyalists',
            r'5[4-5]': 'Champions'
        }
        self.rfm['base_segment'] = self.rfm['rf_score'].replace(seg_map, regex=True)
        self.rfm['value_tier'] = np.where(self.rfm['m_score'].astype(int) >= 4, 'High Value', 'Standard Value')
        self.rfm['actionable_segment'] = self.rfm['value_tier'] + ' ' + self.rfm['base_segment']
        
        print("[+] RFM Pipeline completed successfully.\n")
        return self.rfm