import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class SegmentVisualizer:
    """
    A class dedicated to generating industry-standard visualizations.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'figure.autolayout': True})

    def plot_segment_distribution(self):
        print("[*] Generating Actionable Segment Distribution plot...")
        plt.figure(figsize=(14, 7))
        segment_counts = self.df['actionable_segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        ax = sns.barplot(data=segment_counts, x='Count', y='Segment', palette='viridis', hue='Segment', legend=False)
        plt.title('Customer Distribution by Actionable Segment', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Customers', fontsize=12)
        plt.ylabel('Actionable Segment', fontsize=12)
        plt.show()

    def plot_ml_clusters_scatter(self):
        print("[*] Generating Machine Learning Clusters scatter plot...")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=self.df, x='recency', y='frequency', hue='ml_cluster', palette='Set1',
                        size='monetary', sizes=(20, 400), alpha=0.6)
        
        plt.title('K-Means Clusters: Recency vs Frequency', fontsize=16, fontweight='bold')
        plt.xlabel('Recency (Days)', fontsize=12)
        plt.ylabel('Frequency (Transactions)', fontsize=12)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(title='ML Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()