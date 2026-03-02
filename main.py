from src.data_pipeline import DataPreprocessor
from src.rfm_engine import RFMEngine
from src.ml_clustering import CustomerClustering
from src.visualizer import SegmentVisualizer

def main():
    print("==================================================")
    print(" CUSTOMER SEGMENTATION PIPELINE INITIATED")
    print("==================================================\n")

    # 1. Data Processing
    FILE_PATH = 'Online Retail Data.csv'
    preprocessor = DataPreprocessor(FILE_PATH)
    df_clean = preprocessor.process_data()

    if df_clean is None:
        print("[-] Pipeline terminated due to data loading error.")
        return

    # 2. RFM Calculation
    rfm_engine = RFMEngine(df_clean)
    df_rfm = rfm_engine.generate_rfm()

    # 3. Machine Learning Integration
    ml_pipeline = CustomerClustering(df_rfm)
    ml_pipeline.preprocess_for_ml()
    df_final = ml_pipeline.fit_kmeans(n_clusters=4)

    # 4. Visualization & Reporting
    visualizer = SegmentVisualizer(df_final)
    visualizer.plot_segment_distribution()
    visualizer.plot_ml_clusters_scatter()

    print("\n==================================================")
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("==================================================")

if __name__ == "__main__":
    main()