# Customer Segmentation and Value Tiering via RFM Analysis & K-Means Clustering

## Executive Summary
This project presents an end-to-end data pipeline and machine learning implementation designed to segment a retail customer base. Moving beyond basic heuristic segmentation, this analysis integrates Recency, Frequency, and Monetary (RFM) modeling with K-Means clustering. The primary objective is to identify high-value customer cohorts, mitigate churn risk, and provide actionable, data-driven recommendations to optimize marketing ROI.

## Business Problem
In a highly competitive retail environment, applying a uniform marketing strategy to all customers leads to inefficient budget allocation. The business requires a precise understanding of its customer base to:
1. Retain top-tier customers (Champions).
2. Reactivate high-value customers who are showing signs of churn.
3. Minimize marketing expenditure on low-value, dormant accounts.

## Methodology
The project is structured into four distinct phases, implemented using Object-Oriented Programming (OOP) principles for scalability and reproducibility:

1. **Data Ingestion & Preprocessing:** Developed a `DataPreprocessor` class to handle raw data cleaning, including the removal of anomalies, null values, and cancelled transactions, followed by feature engineering of baseline revenue metrics.
2. **RFM Feature Engineering & Value Tiering:** Built an `RFMEngine` class to calculate RFM metrics. Applied quintile scoring and regex mapping to define base segments, subsequently introducing a "Monetary Value Tier" (High Value vs. Standard Value) to prioritize business actions.
3. **Machine Learning Integration:** Implemented a `CustomerClustering` class utilizing K-Means. The data was normalized using logarithmic transformation to handle right-skewness and standardized to ensure equal feature contribution, allowing the algorithm to uncover hidden behavioral patterns independent of manual rules.
4. **Data Visualization & Strategy:** Created a `SegmentVisualizer` class to generate industry-standard distributions and scatter plots, translating model outputs into clear business strategies.

## Key Strategic Insights
Based on the synthesis of the RFM pipeline and K-Means clustering, the actionable recommendations are as follows:

* **High Value Champions & Loyalists:** Prioritize retention through exclusive early-access programs and VIP perks rather than margin-reducing discounts.
* **High Value At Risk:** These cohorts require immediate intervention. Deploy aggressive, targeted win-back campaigns (e.g., personalized high-value discount codes) to prevent the loss of significant historical revenue.
* **Standard/Low Value Hibernating:** Restrict marketing spend. Transition these accounts to low-cost, automated email pipelines to maintain baseline engagement without eroding profitability.

## Technology Stack
* **Language:** Python 3.x
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib, Seaborn

## Repository Structure
```text
├── data/
│   └── raw_data.csv             # The raw dataset (not tracked in version control)
├── notebooks/
│   └── User_Segmentation.ipynb  # Comprehensive analysis and execution notebook
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py         # DataPreprocessor class
│   ├── rfm_engine.py            # RFMEngine class
│   ├── ml_clustering.py         # CustomerClustering class
│   └── visualizer.py            # SegmentVisualizer class
├── main.py                      # Main execution script for the modular pipeline
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation

```

## How to Run
1. Clone the repository to your local machine.
2. Install the required dependencies: `pip install -r requirements.txt`
3. To view the complete exploratory and analytical process, open and run `notebooks/User_Segmentation.ipynb`.
4. To execute the production-ready pipeline, place your dataset in the `data/` directory and run: `python main.py`

