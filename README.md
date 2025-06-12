[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# ba-simulation

End-to-end data science project simulating British Airways operations, focusing on:
1.  **Lounge Eligibility Lookup Model**
2.  **Flight Booking Prediction Model**

---

## 🚀 Overview

This repository showcases an end-to-end data science workflow, transforming over 50,000 raw booking records into actionable insights for British Airways. The project is divided into four key stages:

-   **Stage 1 (CDAS.py): Data Preparation & EDA**
    -   Load and clean 50,000+ booking records.
    -   Engineer features: time-of-day, route type, region, loyalty tier.
    -   Perform Exploratory Data Analysis (EDA).
-   **Stage 2 (CDAS2.py): Model Training**
    -   Train and tune a `RandomForestClassifier`.
    -   Save the optimized model and pre-processing pipeline.
-   **Stage 3 (CDAS3.py): Model Evaluation**
    -   Generate predictions on the test set.
    -   Compute key metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC (**0.79**).
    -   Visualize performance: Confusion Matrix, ROC curve, Precision-Recall curve.
-   **Stage 4 (CDAS4.py): Insights & Interpretation**
    -   Analyze feature importances (including cumulative importance).
    -   Generate Partial Dependence Plots (PDPs) and explore interaction effects.
    -   Create correlation heatmaps and category breakdown charts.

**Key Insight:** Origin-route pairings and booking timing significantly outweigh price as predictors of flight purchases.

---

## 📁 Repository Structure
```bash
ba-simulation/
├── CDAS_img/ # Stage 1 visuals (cleaning & EDA)
├── CDAS3_img/ # Stage 3 visuals (confusion, ROC/PR)
├── CDAS4_img/ # Stage 4 visuals (feature plots, heatmaps)
│
├── Code/
│ ├── CDAS.py # Stage 1: Data prep, cleaning, EDA
│ ├── CDAS2.py # Stage 2: Model training & tuning
│ ├── CDAS3.py # Stage 3: Predictions & metrics
│ └── CDAS4.py # Stage 4: Advanced visualisations
│
├── data/
│ ├── customer_booking.csv # Raw booking dataset
│ └── final_processed_dataset_stage1.pkl # Cleaned & feature-engineered data
│
├── models/
│ ├── best_random_forest_model.pkl # Trained RandomForest model object
│ └── model_eval_components.pkl # Metrics & plot data for evaluation
│
├── customer_booking_model_slide.pptx # Presentation deck of model results
├── Lounge Eligibility Lookup Table.xlsx # Output table for lounge eligibility
├── requirements.txt # Python dependencies
└── README.md # This file
```
---

## 🗃️ Data Description

-   **`data/customer_booking.csv`**:
    -   Raw dataset containing customer booking information.
    -   Features include: origin, destination, departure time, booking lead time, fare class, loyalty tier, sentiment flags, etc.
    -   Target variable: `booking_made` (binary: Yes/No), indicating if a flight booking was completed.
-   **`data/final_processed_dataset_stage1.pkl`**:
    -   Processed data after cleaning, encoding, normalization, and feature engineering. Ready for model training.

---

## ⚙️ Setup & Installation

**Prerequisites:**
*   Python 3.8+

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/joshua-the-analyst/ba-simulation.git
    cd ba-simulation
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # Using venv (Python's built-in)
    python3 -m venv venv

    # On macOS/Linux
    source venv/bin/activate

    # On Windows (PowerShell)
    .\venv\Scripts\Activate.ps1
    # Or Windows (Command Prompt)
    # venv\Scripts\activate.bat
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## ▶️ Usage

Run the Python scripts sequentially from the `Code/` directory to execute the pipeline:

python Code/CDAS.py      # Stage 1: Data preparation & EDA

python Code/CDAS2.py     # Stage 2: Train & save model 

python Code/CDAS3.py     # Stage 3: Predictions & metrics

python Code/CDAS4.py     # Stage 4: Advanced visualisations


📊 Results & Key Findings

Flight Booking Prediction Model Performance (Test Set):

ROC-AUC: 0.79

Other metrics (Accuracy, Precision, Recall, F1-score) are detailed in Stage 3 outputs and visualizations.

Top Predictors for Flight Booking:

Origin–Route Pair

Departure Time Window

Customer Loyalty Tier

Lounge Demand Insight: The highest premium lounge eligibility is observed for early-morning long-haul flights and flights to/from North America.


🤝 Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check the issues page if you have any.

You can also fork the repository and submit a pull request.


📫 Contact

Oluwadarasimi Bamisaye Joshua - GitHub Profile

Project Link: https://github.com/joshua-the-analyst/ba-simulation


📝 License

This project is licensed under the MIT License.

Copyright © 2025 Oluwadarasimi Bamisaye Joshua.
