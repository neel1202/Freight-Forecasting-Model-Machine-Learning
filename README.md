
# 🚢 Freight Forecasting Model — Machine Learning

An end-to-end, reproducible Machine Learning pipeline for medium-term freight demand forecasting, capacity planning, and logistics cost optimization.


---

## 📌 Executive Summary

In supply chain and intermodal logistics, freight station operations managers must charter cargo ships, secure warehouse bay storage, and roster specialized labor **7 days in advance**. Waiting until demand spikes forces managers into the **"rush window"** — spot market auctions where competing logistics providers bid up prices by 15% to 35%.

This project provides an end-to-end machine learning system that:
1. **Uncovers the Core Forecasting Insight:** Rigorous exploratory data analysis (EDA) shows raw *hourly* demand has near-zero autocorrelation ($|r| \le 0.01$) and is unlearnable ($R^2 \approx 0.02$). However, freight hubs do not book for a single hour — they allocate capacity against a **30-day moving average (planning demand)**.
2. **Predicts 7 Days Ahead:** Models the 30-day planning demand 7 days into the future (168-hour lead time) using chronological time-series cross-validation.
3. **Beats Naive Persistence:** Outperforms the challenging naive persistence benchmark on unseen test data, reaching **$R^2 = 0.5717$** and lowering RMSE from 83.11 to 79.26.
4. **Translates Predictions into Dollars:** Feeds the numeric forecast and demand bands into a capacity decision engine that sizes ships, square footage, and workforce, providing exact cost avoidance figures (e.g., **$72,813 saved** on a single weekly forecast cycle).

---

## 🏗️ 12-Stage Machine Learning Pipeline

The project follows a rigorous 12-stage machine learning methodology designed for transparency, reproducibility, and zero data leakage:

```mermaid
flowchart LR
    A[1. Problem Definition] --> B[2. Data Acquisition]
    B --> C[3. Data Preparation]
    C --> D[4. Preprocessing]
    D --> E[5. Balancing Check]
    E --> F[6. Rigorous EDA]
    F --> G[7. Feature Engineering]
    G --> H[8. Baselines B1/B2/B3]
    H --> I[9. ML Model Training]
    I --> J[10. TimeSeriesSplit CV]
    J --> K[11. Test Evaluation]
    K --> L[12. Decision Engine & Impact]
```

| Stage | Focus & Methodology | Key Findings / Decisions |
|:---:|:---|:---|
| **1** | **Problem Definition** | Freight hub operations manager needs a 7-day ahead forecast for booking decisions. | Frame as a 168-hour horizon regression problem with operational risk metrics. |
| **2** | **Data Acquisition** | Dynamic Supply Chain Logistics Dataset (32,065 hourly rows, Jan 2021 – Aug 2024). | 26 features covering logistics telemetry, weather, congestion, and supplier reliability. |
| **3** | **Data Preparation** | Chronological ordering, timestamp parsing, index integrity, and gap verification. | Validated continuous hourly cadence across 3+ years without duplicate timestamps. |
| **4** | **Data Preprocessing** | Missing value auditing, ordinal encoding for risk, feature scaling. | 0 missing values; mapped `risk_classification` ordinally (`Low`=0, `Moderate`=1, `High`=2). |
| **5** | **Data Balancing** | Continuous regression distribution check; evaluate whether SMOTE is valid. | Continuous regression target requires no synthetic balancing; SMOTE rejected to prevent temporal fabrication. |
| **6** | **Exploratory Data Analysis** | Target distribution, temporal drift, multi-lag autocorrelation, feature cross-correlations. | **Key Discovery:** Raw hourly demand has $|r| \le 0.01$ with all features; hourly demand behaves like white noise. |
| **7** | **Feature Engineering** | 30-day rolling planning demand target ($t+168\text{h}$), multi-period lag vectors, calendar cyclicality. | Target self-correlation at 7 days increases from $+0.007$ (raw) to $+0.771$ (planning level). |
| **8** | **Baselines** | Construct 3 reference models: Mean predictor (B1), Naive Persistence (B2), Raw Hourly Ridge (B3). | B3 fails ($R^2 \approx 0.02$). B2 sets a strong benchmark ($R^2 = 0.5626$, RMSE = 90.71). |
| **9** | **ML Modelling** | Train diverse model families: Regularized Linear (Ridge), Bagging (Random Forest), Boosting (GBDT). | Ridge establishes early dominance due to strong, smooth linear relationships among lag features. |
| **10** | **Hyperparameter Tuning** | `TimeSeriesSplit(n_splits=3)` Grid Search to prevent future information leakage. | Ridge ($\alpha=100.0$) selected as the Champion model over RF and GBDT. |
| **11** | **Validation & Evaluation** | Held-out 15% chronological test set evaluation, horizon sensitivity, 3-class tercile classification. | Ridge beats naive persistence on unseen test data ($R^2 = 0.5717$, Accuracy = 75.3% across demand bands). |
| **12** | **Use Case & Explainability** | Exact linear contribution decomposition, ship/space/labour sizing, rush-window cost savings. | Explains feature drivers for each forecast; demonstrates **$72,813** cost avoidance. |

---

## 📊 Dataset Overview

- **Source:** Kaggle — *Logistics and Supply Chain Dataset*
- **License:** Creative Commons Zero (CC0: Public Domain)
- **Observations:** 32,065 hourly intervals (January 1, 2021 → August 29, 2024)
- **Dimensionality:** 26 columns (24 continuous/discrete numeric, 1 categorical risk level, 1 datetime index)

### Core Feature Groups
* **Supply Chain Telemetry:** `traffic_congestion_level`, `port_congestion_level`, `warehouse_inventory_level`, `shipping_costs`, `lead_time_days`, `eta_variation_hours`.
* **Disruptions & Quality:** `weather_condition_severity`, `supplier_reliability_score`, `loading_unloading_time`, `handling_equipment_availability`, `customs_clearance_time`, `delay_probability`, `risk_classification`.
* **Engineered Signals:** 30-day rolling mean (`planning_demand`), autoregressive lags (`planning_lag_24`, `planning_lag_48`, `planning_lag_168`, `planning_lag_336`, `planning_lag_720`), short-term 7-day rolling statistics (`demand_mean_7d`, `demand_std_7d`), and calendar indicators (`hour_of_day`, `day_of_week`, `month`).

---

## 📈 Benchmark & Experimental Results

### 1. Final Model Performance on Held-Out Test Set (Chronological Last 15%)
*The test set represents future timestamps never exposed during scaling, training, or hyperparameter selection.*

| Model | MAE | RMSE | $R^2$ Score | Status / Note |
|:---|:---:|:---:|:---:|:---|
| **B1: Mean Predictor** | 134.16 | 155.47 | -0.6482 | Fails completely due to long-term regime shift |
| **B2: Naive Persistence** | 67.13 | 83.11 | 0.5290 | Assumes planning demand remains unchanged |
| **Ridge Regression (Tuned)** | **65.13** | **79.26** | **0.5717** | **Champion: Beats naive persistence across all metrics** |

> **Significance:** In time-series forecasting, beating naive persistence ($y_{t+h} = y_t$) is notoriously difficult. Our tuned model reduces RMSE by **4.6%** ($83.11 \rightarrow 79.26$) and lifts $R^2$ from **0.5290 to 0.5717**, proving legitimate predictive power.

---

### 2. Validation Set Comparison (Pre-Tuning & Post-Tuning)

| Model Family | Untuned $R^2$ | Tuned $R^2$ | Tuned MAE | Tuned RMSE | Best Hyperparameters |
|:---|:---:|:---:|:---:|:---:|:---|
| **Ridge Regression** | 0.5587 | **0.5577** | **73.68** | **91.21** | `alpha: 100.0` |
| **Gradient Boosting (GBR)** | 0.2114 | 0.2809 | 99.15 | 116.31 | `n_estimators: 150`, `learning_rate: 0.05` |
| **Random Forest (RF)** | 0.1052 | 0.0992 | 111.73 | 130.17 | `n_estimators: 100`, `max_depth: 10` |

*Tree-based ensembles struggled with smoother rolling aggregates and extrapolation across shifting demand regimes, whereas L2-regularized Ridge regression delivered superior generalization.*

---

### 3. Forecasting Horizon Sensitivity Analysis

Model reliability evaluated at varying decision lead times:

| Lead Time Horizon | Days Ahead | $R^2$ Score | RMSE | Practical Operational Use |
|:---:|:---:|:---:|:---:|:---|
| **24 Hours** | 1 Day | **0.9447** | 33.85 | Immediate dock door & shift labor dispatch |
| **72 Hours** | 3 Days | **0.8390** | 57.81 | Secondary carrier & drayage container pickup |
| **168 Hours** | 7 Days | **0.6492** | 85.44 | **Primary booking horizon (charters, warehouse bays)** |
| **336 Hours** | 14 Days | **0.3759** | 114.24 | Strategic monthly inventory staging |

---

### 4. Demand Band Classification (Operational Terciles)

Predictions partitioned into three balanced operational tiers based on training data terciles:
- **Low Band:** $< 5,958.6$ units
- **Normal Band:** $5,958.6 - 6,051.4$ units
- **High Band:** $> 6,051.4$ units

| Metric | Low Band | Normal Band | High Band | Overall / Macro |
|:---|:---:|:---:|:---:|:---:|
| **Precision** | 0.75 | 0.44 | 0.90 | **0.70 (Macro)** |
| **Recall** | 0.78 | 0.62 | 0.79 | **0.73 (Macro)** |
| **F1-Score** | 0.77 | 0.51 | 0.84 | **0.70 (Macro)** |
| **Overall Accuracy** | — | — | — | **75.3%** |

*(Model accuracy of 75.3% handily beats the random 3-class baseline of 33.3% and the majority-class baseline of 69.3%).*

---

## 💼 Business Impact & Decision Engine

A prediction without an operational recommendation is useless. Stage 12 converts forecasts into physical capacity orders:

$$\text{Ships Needed} = \left\lceil \frac{\text{Predicted Demand}}{250} \right\rceil$$

$$\text{Warehouse Space} = \text{Predicted Demand} \times 0.60 \text{ sq ft}$$

$$\text{Labour Required} = \left\lceil \frac{\text{Predicted Demand}}{120} \right\rceil$$

### Weekly Booking Scenario Analysis
When booking 7 days ahead at contract rates versus waiting for the spot market rush window:
- **Ship Premium:** 30% rush surcharge (anchored to spot ocean freight indices)
- **Warehouse Premium:** 15% emergency bay reservation rate
- **Labour Premium:** 13% overtime / short-notice staffing rate

```
=== FREIGHT STATION BOOKING PLAN ===
Current planning demand      : 5,732.4
Forecast demand (7 days out) : 5,827.3 (+1.66%)
------------------------------------------------
Ships to charter             : 24 vessels
Warehouse space to reserve   : 3,497 sq ft
Labourers to roster          : 49 staff
------------------------------------------------
Cost if booked NOW (early)   : $302,564
Cost if booked in the RUSH   : $375,377
SAVING FROM BOOKING EARLY    : $ 72,813
```

---

## 🔍 Model Interpretability & Explainability

Because the champion model is an L2-regularized linear model, feature contributions for any individual forecast decompose **exactly** without surrogate approximations:

$$\hat{y}_i = w_0 + \sum_{j=1}^{p} w_j \cdot x_{ij}$$

```
Top Drivers for Sample Forecast (Prediction: 5827.3 vs Actual: 5717.4):
• planning_demand   (5,732.42)  --> Contribution: -131.8  [Lowers forecast]
• planning_lag_168  (5,833.31)  --> Contribution:  -37.2  [Lowers forecast]
• demand_mean_7d    (5,562.25)  --> Contribution:  -35.4  [Lowers forecast]
• demand_std_7d     (3,722.02)  --> Contribution:  +29.5  [Raises forecast]
• planning_lag_24   (5,695.15)  --> Contribution:  -24.8  [Lowers forecast]
• planning_lag_48   (5,728.99)  --> Contribution:  +15.0  [Raises forecast]
```

This transparency allows logistics managers to distinguish between persistent macroeconomic demand drift and temporary operational volatility.

---

## 📁 Repository Structure

```
Freight-Forecasting-Model--Machine-Learning/
│
├── dynamic_supply_chain_logistics_dataset.csv  # 32,065-row primary telemetry dataset
├── Freight_Forecasting_Model_Simple.ipynb      # Complete 12-stage executable walkthrough
├── requirements.txt                            # Python dependencies
└── README.md                                   # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.8 or higher installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/neel1202/Freight-Forecasting-Model--Machine-Learning.git
cd Freight-Forecasting-Model--Machine-Learning
```

### 3. Create a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch the Notebook
```bash
jupyter notebook Freight_Forecasting_Model_Simple.ipynb
```
Select **Kernel > Restart & Run All** to reproduce all figures, baseline evaluations, grid searches, and booking tables.

---

## ⚠️ Academic & Practical Limitations

1. **Raw Hourly Unpredictability:** Individual hourly demand is not learnable from this dataset ($R^2 \approx 0.02$). All reported model skill is strictly on the 30-day planning average.
2. **Declared Cost Assumptions:** The dataset does not contain proprietary contract rate tables. The spot premiums (30% ship, 15% warehouse, 13% labor) are declared baseline assumptions based on supply chain literature.
3. **Dataset Provenance:** The dataset is a Kaggle public release (CC0). While structured around Southern California logistics telemetry, deployment to a new logistics hub requires localized calibration.
4. **Temporal Drift:** Test demand levels drift higher than the training period. In real-world operations, the model should be retrained weekly via a sliding window.

---


