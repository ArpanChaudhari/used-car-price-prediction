# 🚗 AUTOVAL — Used Car Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

**A machine learning web app that predicts used car prices in the Indian market using a trained Random Forest model.**

</div>

---

## 📸 Preview

> Dark-themed, premium UI built with Streamlit — featuring real ML predictions, confidence intervals, and key factor analysis.

---

## 🧠 Model Overview

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Estimators | 100 trees |
| Training Samples | 9,535 |
| R² Score | **0.9311** |
| MAE (log scale) | 0.127 |
| MAE (rupees) | ≈ ₹1,66,028 |
| Target Variable | `log(Price in ₹)` |
| Brand Coverage | 39 brands |
| Model Coverage | 400+ models |
| Market | 🇮🇳 India (INR) |

The model was trained on real Indian used car listings. The target variable is **log-transformed** price (in rupees) to stabilize variance and normalize skewed distributions. Predictions are converted back via `exp()`.

---

## ⚙️ Feature Engineering

| Feature | Type | Encoding |
|---|---|---|
| `Age` | Numeric | `2024 - Year` |
| `kmDriven` | Numeric | Raw (cleaned from string) |
| `Transmission` | Categorical | Manual=0, Automatic=1 |
| `Owner` | Categorical | first=0, second=1 |
| `FuelType` | Categorical | Diesel=0, Petrol=1, Hybrid/CNG=2 |
| `Brand_enc` | Frequency | `Brand.value_counts(normalize=True)` |
| `Model_enc` | Frequency | `model.value_counts(normalize=True)` |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ArpanChaudhari/used-car-price-prediction.git
cd used-car-price-prediction

# 2. Install dependencies
pip install streamlit scikit-learn pandas numpy joblib

# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
used-car-price-prediction/
│
├── app.py                    # Streamlit web application
├── Car Prediction.ipynb      # Jupyter notebook — EDA & model training
├── car_price_model.joblib    # Trained RandomForestRegressor model
├── expected_columns.joblib   # Feature column names
├── used_car_dataset.csv      # Raw dataset (9,582 listings)
└── README.md
```

---

## 🖥️ App Features

- **Brand & Model selector** — 39 brands, 400+ models loaded from real data
- **Year slider** — 1990 to 2024
- **Mileage slider** — up to 5,00,000 km
- **Fuel type pills** — Petrol / Diesel / Hybrid/CNG
- **Transmission pills** — Manual / Automatic
- **Ownership history** — 1st Owner / 2nd Owner
- **Valuation output** — predicted price in ₹ Lakhs with:
  - Exact rupee equivalent
  - Price range (±1 std of log-predictions)
  - Real confidence score (from tree variance)
  - Key factor breakdown

---

## 📊 Dataset

- **Source:** Indian used car listings scraped from online platforms
- **Size:** 9,582 rows × 11 columns
- **Brands:** 39 (Maruti Suzuki, Hyundai, Honda, Toyota, BMW, Audi, Mercedes-Benz, etc.)
- **Price range:** ₹15,000 — ₹4.25 Crore

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit + Custom CSS (Orbitron, Rajdhani, Share Tech Mono fonts)
- **ML:** scikit-learn `RandomForestRegressor`
- **Data:** pandas, numpy
- **Serialization:** joblib

---

## 👤 Author

**Arpan Chaudhari**
- GitHub: [@ArpanChaudhari](https://github.com/ArpanChaudhari)
- Email: arpanchaudhari810@gmail.com

---

<div align="center">
  Made with ❤️ and Python
</div>
