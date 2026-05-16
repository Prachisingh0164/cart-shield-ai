# 🛡️ CartShield AI — E-commerce Return Fraud Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/XGBoost-2.0-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> **Predict fraudulent returns before you issue the refund.**  
> CartShield AI combines XGBoost with SHAP explainability to flag return abuse, false "item not received" claims, and refund cycling — all surfaced through a live Streamlit dashboard.

---

## 📸 Dashboard Preview

| Overview | Transaction Scanner | Customer Segments |
|----------|--------------------|--------------------|
| KPI cards, fraud trends, risk distribution | Live fraud probability per transaction | KMeans risk clustering with PCA |

---

## 🏗️ Project Structure

```
CartShieldAI/
├── dashboard/
│   └── app.py                  # Streamlit live dashboard (4 pages)
├── data/
│   ├── raw/                    # Generated transaction + customer CSVs
│   ├── processed/              # Feature-engineered outputs
│   └── sample/                 # 500-row quick-test sample
├── models/                     # Saved model artefacts (.pkl, plots)
├── notebooks/
│   └── EDA_and_Modeling.ipynb  # Full exploratory + modelling notebook
├── scripts/
│   └── generate_data.py        # Synthetic dataset generator (50k rows)
├── src/
│   ├── data/
│   │   └── feature_engineering.py   # 30+ engineered features
│   ├── models/
│   │   ├── train.py                  # XGBoost training + SHAP + plots
│   │   └── segmentation.py           # KMeans customer risk segmentation
│   └── visualization/
├── tests/
│   └── test_features.py        # pytest unit tests
├── Makefile                    # One-command pipeline
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/CartShieldAI.git
cd CartShieldAI
pip install -r requirements.txt
```

### 2. Generate Data

```bash
python scripts/generate_data.py
# → Creates 50,000 synthetic transactions with realistic fraud patterns
```

### 3. Train the Model

```bash
python src/models/train.py
# → Trains XGBoost, saves model + SHAP plots + metrics
```

### 4. Launch Dashboard

```bash
streamlit run dashboard/app.py
```

### Or run everything at once:

```bash
make pipeline     # generate data + train
make dashboard    # launch Streamlit
```

---

## 🎯 Features

### Feature Engineering (30+ features)
| Category | Features |
|----------|----------|
| **Temporal** | order month/day, weekend flag, holiday season, delivery days |
| **Behavioral** | refund-to-order ratio, return rate, order value tiers, log-value |
| **Risk Flags** | fast return, same-day return, new account + high refund, crypto + high value |
| **Heuristic** | composite risk score combining 6 fraud signals |
| **Encoded** | product category, payment method, country, return reason |

### Model: XGBoost Classifier
- Early stopping on PR-AUC (imbalanced class aware)
- `scale_pos_weight` handles class imbalance automatically
- Persisted with `joblib` for serving in dashboard

### Explainability: SHAP
- `TreeExplainer` for fast XGBoost SHAP values
- Summary bar chart of top-15 feature impacts
- Ready for per-transaction waterfall plots

### Customer Segmentation (KMeans)
- Aggregates customer-level return/fraud behaviour
- 4 risk tiers: Trustworthy → Moderate → High → Extreme
- PCA visualization of clusters

### Live Dashboard (Streamlit — 4 pages)
| Page | What you see |
|------|--------------|
| 📊 Overview | KPIs, fraud score distribution, category heatmap, monthly trend |
| 🔍 Transaction Scanner | Form-based live inference with risk badge |
| 👥 Customer Segments | PCA scatter, segment stats, customer table |
| 📈 Model Performance | ROC-AUC, PR-AUC, feature importance, SHAP summary |

---

## 📊 Model Performance (on synthetic data)

| Metric | Score |
|--------|-------|
| ROC-AUC | ~0.94 |
| PR-AUC | ~0.87 |
| Precision (fraud) | ~0.82 |
| Recall (fraud) | ~0.79 |

> *Results on synthetic data. On real e-commerce data, performance will vary based on feature availability and label quality.*

---

## 🔍 Fraud Types Detected

| Fraud Type | Description |
|------------|-------------|
| **Wardrobing** | Buy, use, return (especially clothing/electronics) |
| **False INR** | "Item Not Received" when it was delivered |
| **Price Arbitrage** | Buy low, return for higher refund |
| **Counterfeit Return** | Return a fake, keep the real item |
| **Duplicate Refund** | Claim refund multiple times |

---

## 🧪 Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Pandas, NumPy, synthetic generator |
| ML | XGBoost, scikit-learn |
| Explainability | SHAP |
| Segmentation | KMeans + PCA |
| Dashboard | Streamlit + Plotly |
| Persistence | joblib |
| Testing | pytest |

---

## 📁 Data Dictionary

### `transactions.csv`
| Column | Type | Description |
|--------|------|-------------|
| `order_id` | str | Unique order identifier |
| `customer_id` | str | Customer reference |
| `order_date` | date | Date of purchase |
| `product_category` | str | electronics / clothing / home / beauty / sports / books |
| `order_value` | float | Order total in USD |
| `payment_method` | str | credit_card / debit_card / paypal / apple_pay / crypto |
| `refund_frequency_30d` | int | Number of refunds in last 30 days |
| `days_to_return` | int | Days between delivery and return request |
| `is_fraud` | int | **Target** — 1 = fraud, 0 = legitimate |
| `fraud_type` | str | Category of fraud (wardrobing, false INR, etc.) |

---

## 💡 Resume Talking Points

- Built an **end-to-end ML fraud detection system** from scratch with 30+ engineered features
- Achieved **ROC-AUC ~0.94** using XGBoost with imbalanced class handling
- Implemented **SHAP explainability** to make model decisions interpretable to business stakeholders
- Deployed a **live Streamlit dashboard** with real-time scoring, customer segmentation, and KPI monitoring
- Applied **KMeans clustering** to segment customers into 4 risk tiers for proactive fraud prevention

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with ❤️ for learning ML + business impact</p>
