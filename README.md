# 💳 Credit Card Fraud Detection System

> A machine learning project to identify fraudulent credit card transactions with high accuracy


## 📌 Project Overview

This project was developed as part of my data science learning journey to build a **real-world fraud detection system**. Credit card fraud leads to significant financial losses, and traditional systems often fail to detect complex fraud patterns.

This project demonstrates how machine learning can be used to detect fraudulent transactions effectively.

**Key Achievement**: Achieved high recall and precision in detecting fraudulent transactions.

---

## 🎯 Problem Statement

Credit card transactions are highly imbalanced:
- Fraud cases are extremely rare
- Most transactions are legitimate

This creates a challenge where models may predict all transactions as normal.

**Goal**: Build a model that accurately detects fraud while minimizing false positives.

---

## 📊 Dataset Information

- Transactions include:
  - Time
  - Amount
  - V1–V28 (anonymized features)
- Target variable:
  - `0` → Normal
  - `1` → Fraud

---
flowchart TD

%% =====================
%% START
%% =====================
A([Start]) --> B[Load Credit Card Dataset]

%% =====================
%% DATA PROCESSING
%% =====================
B --> C[Data Preprocessing]
C --> C1[Handle Missing Values]
C --> C2[Feature Scaling]
C --> C3[Split Features & Target]

%% =====================
%% EDA
%% =====================
C3 --> D[Exploratory Data Analysis]
D --> D1[Analyze Fraud vs Normal Distribution]
D --> D2[Visualize Patterns]

%% =====================
%% IMBALANCE HANDLING
%% =====================
D2 --> E[Handle Imbalanced Data]
E --> E1[Apply SMOTE Oversampling]

%% =====================
%% MODEL TRAINING
%% =====================
E1 --> F[Train Machine Learning Models]
F --> F1[Logistic Regression]
F --> F2[Random Forest]
F --> F3[XGBoost]

%% =====================
%% EVALUATION
%% =====================
F1 --> G[Evaluate Models]
F2 --> G
F3 --> G

G --> G1[Precision]
G --> G2[Recall]
G --> G3[F1 Score]
G --> G4[Confusion Matrix]

%% =====================
%% DECISION
%% =====================
G4 --> H{Model Performance Good?}

H -->|No| F
H -->|Yes| I[Select Best Model]

%% =====================
%% DEPLOYMENT
%% =====================
I --> J[Save Trained Model]
J --> K[Deploy for Real-Time Prediction]

%% =====================
%% END
%% =====================
K --> L([End])
---

## ⚙️ Workflow Explanation

### 1. Data Preprocessing
- Clean dataset
- Scale numerical features
- Prepare inputs for model

### 2. Exploratory Data Analysis (EDA)
- Understand fraud distribution
- Identify patterns in transactions

### 3. Handling Imbalanced Data
- SMOTE (oversampling)
- Ensures balanced training dataset

### 4. Model Training
- Logistic Regression
- Random Forest
- XGBoost

### 5. Model Evaluation
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 🛠️ Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- XGBoost
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📈 Implementation Steps

### Load Dataset

```python
import pandas as pd

data = pd.read_csv('creditcard.csv')
```

### Handle Imbalanced Data

```python
from imblearn.over_sampling import SMOTE

X = data.drop('Class', axis=1)
y = data['Class']

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
```

### Train Model

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_resampled, y_resampled)
```

### Evaluate Model

```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_resampled)
print(classification_report(y_resampled, y_pred))
```

---

## 📊 Results

- Successfully detected fraudulent transactions
- Improved recall using SMOTE
- Built an efficient fraud detection pipeline

---

## 🎯 Project Goals

- Achieve high precision and recall (>90%)
- Build a low-latency detection system
- Reduce financial losses

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── predict.py
│
├── models/
│   └── model.pkl
│
├── images/
│   └── flowchart.png
│
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
jupyter notebook
```

---

## 🔮 Future Improvements

- Real-time fraud detection API
- Deep learning models
- Dashboard visualization
- Cloud deployment

---

## 🙋‍♀️ Author

**Mallu Vineela Reddy**  
Data Science Enthusiast

