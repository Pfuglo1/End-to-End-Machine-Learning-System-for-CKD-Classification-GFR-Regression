# 🩺 Capstone Project 9
## End-to-End Machine Learning System for CKD Classification & GFR Regression

---

## 📌 Project Overview

This project builds a complete, end-to-end machine learning pipeline for **Chronic Kidney Disease (CKD)** analysis using a kidney health dataset. The system addresses two prediction tasks:

- **Part A — Classification:** Predict whether a patient has CKD (`CKD_Status`: 0 or 1)
- **Part B — Regression:** Predict the Glomerular Filtration Rate (`GFR`) to assess kidney function severity

---

## 📂 Dataset

| Property | Details |
|---|---|
| **File** | `kidney_dataset (1).csv` |
| **Target (Classification)** | `CKD_Status` (binary: 0 = No CKD, 1 = CKD) |
| **Target (Regression)** | `GFR` (continuous — Glomerular Filtration Rate) |
| **Dropped Column** | `Medication` (removed due to high missing values / irrelevance) |
| **Duplicates** | None found |

### Key Features Used

| Feature | Type | Description |
|---|---|---|
| `Creatinine` | Numerical | Blood creatinine level |
| `GFR` | Numerical | Glomerular Filtration Rate |
| `BUN` | Numerical | Blood Urea Nitrogen |
| `Urine_Output` | Numerical | Volume of urine output |
| `Protein_in_Urine` | Numerical | Protein presence in urine |
| `Water_Intake` | Numerical | Daily water intake |
| `Age` | Numerical | Patient age |
| `Diabetes` | Binary | 0 = No, 1 = Yes |
| `Hypertension` | Binary | 0 = No, 1 = Yes |
| `CKD_Status` | Binary | 0 = No CKD, 1 = CKD (classification target) |

---

## 🛠️ Libraries & Dependencies

```python
pandas
numpy
seaborn
matplotlib
scipy
scikit-learn
xgboost
pickle
warnings
```

Install with:
```bash
pip install pandas numpy seaborn matplotlib scipy scikit-learn xgboost
```

---

## 📊 Workflow

### 1️⃣ Data Loading & Preprocessing

- Loaded dataset from CSV using `pandas`
- Inspected shape, column names, data types, and null values
- Dropped `Medication` column
- Verified zero duplicate rows
- Applied **Yeo-Johnson Power Transformation** (`PowerTransformer`) on all numerical columns to reduce skewness
- Exported cleaned dataset as `dataset_clean.csv`

```python
pt = PowerTransformer(method='yeo-johnson')
df[num_cols] = pt.fit_transform(df[num_cols])
```

---

### 2️⃣ Exploratory Data Analysis (EDA)

#### Class Distribution
- Checked class balance of `CKD_Status` using value counts and a count plot

#### Univariate Analysis
- Separated columns into **binary** (`bin_cols`) and **numerical** (`num_cols`)
- Plotted KDE distributions for all numerical features before and after transformation
- Evaluated skewness and kurtosis

#### Bivariate Analysis
- Box plots of each numerical feature grouped by `CKD_Status`
- Group means of `CKD_Status` by `Diabetes` and `Hypertension`
- Scatter plots:
  - `Creatinine` vs `GFR` (colored by CKD status)
  - `BUN` vs `Protein_in_Urine` (colored by CKD status)

#### Correlation Analysis
- Full correlation matrix
- Bar plot of correlations with `CKD_Status`
- Pairplot colored by `CKD_Status`
- KDE plots for all features split by CKD class

---

## 🧠 Part A — CKD Classification

### Goal
Classify patients as CKD-positive or CKD-negative using clinical features.

### Data Split & Scaling

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

### Models Evaluated

| Model | Accuracy | Precision | Recall | F1-Score | MCC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9950 | 0.9813 | 1.0000 | 0.9906 | 0.9873 |
| Decision Tree Classifier | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest Classifier | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| KNN Classifier | 0.9890 | 0.9599 | 1.0000 | 0.9795 | 0.9724 |
| XGB Classifier | 0.9960 | 0.9850 | 1.0000 | 0.9925 | 0.9898 |

### Best Model: **Random Forest Classifier**

Final model trained on the **top 5 most important features**:

```python
imp_cols = ['Creatinine', 'GFR', 'BUN', 'Urine_Output', 'Protein_in_Urine']
```

#### Final Classification Metrics (on important features)

| Metric | Score |
|---|---|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-Score | 1.0000 |
| MCC | 1.0000 |

### Saved Model
```python
# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Load & Predict
with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

test = [[-0.772036, 0.807323, 0.044457, 0.760896, -0.045423]]
test = scaler.transform(test)
loaded_model.predict(test)
```

---

## 📈 Part B — GFR Regression

### Goal
Predict the GFR value (continuous) to estimate kidney function severity.

### Features Used

```python
df = df[['Creatinine', 'BUN', 'Urine_Output', 'Diabetes', 'Hypertension',
         'Age', 'Protein_in_Urine', 'Water_Intake', 'CKD_Status', 'GFR']]
```

### Data Split & Scaling

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
```

### Models Evaluated

| Model | MAE | R² | MAPE |
|---|---|---|---|
| Linear Regression | 0.111984 | 0.975767 | 0.177301 |
| Decision Tree Regressor | 0.046924 | 0.994912 | 0.136912 |
| Random Forest Regressor | 0.025466 | 0.998480 | 0.064539 |
| KNN Regressor | 0.082000 | 0.983300 | 0.188812 |
| **XGB Regressor** | **0.019700** | **0.999208** | **0.060418** |

> ✅ **Best model by metrics: XGBRegressor** (R² = 0.9992, MAE = 0.0197)
> 🔁 **Final model trained: RandomForestRegressor** (used for feature selection and final pipeline)

### Feature Importance (Top 7 Selected)

Feature importances extracted from `RandomForestRegressor` and top 7 features selected for the final model using:

```python
imp_cols = results.sort_values(by='Importances', ascending=False).head(7)['Features'].values.tolist()
```

### Final Regression Metrics (on top 7 features)

| Metric | Score |
|---|---|
| MAE | ~0.025 |
| R² | ~0.9985 |
| MAPE | ~0.065 |

#### Diagnostics
- **Actual vs Predicted scatter plot** — tightly clustered along diagonal
- **Residuals KDE plot** — centered near 0, approximately normal distribution

### Saved Model
```python
# Save
with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

# Load & Predict
with open('final_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

test = [[0, -0.502194, -1.743049, 1.050941, -0.18237, 1, 0]]
test = scaler.transform(test)
loaded_model.predict(test)
```

---

## 📁 Output Files

| File | Description |
|---|---|
| `dataset_clean.csv` | Cleaned & transformed dataset after preprocessing |
| `model.pkl` | Saved RandomForestClassifier for CKD classification |
| `final_model.pkl` | Saved RandomForestRegressor for GFR prediction |

---

## 🔁 Project Pipeline Summary

```
Raw Dataset (kidney_dataset.csv)
        ↓
Data Cleaning (drop Medication, handle nulls)
        ↓
Feature Transformation (Yeo-Johnson Power Transform)
        ↓
EDA (Univariate → Bivariate → Correlation)
        ↓
        ├── PART A: CKD Classification
        │       ├── Train/Test Split (stratified, 80/20)
        │       ├── StandardScaler
        │       ├── Benchmark 5 Classifiers
        │       ├── Select Best: RandomForestClassifier
        │       ├── Feature Selection (Top 5)
        │       └── Save model.pkl
        │
        └── PART B: GFR Regression
                ├── Train/Test Split (80/20)
                ├── StandardScaler
                ├── Benchmark 5 Regressors
                ├── Best by metrics: XGBRegressor
                ├── Final model: RandomForestRegressor
                ├── Feature Selection (Top 7)
                └── Save final_model.pkl
```

---

## ✅ Key Takeaways

- `Creatinine`, `GFR`, and `BUN` are the most important features for CKD classification
- Random Forest achieved **perfect classification performance** on the test set
- XGBRegressor achieved the best GFR regression metrics (R² = 0.9992)
- Power transformation significantly reduced skewness in numerical features
- Both models are serialized with `pickle` and ready for deployment
