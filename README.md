# 🚀 Spaceship Titanic - Machine Learning Pipeline

## 📌 Project Overview

This project is a complete end-to-end machine learning pipeline built to solve the **Spaceship Titanic Kaggle competition**.

The goal is to predict whether a passenger was transported to another dimension based on various features such as spending behavior, cabin location, and personal information.

---

## 🧠 Problem Type

* Binary Classification
* Evaluation Metric: **F1 Score**

---

## 📂 Project Structure

```
Pipeline/
├── config/
|   |__config.py
|   |__ __init__.py
|
|
├── data/
|   |__data.csv
│   ├── load_data.py
|   |__feature_engineering.py
│   ├── preprocessing.py
│   ├── split_data.py
|
|
├── models/
│   ├── train_model.py
│   ├── compare_models.py
|
|___utils/
|   |__ML.flow
|
|___deployment/
|   |__app.py
├── main.py
├── submit_xgb.py
└── README.md
```

---

## 🔧 Workflow

### 1. Data Loading

* Load training and test datasets
* Ensure consistent structure

---

### 2. Data Preprocessing & Feature Engineering

Key transformations:

* Split `PassengerId` → `Group`, `Group_size`
* Extract `Deck`, `Num`, `Side` from `Cabin`
* Handle missing values:

  * Numerical → median / 0
  * Categorical → "Unknown"
* Feature creation:

  * `Total_spending`
  * `No_spending`
  * `Is_alone`
* Convert boolean features to integers

---

### 3. Preprocessing Pipeline

* `ColumnTransformer` with:

  * `OneHotEncoder` for categorical features
* Numerical features passed through

---

### 4. Model Training

Models tested:

* Random Forest
* Gradient Boosting
* XGBoost
* Ensemble (Voting Classifier)

---

### 5. Model Comparison

A custom comparison system was built to:

* Train multiple models
* Evaluate using F1 score
* Select the best performing model

---

### 6. Final Model

* XGBoost was used for final submission
* Tuned parameters for better performance

---

### 7. Submission

* Train on full dataset
* Predict on test set
* Generate submission file:

```
```

---

## 📊 Results

| Model            | F1 Score |
| ---------------- | -------- |
| RandomForest     | ~0.79    |
| GradientBoosting | ~0.80    |
| XGBoost          | ~0.814   |
| Ensemble         | ~0.80+   |

---

## ⚠️ Key Learnings

* Feature engineering has a bigger impact than model choice
* Not all tuning improves performance
* Model comparison is critical
* Ensemble models must be carefully designed
* Always validate with proper splits

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn xgboost
```

### 2. Train & evaluate

```bash
python main.py
```

### 3. Create submission

```bash
python submit_xgb.py
```

---

## 🧠 Future Improvements

* Hyperparameter optimization (GridSearch / Optuna)
* Better feature interactions
* Cross-validation
* Advanced ensemble techniques

---

## 👨‍💻 Author

Machine Learning project built with a focus on:

* Clean pipeline design
* Experimentation
* Performance optimization

---

## 🏁 Final Note

This project demonstrates the transition from:

> Notebook-based experimentation → Structured ML pipeline

---
