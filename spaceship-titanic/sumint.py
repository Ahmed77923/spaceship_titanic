import pandas as pd

from data.load_data import load_data
from data.preprocessing import preprocess_data

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier


# =========================
# Load data
# =========================
train = load_data('data/train.csv')
test = load_data('data/test.csv')

# ⚠️ احفظ PassengerId قبل ما يضيع
test_ids = test["PassengerId"]


# =========================
# Preprocess
# =========================
train = preprocess_data(train)
test = preprocess_data(test)


# =========================
# Split train
# =========================
X = train.drop("Transported", axis=1)
y = train["Transported"]

# ⚠️ مهم: test ما فيه target
X_test = test.copy()


# =========================
# Preprocessor
# =========================
cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Num_bin']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='passthrough'
)


# =========================
# Model (XGBoost)
# =========================
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    ))
])


# =========================
# Train FULL data
# =========================
model.fit(X, y)


# =========================
# Predict
# =========================
preds = model.predict(X_test)


# =========================
# Submission
# =========================
submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": preds.astype(bool)
})

submission.to_csv("submission_xgb.csv", index=False)

print("✅ submission_xgb.csv created successfully!")