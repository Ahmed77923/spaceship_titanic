import pandas as pd
from sklearn.metrics import f1_score

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def build_preprocessor():

    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Num_bin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )

    return preprocessor


def get_models():

    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
    }

    return models


def compare_models(X_train, X_test, y_train, y_test):

    preprocessor = build_preprocessor()
    models = get_models()

    results = []

    for name, clf in models.items():

        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('classifier', clf)
        ])

        # train
        pipeline.fit(X_train, y_train)

        # predict
        y_pred = pipeline.predict(X_test)

        # score
        f1 = f1_score(y_test, y_pred)

        print(f"{name} F1: {f1:.4f}")

        results.append((name, f1))

    # sort results
    results = sorted(results, key=lambda x: x[1], reverse=True)

    print("\nBest Model:", results[0][0])
    print("Best F1:", results[0][1])
    
    return results