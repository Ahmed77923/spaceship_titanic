from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

def build_pipeline():

    cat_cols = ['HomePlanet', 'Destination', 'Deck', 'Side', 'Num_bin']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('classifier',XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        ))
    ])
    print('='* 50   )
    print('\nPipeline building completed successfully!\n')
    print('='* 50)
    return model

# if __name__ == "__main__":
#     model = build_pipeline()
#     print(model)