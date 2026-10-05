from data.split_data import split_data
from data.load_data import load_data
from data.preprocessing import preprocess_data

from models.compare_models import compare_models


# load + preprocess
df = load_data('data/train.csv')
df = preprocess_data(df)

# split
X_train, X_test, y_train, y_test = split_data(df)

# 🔥 compare models
results = compare_models(X_train, X_test, y_train, y_test)