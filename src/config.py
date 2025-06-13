LOG_REG_PARAMS = {
    'C': 100,
    'solver': 'liblinear'
}

RF_PARAMS = {
    'n_estimators': 100
}

# MLflow configuration
EXPERIMENT_NAME = "Diabetes Binary Classification Comparison"
TRACKING_URI = "http://127.0.0.1:5000/"

# Feature configuration
NUMERIC_FEATURES = [0, 1, 2, 3, 4, 5, 6]
CATEGORICAL_FEATURES = [7]