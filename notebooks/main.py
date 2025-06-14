import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent))


import pandas as pd
import mlflow
from typing import Optional
from src.models import train_and_log_logistic_regression, train_and_log_random_forest
from src.metrics import get_metrics
from src.preprocessing import create_preprocessor
from src.config import LOG_REG_PARAMS, RF_PARAMS, EXPERIMENT_NAME, TRACKING_URI, NUMERIC_FEATURES, CATEGORICAL_FEATURES
from sklearn.model_selection import train_test_split


data_path =  "dataset/diabetes.csv"

def read_data(filepath) -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(filepath)
        print("Data loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"File not found at {filepath}. Check path structure.")
        return None

def run_training():
    data = read_data(data_path)
    if data is not None:
        features = [col for col in data.columns][1:9]
        labels = [col for col in data.columns][9]

        X, y = data[features].values, data[labels].values

        for n in range(0,4):
            print("Patient", str(n+1), "\n  Features:",list(X[n]), "\n  Label:", y[n])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        print ('Training cases: %d\nTest cases: %d' % (X_train.shape[0], X_test.shape[0]))
        

        preprocessor = create_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        log_model  = train_and_log_logistic_regression(X_train, X_test, y_train, y_test, preprocessor, 
                                                    data_path=data_path, **LOG_REG_PARAMS)

        rf_model = train_and_log_random_forest(X_train, X_test, y_train, y_test, preprocessor, 
                                            data_path=data_path, **RF_PARAMS)
        
        print("Training models completed and models logged to MLflow.")




if __name__ == "__main__":
    run_training()