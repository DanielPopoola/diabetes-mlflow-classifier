# Diabetes Binary Classification with MLflow Tracking

A machine learning project for binary classification of diabetes using scikit-learn with comprehensive MLflow experiment tracking and model management.

## 🎯 Project Overview

This project implements and compares two machine learning models for diabetes prediction:
- **Logistic Regression** with L1/L2 regularization
- **Random Forest Classifier** with ensemble learning

All experiments are tracked using MLflow with a dedicated tracking server, enabling model comparison, versioning, and deployment preparation.

## 🛠️ Features

- **Automated MLflow Tracking**: Complete experiment logging with parameters, metrics, and artifacts
- **Model Comparison**: Side-by-side comparison of multiple algorithms
- **Data Versioning**: File hash tracking for dataset reproducibility  
- **Model Registry**: Automatic model registration for deployment
- **AWS Lambda Ready**: Models saved in deployment-ready format
- **Reusable Functions**: Clean, modular code architecture

## 📋 Requirements

```bash
pip install mlflow==2.16.2  # Recommended stable version
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Set Up MLflow Tracking Server

```bash
# Create directories
mkdir mlruns_server
mkdir mlruns_server/artifacts

# Start MLflow server
mlflow server \
  --backend-store-uri sqlite:///mlruns_server/mlflow.db \
  --default-artifact-root ./mlruns_server/artifacts \
  --host 127.0.0.1 \
  --port 5000
```

### 2. Run the Experiment

```python
import mlflow

# Set up experiment
mlflow.set_experiment("Diabetes Binary Classification Comparison")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")

# Train and log models
log_model = train_and_log_logistic_regression(
    X_train, X_test, y_train, y_test, preprocessor,
    data_path="your_dataset.csv",
    C=100, solver='liblinear'
)

rf_model = train_and_log_random_forest(
    X_train, X_test, y_train, y_test, preprocessor,
    data_path="your_dataset.csv", 
    n_estimators=100
)
```

### 3. View Results

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) to explore your experiments in the MLflow UI.

## 📊 Tracked Metrics

Each model logs the following performance metrics:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Positive prediction accuracy
- **Recall**: True positive detection rate  
- **AUC**: Area under the ROC curve
- **Entropy**: Log loss for probability calibration

## 🔧 Model Pipeline

### Data Preprocessing
- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: OneHotEncoder with unknown handling
- **Pipeline Integration**: Seamless preprocessing with model training

### Feature Configuration
```python
numeric_features = [0, 1, 2, 3, 4, 5, 6]  # Adjust based on your dataset
categorical_features = [7]                  # Adjust based on your dataset
```

## 📁 Project Structure

```
diabetes-classification/
├── mlruns_server/
│   ├── artifacts/          # Model artifacts and files
│   └── mlflow.db          # SQLite tracking database
├── notebooks/             # Jupyter notebooks (if any)
├── src/
│   ├── models.py         # Training functions
│   ├── metrics.py        # Metrics calculation
│   └── preprocessing.py  # Data preprocessing
├── data/                 # Dataset files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎛️ Function Reference

### Core Training Functions

```python
train_and_log_logistic_regression(X_train, X_test, y_train, y_test, 
                                 preprocessor, data_path=None, **params)
```

```python
train_and_log_random_forest(X_train, X_test, y_train, y_test,
                           preprocessor, data_path=None, **params)
```

### Metrics Calculation

```python
get_metrics(y_test, y_pred, y_pred_prob)
# Returns: {'accuracy', 'precision', 'recall', 'entropy', 'auc'}
```

### Data Versioning

```python
get_file_hash(filepath)
# Returns: MD5 hash for dataset reproducibility
```

## 🚀 Model Deployment

Models are automatically saved with:
- **MLmodel format**: For easy loading and serving
- **Input/Output signatures**: Type safety for production
- **Input examples**: Sample data for testing
- **Model registry**: Centralized model management

### Loading a Registered Model

```python
import mlflow.sklearn

# Load latest version
model = mlflow.sklearn.load_model("models:/LogisticRegressionModel/latest")

# Make predictions
predictions = model.predict(new_data)
```

## 📈 Experiment Management

### Comparing Models
- Use the MLflow UI to compare metrics across runs
- Filter experiments by tags and parameters
- Download models and artifacts directly

### Model Versioning
- Models are automatically registered with timestamps
- Track model lineage and data dependencies
- Easy rollback to previous versions

## 🐛 Troubleshooting

### MLflow Version Issues
If you encounter `run_uuid` errors:
```bash
pip uninstall mlflow
pip install mlflow==2.16.2
```

### Server Connection Issues
- Ensure MLflow server is running on port 5000
- Check firewall/network restrictions
- Verify tracking URI configuration

### Environment Setup
- Use virtual environments to avoid conflicts
- Ensure all dependencies are installed
- Check Python version compatibility

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Useful Links

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [AWS Lambda Deployment Guide](https://docs.aws.amazon.com/lambda/)

---

**Note**: This project is designed for educational purposes and demonstrates best practices in ML experiment tracking and model management.