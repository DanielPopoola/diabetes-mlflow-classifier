from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score, log_loss
)


def get_metrics(y_test, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    entropy = log_loss(y_test, y_pred_prob)
    auc_score = str(roc_auc_score(y_test, y_pred_prob))
    return {'accuracy': round(float(accuracy), 2), 
            'precision': round(float(precision), 2), 
            'recall': round(float(recall), 2), 
            'entropy': round(float(entropy), 2),
            'auc': auc_score
        }