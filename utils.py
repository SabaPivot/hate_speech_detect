from sklearn.metrics import accuracy_score, f1_score
from typing import Dict

def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics for model predictions.
    
    Args:
        pred: Prediction object containing label_ids and predictions
        
    Returns:
        Dictionary containing accuracy and F1 scores
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="micro")
    }

    return metrics