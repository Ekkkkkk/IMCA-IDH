import numpy as np
from sklearn.linear_model import LogisticRegression

from .training import calculate_binary_metrics


def fit_logistic_meta_learner(train_meta_features, train_labels, test_meta_features=None, test_labels=None):
    model = LogisticRegression(random_state=42)
    model.fit(train_meta_features, train_labels)
    outputs = {"model": model}

    if test_meta_features is not None:
        test_probabilities = model.predict_proba(test_meta_features)[:, 1]
        test_predictions = (test_probabilities >= 0.5).astype(int)
        outputs["test_probabilities"] = test_probabilities
        outputs["test_predictions"] = test_predictions
        outputs["test_metrics"] = calculate_binary_metrics(test_labels, test_predictions, test_probabilities)
    return outputs


def build_meta_features(static_scores, sequence_scores):
    static_scores = np.asarray(static_scores).reshape(-1)
    sequence_scores = np.asarray(sequence_scores).reshape(-1)
    return np.column_stack([sequence_scores, static_scores])
