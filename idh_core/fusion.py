import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC

from .training import calculate_binary_metrics


def fit_logistic_meta_learner(train_meta_features, train_labels, test_meta_features=None, test_labels=None):
    meta_model = LogisticRegression()
    base_models = [
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('lg', LogisticRegression(random_state=42)),
        ('svc', SVC(probability=True, kernel='rbf',random_state=42))
    ]
    model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    )
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
