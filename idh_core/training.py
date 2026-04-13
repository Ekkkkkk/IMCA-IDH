from dataclasses import dataclass
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

from .models import DEVICE, LSTMClassifier, LSTMWithAttention, GRUClassifier, BiLSTMClassifier, TransformerClassifier


@dataclass
class ExperimentConfig:
    channels: int
    window_size: int
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epoch: int = 30
    emb_dim: int = 64
    rank_loss_weight: float = 1.0


def calculate_binary_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score).astype(float)
    auc = 0.5 if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_score)
    pr_auc = 0.5 if len(np.unique(y_true)) < 2 else average_precision_score(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc": auc,
        "pr_auc": pr_auc,
    }


def get_static_model(model_name):
    if model_name == "Logistic":
        return LogisticRegression(penalty="l1", C=0.5, solver="liblinear", class_weight="balanced", max_iter=10000)
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=66, n_estimators=100)
    if model_name == "SVC":
        return SVC(C=0.2, kernel="rbf", class_weight="balanced", random_state=2, probability=True)
    raise ValueError(f"Unsupported static model: {model_name}")


def get_sequence_model(config, model_name):
    hidden_size = 5 * config.channels
    if model_name == "LSTM":
        return LSTMClassifier(config.channels, hidden_size, 2)
    if model_name == "LSTMAtten":
        return LSTMWithAttention(config.channels, hidden_size, 2)
    if model_name == "GRU":
        return GRUClassifier(config.channels, hidden_size, 2)
    if model_name == "BiLSTM":
        return BiLSTMClassifier(config.channels, config.emb_dim, 2)
    if model_name == "Transformer":
        return TransformerClassifier(config.channels, hidden_size, 2)
    raise ValueError(f"Unsupported sequence model: {model_name}")


def _predict_static_probabilities(model, X):
    return model.predict_proba(X)[:, 1]


def fit_static_model(model_name, train_X, train_y, eval_X=None, eval_y=None):
    model = get_static_model(model_name)
    model.fit(train_X, train_y)
    outputs = {"model": model}
    if eval_X is not None:
        eval_prob = _predict_static_probabilities(model, eval_X)
        eval_pred = (eval_prob >= 0.5).astype(int)
        outputs["eval_probabilities"] = eval_prob
        outputs["eval_metrics"] = calculate_binary_metrics(eval_y, eval_pred, eval_prob)
    return outputs


def _get_stratified_splitter(y, desired_splits=5):
    y = np.asarray(y).astype(int)
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) < 2:
        return None
    n_splits = min(desired_splits, int(counts.min()))
    return None if n_splits < 2 else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def generate_static_oof_probabilities(model_name, train_X, train_y):
    train_y = np.asarray(train_y).astype(int)
    splitter = _get_stratified_splitter(train_y)
    if splitter is None:
        return fit_static_model(model_name, train_X, train_y, train_X, train_y)["eval_probabilities"]
    oof = np.zeros(len(train_y), dtype=float)
    for fit_idx, holdout_idx in splitter.split(train_X, train_y):
        fold_model = get_static_model(model_name)
        fold_model.fit(train_X[fit_idx], train_y[fit_idx])
        oof[holdout_idx] = _predict_static_probabilities(fold_model, train_X[holdout_idx])
    return oof


def pairwise_ranking_surrogate_loss(probabilities, labels):
    labels = labels.long()
    positive_scores = probabilities[labels == 1]
    negative_scores = probabilities[labels == 0]

    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return torch.tensor(0.0, device=probabilities.device)

    pairwise_differences = negative_scores.unsqueeze(0) - positive_scores.unsqueeze(1)
    return torch.sigmoid(pairwise_differences).mean()


def predict_sequence_probabilities(model, data_X, batch_size):
    if not isinstance(data_X, torch.Tensor):
        data_X = torch.tensor(data_X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data_X), batch_size=batch_size, shuffle=False, drop_last=False)
    model = model.to(DEVICE)
    model.eval()
    outputs = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(DEVICE)
            predictions = model(inputs)
            outputs.append(predictions.softmax(dim=1)[:, 1].detach().cpu().numpy())
    return np.concatenate(outputs, axis=0) if outputs else np.array([], dtype=float)


def train_sequence_model(model, train_X, train_y, val_X, val_y, config):
    if not isinstance(train_X, torch.Tensor):
        train_X = torch.tensor(train_X, dtype=torch.float32)
    if not isinstance(val_X, torch.Tensor):
        val_X = torch.tensor(val_X, dtype=torch.float32)
    train_y = torch.nn.functional.one_hot(torch.tensor(train_y, dtype=torch.long), num_classes=2).float()
    val_y_onehot = torch.nn.functional.one_hot(torch.tensor(val_y, dtype=torch.long), num_classes=2).float()

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(val_X, val_y_onehot), batch_size=config.batch_size, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8], device=DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    model = model.to(DEVICE)
    best_state = copy.deepcopy(model.state_dict())
    best_val_auc = -1.0

    for _ in range(config.max_epoch):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            ce_loss = criterion(outputs, labels)
            ranking_loss = pairwise_ranking_surrogate_loss(outputs.softmax(dim=1)[:, 1], labels.argmax(1))
            loss = ce_loss + config.rank_loss_weight * ranking_loss
            loss.backward()
            optimizer.step()

        model.eval()
        val_probabilities = []
        val_labels = []
        val_loss_sum = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(inputs)
                val_probabilities.append(outputs.softmax(dim=1)[:, 1].cpu().numpy())
                val_labels.append(labels.argmax(1).cpu().numpy())
                val_ranking_loss = pairwise_ranking_surrogate_loss(outputs.softmax(dim=1)[:, 1], labels.argmax(1))
                val_loss_sum += (criterion(outputs, labels) + config.rank_loss_weight * val_ranking_loss).item()
        val_probabilities = np.concatenate(val_probabilities, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_auc = 0.5 if len(np.unique(val_labels)) < 2 else roc_auc_score(val_labels, val_probabilities)
        scheduler.step(val_loss_sum / max(len(val_probabilities), 1))
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_val_auc


def generate_sequence_oof_probabilities(config, model_name, train_X, train_y):
    train_y = np.asarray(train_y).astype(int)
    splitter = _get_stratified_splitter(train_y)
    if splitter is None:
        model = get_sequence_model(config, model_name)
        model, _ = train_sequence_model(model, train_X, train_y, train_X, train_y, config)
        return predict_sequence_probabilities(model, train_X, config.batch_size)

    oof = np.zeros(len(train_y), dtype=float)
    for fit_idx, holdout_idx in splitter.split(np.zeros(len(train_y)), train_y):
        model = get_sequence_model(config, model_name)
        model, _ = train_sequence_model(model, train_X[fit_idx], train_y[fit_idx], train_X[holdout_idx], train_y[holdout_idx], config)
        oof[holdout_idx] = predict_sequence_probabilities(model, train_X[holdout_idx], config.batch_size)
    return oof
