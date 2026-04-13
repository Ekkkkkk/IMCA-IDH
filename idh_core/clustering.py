import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS


def gower_distance_matrix(X, categorical_indices=None):
    X = np.asarray(X, dtype=float)
    n_samples, n_features = X.shape
    categorical_mask = np.zeros(n_features, dtype=bool)
    if categorical_indices is not None:
        categorical_mask[categorical_indices] = True

    ranges = np.ptp(X[:, ~categorical_mask], axis=0) if np.any(~categorical_mask) else np.array([])
    ranges = np.where(ranges == 0, 1.0, ranges)
    distance_matrix = np.zeros((n_samples, n_samples), dtype=float)

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances = []
            cont_idx = 0
            for feature_idx in range(n_features):
                if categorical_mask[feature_idx]:
                    distances.append(0.0 if X[i, feature_idx] == X[j, feature_idx] else 1.0)
                else:
                    distances.append(abs(X[i, feature_idx] - X[j, feature_idx]) / ranges[cont_idx])
                    cont_idx += 1
            pair_distance = float(np.mean(distances))
            distance_matrix[i, j] = pair_distance
            distance_matrix[j, i] = pair_distance
    return distance_matrix


def _assign_to_medoids(distance_matrix, medoid_indices):
    return np.argmin(distance_matrix[:, medoid_indices], axis=1)


def pam_cluster(distance_matrix, n_clusters, max_iter=100):
    n_samples = distance_matrix.shape[0]
    total_distances = distance_matrix.sum(axis=1)
    medoid_indices = [int(np.argmin(total_distances))]
    current_min_distances = distance_matrix[:, medoid_indices[0]]

    while len(medoid_indices) < n_clusters:
        remaining = np.setdiff1d(np.arange(n_samples), np.array(medoid_indices), assume_unique=True)
        candidate_distances = distance_matrix[:, remaining]
        improved_distances = np.minimum(current_min_distances[:, None], candidate_distances)
        candidate_costs = improved_distances.sum(axis=0)
        best_candidate = int(remaining[np.argmin(candidate_costs)])
        medoid_indices.append(best_candidate)
        current_min_distances = np.minimum(current_min_distances, distance_matrix[:, best_candidate])

    medoid_indices = np.array(medoid_indices, dtype=int)
    current_cost = distance_matrix[:, medoid_indices].min(axis=1).sum()

    for _ in range(max_iter):
        best_swap = None
        best_cost = current_cost
        non_medoids = np.setdiff1d(np.arange(n_samples), medoid_indices, assume_unique=True)
        for medoid_pos in range(len(medoid_indices)):
            for candidate in non_medoids:
                trial_medoids = medoid_indices.copy()
                trial_medoids[medoid_pos] = candidate
                trial_cost = distance_matrix[:, trial_medoids].min(axis=1).sum()
                if trial_cost < best_cost:
                    best_cost = trial_cost
                    best_swap = trial_medoids
        if best_swap is None:
            break
        medoid_indices = best_swap
        current_cost = best_cost

    labels = _assign_to_medoids(distance_matrix, medoid_indices)
    return labels, medoid_indices


def fit_gower_pam(X_train, n_clusters, categorical_indices=None):
    distance_matrix = gower_distance_matrix(X_train, categorical_indices=categorical_indices)
    labels, medoid_indices = pam_cluster(distance_matrix, n_clusters)
    medoid_points = np.asarray(X_train, dtype=float)[medoid_indices]
    return {
        "labels": labels,
        "medoid_indices": medoid_indices,
        "medoid_points": medoid_points,
        "distance_matrix": distance_matrix,
    }


def assign_to_reference_medoids(X, medoid_points, categorical_indices=None):
    X = np.asarray(X, dtype=float)
    medoid_points = np.asarray(medoid_points, dtype=float)
    n_features = X.shape[1]
    categorical_mask = np.zeros(n_features, dtype=bool)
    if categorical_indices is not None:
        categorical_mask[categorical_indices] = True
    ranges = np.ptp(medoid_points[:, ~categorical_mask], axis=0) if np.any(~categorical_mask) else np.array([])
    ranges = np.where(ranges == 0, 1.0, ranges)

    distances = np.zeros((X.shape[0], medoid_points.shape[0]), dtype=float)
    for i in range(X.shape[0]):
        for j in range(medoid_points.shape[0]):
            feature_distances = []
            cont_idx = 0
            for feature_idx in range(n_features):
                if categorical_mask[feature_idx]:
                    feature_distances.append(0.0 if X[i, feature_idx] == medoid_points[j, feature_idx] else 1.0)
                else:
                    feature_distances.append(abs(X[i, feature_idx] - medoid_points[j, feature_idx]) / ranges[cont_idx])
                    cont_idx += 1
            distances[i, j] = float(np.mean(feature_distances))
    return np.argmin(distances, axis=1)


def plot_mds_by_split(X_train, X_val, X_test, y_train, y_val, y_test, output_path):
    all_x = np.vstack([X_train, X_val, X_test])
    all_labels = np.concatenate([y_train, y_val, y_test])
    split_labels = ["Train"] * len(X_train) + ["Validation"] * len(X_val) + ["Test"] * len(X_test)
    distance_matrix = gower_distance_matrix(all_x, categorical_indices=[0])
    try:
        embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=42, normalized_stress="auto").fit_transform(distance_matrix)
    except TypeError:
        embedding = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(distance_matrix)

    palette = sns.color_palette("tab10", n_colors=len(np.unique(all_labels)))
    marker_map = {"Train": "o", "Validation": "^", "Test": "s"}
    plt.figure(figsize=(10, 8))
    for split_name, marker in marker_map.items():
        split_mask = np.array(split_labels) == split_name
        for cluster_id in np.unique(all_labels):
            mask = split_mask & (all_labels == cluster_id)
            if np.any(mask):
                plt.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    label=f"{split_name}-C{int(cluster_id) + 1}",
                    color=palette[int(cluster_id) % len(palette)],
                    marker=marker,
                    alpha=0.75,
                    s=50,
                )
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.title("Train-fit clustering with validation/test assignment")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
