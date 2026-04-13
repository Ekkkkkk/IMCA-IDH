import numpy as np

from idh_core.clustering import fit_gower_pam, assign_to_reference_medoids
from idh_core.training import (
    ExperimentConfig,
    fit_static_model,
    generate_static_oof_probabilities,
    get_sequence_model,
    train_sequence_model,
    generate_sequence_oof_probabilities,
    predict_sequence_probabilities,
)
from idh_core.fusion import build_meta_features, fit_logistic_meta_learner


def main():
    rng = np.random.RandomState(42)

    train_patient_static = rng.normal(size=(120, 4))
    val_patient_static = rng.normal(size=(30, 4))
    test_patient_static = rng.normal(size=(30, 4))

    clustering = fit_gower_pam(train_patient_static, n_clusters=4, categorical_indices=[0])
    val_clusters = assign_to_reference_medoids(val_patient_static, clustering["medoid_points"], categorical_indices=[0])
    test_clusters = assign_to_reference_medoids(test_patient_static, clustering["medoid_points"], categorical_indices=[0])
    print("Train cluster counts:", np.bincount(clustering["labels"]))
    print("Validation cluster counts:", np.bincount(val_clusters))
    print("Test cluster counts:", np.bincount(test_clusters))

    n_train_windows, n_val_windows, n_test_windows = 600, 120, 120
    seq_len, channels = 30, 7
    train_time_x = rng.normal(size=(n_train_windows, seq_len, channels)).astype(np.float32)
    val_time_x = rng.normal(size=(n_val_windows, seq_len, channels)).astype(np.float32)
    test_time_x = rng.normal(size=(n_test_windows, seq_len, channels)).astype(np.float32)
    train_y = rng.binomial(1, 0.2, size=n_train_windows)
    val_y = rng.binomial(1, 0.2, size=n_val_windows)
    test_y = rng.binomial(1, 0.2, size=n_test_windows)

    train_static_x = rng.normal(size=(n_train_windows, 8))
    test_static_x = rng.normal(size=(n_test_windows, 8))

    static_oof = generate_static_oof_probabilities("RandomForest", train_static_x, train_y)
    static_fit = fit_static_model("RandomForest", train_static_x, train_y, test_static_x, test_y)

    config = ExperimentConfig(channels=channels, window_size=seq_len, batch_size=64, max_epoch=5)
    sequence_oof = generate_sequence_oof_probabilities(config, "LSTM", train_time_x, train_y)
    sequence_model = get_sequence_model(config, "LSTM")
    sequence_model, _ = train_sequence_model(sequence_model, train_time_x, train_y, val_time_x, val_y, config)
    sequence_test_probabilities = predict_sequence_probabilities(sequence_model, test_time_x, config.batch_size)

    train_meta = build_meta_features(static_oof, sequence_oof)
    test_meta = build_meta_features(static_fit["eval_probabilities"], sequence_test_probabilities)
    meta_outputs = fit_logistic_meta_learner(train_meta, train_y, test_meta, test_y)
    print("Meta test metrics:", meta_outputs["test_metrics"])


if __name__ == "__main__":
    main()
