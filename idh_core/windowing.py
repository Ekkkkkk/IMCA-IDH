import numpy as np
import torch


def _to_scalar_label(label):
    if isinstance(label, torch.Tensor):
        return int(label.argmax().item()) if label.ndim > 0 else int(label.item())
    if isinstance(label, np.ndarray):
        return int(label.argmax()) if label.ndim > 0 else int(label.item())
    if isinstance(label, list):
        return int(np.argmax(label)) if len(label) > 1 else int(label[0])
    return int(label)


def slice_windows(machine_data_list, labels, end_list, window_size, step_size, static_outputs=None):
    windowed_data = []
    windowed_labels = []
    windowed_static_outputs = []
    static_outputs = [] if static_outputs is None else static_outputs

    for idx, machine_data in enumerate(machine_data_list):
        label = _to_scalar_label(labels[idx])
        end = end_list[idx]
        static_output = static_outputs[idx] if len(static_outputs) > 0 else 0.0

        for start in range(0, int(end) - window_size + 1, step_size):
            windowed_data.append(machine_data[:, start:start + window_size])
            windowed_labels.append(label)
            windowed_static_outputs.append(static_output)

    X = torch.stack(windowed_data, dim=0).numpy()
    y = np.asarray(windowed_labels, dtype=np.int64)
    static_scores = np.asarray(windowed_static_outputs, dtype=float)
    return X, y, static_scores
