import numpy as np
import torch


def compute_average_precision(targets: np.ndarray, predictions: np.ndarray):
    """
    Computes Average Precision (AP) for a single binary class.
    targets: 1D array of binary ground-truth labels (0 or 1).
    predictions: 1D array of predicted confidence scores or logits.
    """
    num_positives = np.sum(targets)
    if num_positives == 0:
        return None

    # Sort descending by prediction scores
    sort_idx = np.argsort(-predictions)
    sorted_targets = targets[sort_idx]

    # Cumulative true positives and false positives
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1 - sorted_targets)

    precision = tp / (tp + fp)
    recall = tp / num_positives

    # Trapezoidal approximation using step-wise precision
    recall_diff = np.diff(np.concatenate(([0.0], recall)))
    ap = np.sum(precision * recall_diff)
    return float(ap)


def compute_roc_auc(targets: np.ndarray, predictions: np.ndarray):
    """
    Computes Area Under ROC Curve (ROC-AUC) for a single binary class using Wilcoxon-Mann-Whitney.
    """
    n_pos = int(np.sum(targets == 1))
    n_neg = int(np.sum(targets == 0))
    if n_pos == 0 or n_neg == 0:
        return None

    sort_idx = np.argsort(predictions)
    ranks = np.empty_like(sort_idx)
    ranks[sort_idx] = np.arange(len(predictions)) + 1

    pos_ranks_sum = np.sum(ranks[targets == 1])
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


class MultiLabelClassificationMetrics:
    """
    Tracks and computes comprehensive multi-label evaluation metrics:
      - mAP (Mean Average Precision)
      - mAUC (Mean ROC-AUC)
      - Micro-F1 & Macro-F1 (at threshold 0.5)
      - Top-1 and Top-5 Hit Rates
    """
    def __init__(self, num_classes=200, threshold=0.5):
        self.num_classes = num_classes
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.all_predictions = []
        self.all_targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        tgts = targets.detach().cpu().numpy()

        self.all_predictions.append(probs)
        self.all_targets.append(tgts)

    def compute(self):
        if not self.all_predictions:
            return {
                "mAP": 0.0,
                "mAUC": 0.0,
                "micro_f1": 0.0,
                "macro_f1": 0.0,
                "top1_hit": 0.0,
                "top5_hit": 0.0,
            }

        preds = np.concatenate(self.all_predictions, axis=0)  # [N, num_classes]
        targets = np.concatenate(self.all_targets, axis=0)    # [N, num_classes]
        N, num_classes = preds.shape

        # 1. mAP and mAUC per class
        ap_list = []
        auc_list = []
        for c in range(num_classes):
            ap = compute_average_precision(targets[:, c], preds[:, c])
            if ap is not None:
                ap_list.append(ap)

            auc = compute_roc_auc(targets[:, c], preds[:, c])
            if auc is not None:
                auc_list.append(auc)

        mean_ap = float(np.mean(ap_list)) if ap_list else 0.0
        mean_auc = float(np.mean(auc_list)) if auc_list else 0.0

        # 2. Binary predictions at threshold
        bin_preds = (preds >= self.threshold).astype(int)
        bin_targets = targets.astype(int)

        # Micro-F1
        tp = np.sum((bin_preds == 1) & (bin_targets == 1))
        fp = np.sum((bin_preds == 1) & (bin_targets == 0))
        fn = np.sum((bin_preds == 0) & (bin_targets == 1))
        micro_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)

        # Macro-F1
        class_f1s = []
        for c in range(num_classes):
            c_tp = np.sum((bin_preds[:, c] == 1) & (bin_targets[:, c] == 1))
            c_fp = np.sum((bin_preds[:, c] == 1) & (bin_targets[:, c] == 0))
            c_fn = np.sum((bin_preds[:, c] == 0) & (bin_targets[:, c] == 1))
            denom = 2 * c_tp + c_fp + c_fn
            if denom > 0:
                class_f1s.append((2 * c_tp) / denom)
            elif np.sum(bin_targets[:, c]) == 0:
                # Class not present in ground truth
                continue
            else:
                class_f1s.append(0.0)
        macro_f1 = float(np.mean(class_f1s)) if class_f1s else 0.0

        # 3. Top-1 and Top-5 Hit Rate
        top1_indices = np.argmax(preds, axis=1)
        top1_hits = [bin_targets[i, top1_indices[i]] == 1 for i in range(N)]
        top1_hit_rate = float(np.mean(top1_hits)) if N > 0 else 0.0

        k = min(5, num_classes)
        top5_indices = np.argsort(-preds, axis=1)[:, :k]
        top5_hits = [np.any(bin_targets[i, top5_indices[i]] == 1) for i in range(N)]
        top5_hit_rate = float(np.mean(top5_hits)) if N > 0 else 0.0

        return {
            "mAP": mean_ap,
            "mAUC": mean_auc,
            "micro_f1": float(micro_f1),
            "macro_f1": macro_f1,
            "top1_hit": top1_hit_rate,
            "top5_hit": top5_hit_rate,
            "num_evaluated_classes": len(ap_list),
        }
