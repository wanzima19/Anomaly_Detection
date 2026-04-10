import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import keras
except Exception:  # pragma: no cover
    keras = None


def _safe_roc_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_avg_precision(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _safe_log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(log_loss(y_true, y_prob))


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "avg_precision": _safe_avg_precision(y_true, y_prob),
        "mae": float(mean_absolute_error(y_true, y_prob)),
        "log_loss": _safe_log_loss(y_true, y_prob),
    }


def get_ram_mb():
    try:
        import psutil

        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss / (1024**2))
    except Exception:
        return float("nan")


def get_vram_mb():
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return float("nan")
        info = tf.config.experimental.get_memory_info("GPU:0")
        # current is bytes
        return float(info.get("current", 0.0) / (1024**2))
    except Exception:
        return float("nan")


def normalize_dl_output(pred_output):
    if isinstance(pred_output, (list, tuple)):
        if len(pred_output) == 0:
            return np.array([], dtype=float)
        pred_output = pred_output[0]
    return np.asarray(pred_output, dtype=float).reshape(-1)


def load_artifacts(base_dir):
    with open(os.path.join(base_dir, "model_config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)

    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))

    models = {
        "random_forest": joblib.load(os.path.join(base_dir, "model_random_forest.pkl")),
        "xgboost": joblib.load(os.path.join(base_dir, "model_xgboost.pkl")),
        "logistic": joblib.load(os.path.join(base_dir, "model_logistic.pkl")),
    }

    dl_model = None
    if keras is not None:
        dl_model = keras.models.load_model(
            os.path.join(base_dir, "model_deep_learning.keras"), compile=False
        )

    return config, scaler, models, dl_model


def base_ensemble_proba(X_raw, X_scaled, models, dl_model, weights):
    preds = {}
    for name, model in models.items():
        preds[name] = model.predict_proba(X_raw)[:, 1]

    if dl_model is not None:
        dl_pred = normalize_dl_output(dl_model.predict(X_scaled, verbose=0))
        preds["deep_learning"] = dl_pred
    else:
        preds["deep_learning"] = np.zeros(X_raw.shape[0], dtype=float)

    all_names = ["random_forest", "xgboost", "logistic", "deep_learning"]
    score = np.zeros(X_raw.shape[0], dtype=float)
    for name in all_names:
        w = float(weights.get(name, 0.25))
        score += w * preds[name]
    return np.clip(score, 0.0, 1.0)


def make_adapter_features(base_prob):
    base_prob = np.asarray(base_prob, dtype=float)
    return np.column_stack(
        [
            base_prob,
            base_prob**2,
            np.abs(base_prob - 0.5),
            np.log1p(base_prob),
        ]
    )


def apply_covariate_drift(X_scaled, strength, rng, drift_idx):
    Xd = X_scaled.copy()
    shift = 0.2 * strength
    noise = 0.03 * strength
    Xd[:, drift_idx] = Xd[:, drift_idx] + shift + rng.normal(
        0.0, noise, size=(Xd.shape[0], len(drift_idx))
    )
    return Xd


def sample_interval_indices(y, total_n, target_pos_rate, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    n_pos = int(total_n * target_pos_rate)
    n_pos = min(max(n_pos, 20), len(pos_idx))
    n_neg = total_n - n_pos
    n_neg = min(max(n_neg, 100), len(neg_idx))

    idx_pos = rng.choice(pos_idx, size=n_pos, replace=(n_pos > len(pos_idx)))
    idx_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([idx_pos, idx_neg])
    rng.shuffle(idx)
    return idx


def measure_inference_latency(adapter, Z_eval, runs=5):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = adapter.predict_proba(Z_eval)[:, 1]
        t1 = time.perf_counter()
        per_sample_ms = (t1 - t0) * 1000.0 / max(1, len(Z_eval))
        times.append(per_sample_ms)
    arr = np.array(times, dtype=float)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_samples_per_sec": float(1000.0 / max(1e-9, np.mean(arr))),
    }


def run_experiment(base_dir):
    rng = np.random.default_rng(42)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    config, scaler, models, dl_model = load_artifacts(base_dir)
    weights = config.get(
        "model_weights",
        {"random_forest": 0.25, "xgboost": 0.25, "logistic": 0.25, "deep_learning": 0.25},
    )

    X_scaled_all = np.load(os.path.join(os.path.dirname(base_dir), "X_test_scaled.npy"))
    y_all = np.load(os.path.join(os.path.dirname(base_dir), "y_test.npy")).astype(int)
    X_raw_all = scaler.inverse_transform(X_scaled_all)

    idx = np.arange(len(y_all))
    idx_init, idx_pool = train_test_split(
        idx,
        test_size=0.8,
        stratify=y_all,
        random_state=42,
    )

    base_prob_init = base_ensemble_proba(
        X_raw_all[idx_init], X_scaled_all[idx_init], models, dl_model, weights
    )
    Z_init = make_adapter_features(base_prob_init)
    y_init = y_all[idx_init]

    adapter = SGDClassifier(
        loss="log_loss",
        alpha=1e-4,
        learning_rate="optimal",
        class_weight=None,
        random_state=42,
    )
    classes = np.array([0, 1])
    init_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_init)
    init_w_map = {0: float(init_w[0]), 1: float(init_w[1])}
    init_sample_weight = np.array([init_w_map[int(v)] for v in y_init], dtype=float)
    adapter.partial_fit(
        Z_init,
        y_init,
        classes=classes,
        sample_weight=init_sample_weight,
    )

    records = []
    eff_records = []

    n_steps = 6
    interval_n = 6000
    drift_idx = list(range(min(7, X_scaled_all.shape[1])))

    for step in range(1, n_steps + 1):
        drift_strength = 0.4 + 0.3 * step
        target_pos_rate = min(0.005 + 0.003 * step, 0.03)

        idx_step = sample_interval_indices(
            y_all[idx_pool],
            total_n=interval_n,
            target_pos_rate=target_pos_rate,
            rng=rng,
        )
        pool_indices = idx_pool[idx_step]

        X_scaled_step = X_scaled_all[pool_indices]
        y_step = y_all[pool_indices]
        X_scaled_step = apply_covariate_drift(X_scaled_step, drift_strength, rng, drift_idx)
        X_raw_step = scaler.inverse_transform(X_scaled_step)

        X_tr, X_ev, y_tr, y_ev = train_test_split(
            X_scaled_step,
            y_step,
            test_size=0.3,
            stratify=y_step,
            random_state=100 + step,
        )

        X_raw_tr = scaler.inverse_transform(X_tr)
        X_raw_ev = scaler.inverse_transform(X_ev)

        base_prob_tr = base_ensemble_proba(X_raw_tr, X_tr, models, dl_model, weights)
        base_prob_ev = base_ensemble_proba(X_raw_ev, X_ev, models, dl_model, weights)

        Z_tr = make_adapter_features(base_prob_tr)
        Z_ev = make_adapter_features(base_prob_ev)

        # Before update metrics
        pred_before = adapter.predict_proba(Z_ev)[:, 1]
        m_before = compute_metrics(y_ev, pred_before)
        lat_before = measure_inference_latency(adapter, Z_ev)

        records.append(
            {
                "step": step,
                "phase": "before_update",
                "drift_strength": drift_strength,
                "target_pos_rate": target_pos_rate,
                **m_before,
            }
        )

        # Continual update
        ram_before = get_ram_mb()
        vram_before = get_vram_mb()
        t0 = time.perf_counter()
        batch_size = 1024
        upd_w = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        upd_w_map = {0: float(upd_w[0]), 1: float(upd_w[1])}
        for i in range(0, len(Z_tr), batch_size):
            yb = y_tr[i : i + batch_size]
            sw = np.array([upd_w_map[int(v)] for v in yb], dtype=float)
            adapter.partial_fit(Z_tr[i : i + batch_size], yb, sample_weight=sw)
        update_time_s = time.perf_counter() - t0
        ram_after = get_ram_mb()
        vram_after = get_vram_mb()

        # After update metrics
        pred_after = adapter.predict_proba(Z_ev)[:, 1]
        m_after = compute_metrics(y_ev, pred_after)
        lat_after = measure_inference_latency(adapter, Z_ev)

        records.append(
            {
                "step": step,
                "phase": "after_update",
                "drift_strength": drift_strength,
                "target_pos_rate": target_pos_rate,
                **m_after,
            }
        )

        eff_records.append(
            {
                "step": step,
                "update_batch_size": batch_size,
                "update_samples": int(len(Z_tr)),
                "update_time_s": float(update_time_s),
                "ram_before_mb": ram_before,
                "ram_after_mb": ram_after,
                "ram_delta_mb": float(ram_after - ram_before)
                if np.isfinite(ram_after) and np.isfinite(ram_before)
                else float("nan"),
                "vram_before_mb": vram_before,
                "vram_after_mb": vram_after,
                "vram_delta_mb": float(vram_after - vram_before)
                if np.isfinite(vram_after) and np.isfinite(vram_before)
                else float("nan"),
                "lat_before_p50_ms": lat_before["p50_ms"],
                "lat_before_p90_ms": lat_before["p90_ms"],
                "lat_after_p50_ms": lat_after["p50_ms"],
                "lat_after_p90_ms": lat_after["p90_ms"],
                "throughput_before_sps": lat_before["throughput_samples_per_sec"],
                "throughput_after_sps": lat_after["throughput_samples_per_sec"],
            }
        )

    df_metrics = pd.DataFrame(records)
    df_eff = pd.DataFrame(eff_records)

    metrics_csv = os.path.join(base_dir, f"continual_metrics_{ts}.csv")
    eff_csv = os.path.join(base_dir, f"continual_efficiency_{ts}.csv")
    df_metrics.to_csv(metrics_csv, index=False)
    df_eff.to_csv(eff_csv, index=False)

    # Trajectory plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for metric, ax in [
        ("f1", axes[0, 0]),
        ("accuracy", axes[0, 1]),
        ("mae", axes[1, 0]),
        ("roc_auc", axes[1, 1]),
    ]:
        for phase, style in [("before_update", "--o"), ("after_update", "-o")]:
            d = df_metrics[df_metrics["phase"] == phase]
            ax.plot(d["step"], d[metric], style, label=phase)
        ax.set_title(metric)
        ax.set_xlabel("Drift interval step")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plot_metrics_png = os.path.join(base_dir, f"continual_metric_trajectories_{ts}.png")
    fig.savefig(plot_metrics_png, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df_eff["step"], df_eff["update_time_s"], "-o")
    axes[0].set_title("Update Time (s)")
    axes[0].set_xlabel("Drift interval step")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_eff["step"], df_eff["lat_before_p90_ms"], "--o", label="before p90")
    axes[1].plot(df_eff["step"], df_eff["lat_after_p90_ms"], "-o", label="after p90")
    axes[1].set_title("Inference Latency p90 (ms)")
    axes[1].set_xlabel("Drift interval step")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plot_eff_png = os.path.join(base_dir, f"continual_efficiency_trajectories_{ts}.png")
    fig.savefig(plot_eff_png, dpi=150)
    plt.close(fig)

    summary = {
        "timestamp": ts,
        "n_steps": n_steps,
        "interval_size": interval_n,
        "update_method": "SGDClassifier.partial_fit adapter over base ensemble score",
        "artifacts": {
            "metrics_csv": metrics_csv,
            "efficiency_csv": eff_csv,
            "metric_plot_png": plot_metrics_png,
            "efficiency_plot_png": plot_eff_png,
        },
        "final_step": {
            "before": df_metrics[(df_metrics["step"] == n_steps) & (df_metrics["phase"] == "before_update")]
            .iloc[0]
            .to_dict(),
            "after": df_metrics[(df_metrics["step"] == n_steps) & (df_metrics["phase"] == "after_update")]
            .iloc[0]
            .to_dict(),
        },
        "aggregate": {
            "mean_f1_before": float(df_metrics[df_metrics["phase"] == "before_update"]["f1"].mean()),
            "mean_f1_after": float(df_metrics[df_metrics["phase"] == "after_update"]["f1"].mean()),
            "mean_accuracy_before": float(
                df_metrics[df_metrics["phase"] == "before_update"]["accuracy"].mean()
            ),
            "mean_accuracy_after": float(
                df_metrics[df_metrics["phase"] == "after_update"]["accuracy"].mean()
            ),
            "mean_mae_before": float(df_metrics[df_metrics["phase"] == "before_update"]["mae"].mean()),
            "mean_mae_after": float(df_metrics[df_metrics["phase"] == "after_update"]["mae"].mean()),
            "mean_update_time_s": float(df_eff["update_time_s"].mean()),
            "mean_ram_delta_mb": float(df_eff["ram_delta_mb"].mean()),
        },
    }

    summary_json = os.path.join(base_dir, f"continual_summary_{ts}.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Experiment completed.")
    print(f"- {metrics_csv}")
    print(f"- {eff_csv}")
    print(f"- {plot_metrics_png}")
    print(f"- {plot_eff_png}")
    print(f"- {summary_json}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    run_experiment(BASE_DIR)
