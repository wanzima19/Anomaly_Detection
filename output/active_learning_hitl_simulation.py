import json
import os
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
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


def metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_roc_auc(y_true, y_prob),
        "avg_precision": _safe_avg_precision(y_true, y_prob),
        "positive_rate": float(y_pred.mean()),
    }


def normalize_dl_output(pred_output):
    if isinstance(pred_output, (list, tuple)):
        pred_output = pred_output[0] if len(pred_output) else np.array([])
    return np.asarray(pred_output, dtype=float).reshape(-1)


def adapter_features(base_prob):
    p = np.asarray(base_prob, dtype=float)
    return np.column_stack([p, p**2, np.abs(p - 0.5), np.log1p(p)])


def latency_stats(model, Z, runs=6):
    vals = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict_proba(Z)[:, 1]
        dt_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(Z))
        vals.append(dt_ms)
    arr = np.array(vals)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "throughput_sps": float(1000.0 / max(1e-12, np.mean(arr))),
    }


def load_stack(base_dir):
    with open(os.path.join(base_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    scaler = joblib.load(os.path.join(base_dir, "scaler.pkl"))
    models = {
        "random_forest": joblib.load(os.path.join(base_dir, "model_random_forest.pkl")),
        "xgboost": joblib.load(os.path.join(base_dir, "model_xgboost.pkl")),
        "logistic": joblib.load(os.path.join(base_dir, "model_logistic.pkl")),
    }

    dl = None
    if keras is not None:
        dl_path = os.path.join(base_dir, "model_deep_learning.keras")
        if os.path.exists(dl_path):
            dl = keras.models.load_model(dl_path, compile=False)

    return cfg, scaler, models, dl


def base_ensemble_prob(X_raw_df, X_scaled, models, dl_model, weights):
    rf = models["random_forest"].predict_proba(X_raw_df)[:, 1]
    xgb = models["xgboost"].predict_proba(X_raw_df)[:, 1]
    lg = models["logistic"].predict_proba(X_raw_df)[:, 1]
    if dl_model is not None:
        dl = normalize_dl_output(dl_model.predict(X_scaled, verbose=0))
    else:
        dl = np.zeros(len(X_raw_df), dtype=float)

    p = (
        float(weights.get("random_forest", 0.25)) * rf
        + float(weights.get("xgboost", 0.25)) * xgb
        + float(weights.get("logistic", 0.25)) * lg
        + float(weights.get("deep_learning", 0.25)) * dl
    )
    return np.clip(p, 0.0, 1.0)


def synthetic_human_label(y_true, base_prob, pca_mag, rng):
    # Synthetic HITL annotator: high quality overall, lower quality on ambiguous/OOD points.
    conf = np.abs(base_prob - 0.5)
    ood = (pca_mag > np.percentile(pca_mag, 90)).astype(float)
    acc = 0.86 + 0.12 * conf - 0.05 * ood
    acc = np.clip(acc, 0.80, 0.98)
    flips = rng.random(len(y_true)) > acc
    y_ann = y_true.copy()
    y_ann[flips] = 1 - y_ann[flips]
    return y_ann.astype(int), acc


def update_adapter(model, Z, y):
    classes = np.array([0, 1])
    uniq = np.unique(y)
    if len(uniq) < 2:
        # Skip unsafe one-class updates to avoid catastrophic swings.
        return False

    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw = np.clip(cw, 0.5, 12.0)
    wmap = {0: float(cw[0]), 1: float(cw[1])}
    sw = np.array([wmap[int(v)] for v in y], dtype=float)

    rng_local = np.random.default_rng(123)
    order = rng_local.permutation(len(y))
    batch = 128
    for i in range(0, len(order), batch):
        idx = order[i : i + batch]
        model.partial_fit(Z[idx], y[idx], classes=classes, sample_weight=sw[idx])
    return True


def choose_active_indices(p_current, budget, rng):
    # Uncertainty sampling with light diversity: 70% most uncertain + 30% random in uncertain band.
    uncertainty = np.abs(p_current - 0.5)
    rank = np.argsort(uncertainty)
    top = rank[: int(budget * 0.7)]
    band = np.where((p_current >= 0.35) & (p_current <= 0.65))[0]
    if len(band) == 0:
        band = rank[: min(len(rank), max(budget, 1))]
    n_rand = budget - len(top)
    if n_rand > 0:
        pick_rand = rng.choice(band, size=min(n_rand, len(band)), replace=False)
        chosen = np.unique(np.concatenate([top, pick_rand]))
    else:
        chosen = np.unique(top)

    if len(chosen) < budget:
        remainder = np.setdiff1d(np.arange(len(p_current)), chosen)
        add = rng.choice(remainder, size=min(budget - len(chosen), len(remainder)), replace=False)
        chosen = np.unique(np.concatenate([chosen, add]))
    return chosen


def run_simulation(base_dir):
    rng = np.random.default_rng(17)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg, scaler, models, dl_model = load_stack(base_dir)
    feature_names = cfg["feature_names"]
    weights = cfg.get(
        "model_weights",
        {"random_forest": 0.25, "xgboost": 0.25, "logistic": 0.25, "deep_learning": 0.25},
    )

    root = os.path.dirname(base_dir)
    X_scaled = np.load(os.path.join(root, "X_test_scaled.npy"))
    y = np.load(os.path.join(root, "y_test.npy")).astype(int)
    X_raw = scaler.inverse_transform(X_scaled)
    X_raw_df = pd.DataFrame(X_raw, columns=feature_names)

    idx = np.arange(len(y))
    idx_train, idx_eval = train_test_split(idx, test_size=0.25, stratify=y, random_state=17)

    # Start with a small seed labeled set to imitate initial reviewed cases.
    idx_seed, idx_unlabeled = train_test_split(
        idx_train,
        train_size=min(2000, len(idx_train) // 3),
        stratify=y[idx_train],
        random_state=19,
    )

    X_raw_seed = X_raw_df.iloc[idx_seed]
    X_scaled_seed = X_scaled[idx_seed]
    y_seed_true = y[idx_seed]
    p_seed = base_ensemble_prob(X_raw_seed, X_scaled_seed, models, dl_model, weights)
    Z_seed = adapter_features(p_seed)
    y_seed_ann, _ = synthetic_human_label(
        y_seed_true,
        p_seed,
        X_raw_seed["PCA_magnitude"].to_numpy(),
        rng,
    )

    active_model = SGDClassifier(
        loss="log_loss",
        alpha=5e-4,
        learning_rate="constant",
        eta0=0.005,
        penalty="l2",
        average=True,
        random_state=17,
    )
    random_model = SGDClassifier(
        loss="log_loss",
        alpha=5e-4,
        learning_rate="constant",
        eta0=0.005,
        penalty="l2",
        average=True,
        random_state=27,
    )
    update_adapter(active_model, Z_seed, y_seed_ann)
    update_adapter(random_model, Z_seed, y_seed_ann)

    X_raw_eval = X_raw_df.iloc[idx_eval]
    X_scaled_eval = X_scaled[idx_eval]
    y_eval = y[idx_eval]
    p_eval_base = base_ensemble_prob(X_raw_eval, X_scaled_eval, models, dl_model, weights)
    Z_eval = adapter_features(p_eval_base)

    cycles = 6
    query_budget = 250
    cycle_pool_size = 4500

    rows = []
    eff_rows = []
    budget_rows = []

    available = idx_unlabeled.copy()

    for c in range(1, cycles + 1):
        if len(available) < cycle_pool_size:
            available = idx_unlabeled.copy()

        pool_idx = rng.choice(available, size=min(cycle_pool_size, len(available)), replace=False)
        available = np.setdiff1d(available, pool_idx)

        X_raw_pool = X_raw_df.iloc[pool_idx]
        X_scaled_pool = X_scaled[pool_idx]
        y_pool_true = y[pool_idx]

        p_pool_base = base_ensemble_prob(X_raw_pool, X_scaled_pool, models, dl_model, weights)
        Z_pool = adapter_features(p_pool_base)

        # Active arm before update
        p_before_active = active_model.predict_proba(Z_eval)[:, 1]
        m_before_active = metrics(y_eval, p_before_active)
        lat_before = latency_stats(active_model, Z_eval)

        # Random arm before update (for burden/comparison)
        p_before_random = random_model.predict_proba(Z_eval)[:, 1]
        m_before_random = metrics(y_eval, p_before_random)

        # Query selection
        p_pool_active = active_model.predict_proba(Z_pool)[:, 1]
        q_active_local = choose_active_indices(p_pool_active, budget=query_budget, rng=rng)
        q_random_local = rng.choice(np.arange(len(pool_idx)), size=query_budget, replace=False)

        # Simulated human annotation
        y_ann_active, ann_acc_active = synthetic_human_label(
            y_pool_true[q_active_local],
            p_pool_base[q_active_local],
            X_raw_pool.iloc[q_active_local]["PCA_magnitude"].to_numpy(),
            rng,
        )
        y_ann_random, ann_acc_random = synthetic_human_label(
            y_pool_true[q_random_local],
            p_pool_base[q_random_local],
            X_raw_pool.iloc[q_random_local]["PCA_magnitude"].to_numpy(),
            rng,
        )

        # Update
        t0 = time.perf_counter()
        _ = update_adapter(active_model, Z_pool[q_active_local], y_ann_active)
        active_update_s = time.perf_counter() - t0

        t1 = time.perf_counter()
        _ = update_adapter(random_model, Z_pool[q_random_local], y_ann_random)
        random_update_s = time.perf_counter() - t1

        p_after_active = active_model.predict_proba(Z_eval)[:, 1]
        p_after_random = random_model.predict_proba(Z_eval)[:, 1]
        m_after_active = metrics(y_eval, p_after_active)
        m_after_random = metrics(y_eval, p_after_random)
        lat_after = latency_stats(active_model, Z_eval)

        rows.append(
            {
                "cycle": c,
                "arm": "active",
                "phase": "before",
                **m_before_active,
            }
        )
        rows.append(
            {
                "cycle": c,
                "arm": "active",
                "phase": "after",
                **m_after_active,
            }
        )
        rows.append(
            {
                "cycle": c,
                "arm": "random",
                "phase": "before",
                **m_before_random,
            }
        )
        rows.append(
            {
                "cycle": c,
                "arm": "random",
                "phase": "after",
                **m_after_random,
            }
        )

        eff_rows.append(
            {
                "cycle": c,
                "active_update_time_s": float(active_update_s),
                "random_update_time_s": float(random_update_s),
                "active_latency_p50_before_ms": lat_before["p50_ms"],
                "active_latency_p90_before_ms": lat_before["p90_ms"],
                "active_latency_p50_after_ms": lat_after["p50_ms"],
                "active_latency_p90_after_ms": lat_after["p90_ms"],
                "active_throughput_before_sps": lat_before["throughput_sps"],
                "active_throughput_after_sps": lat_after["throughput_sps"],
                "sim_annotator_acc_active": float(np.mean(ann_acc_active)),
                "sim_annotator_acc_random": float(np.mean(ann_acc_random)),
            }
        )

        budget_rows.append(
            {
                "cycle": c,
                "pool_size": int(len(pool_idx)),
                "labels_active": int(len(q_active_local)),
                "labels_random": int(len(q_random_local)),
                "labels_full_supervision": int(len(pool_idx)),
                "label_saving_vs_full_pct": float(1 - len(q_active_local) / len(pool_idx)),
            }
        )

    metrics_df = pd.DataFrame(rows)
    eff_df = pd.DataFrame(eff_rows)
    budget_df = pd.DataFrame(budget_rows)

    out_metrics = os.path.join(base_dir, f"active_learning_metrics_{ts}.csv")
    out_eff = os.path.join(base_dir, f"active_learning_efficiency_{ts}.csv")
    out_budget = os.path.join(base_dir, f"active_learning_labeling_{ts}.csv")

    metrics_df.to_csv(out_metrics, index=False)
    eff_df.to_csv(out_eff, index=False)
    budget_df.to_csv(out_budget, index=False)

    # Stability and burden summary
    def after_series(df, arm, metric_name):
        return (
            df[(df["arm"] == arm) & (df["phase"] == "after")]
            .sort_values("cycle")[metric_name]
            .to_numpy()
        )

    active_after_f1 = after_series(metrics_df, "active", "f1")
    random_after_f1 = after_series(metrics_df, "random", "f1")
    active_delta_f1 = (
        metrics_df[(metrics_df["arm"] == "active") & (metrics_df["phase"] == "after")]
        .sort_values("cycle")["f1"]
        .to_numpy()
        - metrics_df[(metrics_df["arm"] == "active") & (metrics_df["phase"] == "before")]
        .sort_values("cycle")["f1"]
        .to_numpy()
    )

    summary = {
        "timestamp": ts,
        "setup": {
            "cycles": cycles,
            "pool_per_cycle": cycle_pool_size,
            "query_budget": query_budget,
            "strategy": "uncertainty sampling + synthetic HITL labels",
        },
        "performance": {
            "active_final_f1_after": float(active_after_f1[-1]),
            "random_final_f1_after": float(random_after_f1[-1]),
            "active_mean_f1_after": float(np.mean(active_after_f1)),
            "random_mean_f1_after": float(np.mean(random_after_f1)),
            "active_f1_gain_from_cycle1_to_cycleN": float(active_after_f1[-1] - active_after_f1[0]),
            "random_f1_gain_from_cycle1_to_cycleN": float(random_after_f1[-1] - random_after_f1[0]),
        },
        "efficiency": {
            "avg_update_time_active_s": float(eff_df["active_update_time_s"].mean()),
            "avg_update_time_random_s": float(eff_df["random_update_time_s"].mean()),
            "avg_latency_p90_before_ms": float(eff_df["active_latency_p90_before_ms"].mean()),
            "avg_latency_p90_after_ms": float(eff_df["active_latency_p90_after_ms"].mean()),
            "avg_label_saving_vs_full_pct": float(budget_df["label_saving_vs_full_pct"].mean()),
        },
        "stability": {
            "std_delta_f1_active": float(np.std(active_delta_f1)),
            "negative_delta_cycles_active": int(np.sum(active_delta_f1 < 0)),
        },
        "artifacts": {
            "metrics_csv": out_metrics,
            "efficiency_csv": out_eff,
            "labeling_csv": out_budget,
        },
    }

    # Plot trajectories
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for arm, color in [("active", "tab:blue"), ("random", "tab:orange")]:
        d_before = metrics_df[(metrics_df["arm"] == arm) & (metrics_df["phase"] == "before")].sort_values("cycle")
        d_after = metrics_df[(metrics_df["arm"] == arm) & (metrics_df["phase"] == "after")].sort_values("cycle")
        axes[0, 0].plot(d_before["cycle"], d_before["f1"], "--o", color=color, alpha=0.6, label=f"{arm} before")
        axes[0, 0].plot(d_after["cycle"], d_after["f1"], "-o", color=color, label=f"{arm} after")
        axes[0, 1].plot(d_before["cycle"], d_before["accuracy"], "--o", color=color, alpha=0.6, label=f"{arm} before")
        axes[0, 1].plot(d_after["cycle"], d_after["accuracy"], "-o", color=color, label=f"{arm} after")

    axes[0, 0].set_title("F1 across cycles")
    axes[0, 1].set_title("Accuracy across cycles")
    axes[0, 0].set_xlabel("Cycle")
    axes[0, 1].set_xlabel("Cycle")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[1, 0].plot(eff_df["cycle"], eff_df["active_update_time_s"], "-o", label="update time (active)")
    axes[1, 0].plot(eff_df["cycle"], eff_df["random_update_time_s"], "-o", label="update time (random)")
    axes[1, 0].set_title("Update time")
    axes[1, 0].set_xlabel("Cycle")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(eff_df["cycle"], eff_df["active_latency_p90_before_ms"], "--o", label="p90 before")
    axes[1, 1].plot(eff_df["cycle"], eff_df["active_latency_p90_after_ms"], "-o", label="p90 after")
    axes[1, 1].set_title("Active arm latency p90")
    axes[1, 1].set_xlabel("Cycle")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    perf_plot = os.path.join(base_dir, f"active_learning_trajectories_{ts}.png")
    fig.savefig(perf_plot, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(budget_df["cycle"], budget_df["labels_active"], "-o", label="labeled (active)")
    ax.plot(budget_df["cycle"], budget_df["labels_full_supervision"], "-o", label="labeled (full supervision)")
    ax.set_title("Labeling burden per cycle")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Samples labeled")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    burden_plot = os.path.join(base_dir, f"active_learning_label_burden_{ts}.png")
    fig.savefig(burden_plot, dpi=150)
    plt.close(fig)

    summary["artifacts"]["trajectory_plot_png"] = perf_plot
    summary["artifacts"]["label_burden_plot_png"] = burden_plot
    summary_path = os.path.join(base_dir, f"active_learning_summary_{ts}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Active learning HITL simulation complete.")
    print(f"- {out_metrics}")
    print(f"- {out_eff}")
    print(f"- {out_budget}")
    print(f"- {perf_plot}")
    print(f"- {burden_plot}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    run_simulation(BASE_DIR)
