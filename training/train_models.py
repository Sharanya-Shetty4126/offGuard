"""Machine learning training and rigorous benchmarking pipeline for offGuard.

Implements:
1. Episode-grouped dataset splitting (zero row-level data leakage).
2. Scenario holdout splitting (e.g. hold out I/O thrash to test true generalization).
3. Class-balanced model training:
   - Calibrated Logistic Regression (L2 regularized, Platt scaled).
   - Calibrated HistGradientBoostingClassifier (shallow trees, early stopping, Platt scaled).
4. Direct 4-way evaluation through the same hysteresis state machine:
   - StaticThresholdBaseline vs TimeToExhaustionBaseline vs LogisticRegression vs GradientBoosting.
5. Lag ablation and hard-negative ablation analysis.
"""

from dataclasses import dataclass, field
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from agent.baselines import StaticThresholdBaseline, TimeToExhaustionBaseline
from agent.config import DEFAULT_CONFIG, AgentConfig
from agent.features import RollingFeaturePipeline
from training.evaluation import EvaluationReport, evaluate_detector


@dataclass
class ModelArtifact:
    """Packaged model artifact with metadata for deployment."""

    model_name: str
    feature_names: List[str]
    windows_sec: List[int]
    include_lag: bool
    scaler: Optional[StandardScaler]
    classifier: Any
    calibrator: Optional[CalibratedClassifierCV]
    optimal_threshold: float
    training_metadata: Dict[str, Any]

    def predict_sample(self, raw_features: Dict[str, float]) -> Tuple[bool, float]:
        """Predict alert flag and calibrated risk score for a feature vector."""
        x = np.array([[raw_features.get(k, 0.0) for k in self.feature_names]], dtype=np.float32)

        if self.scaler is not None:
            x = self.scaler.transform(x)

        if self.calibrator is not None:
            risk = float(self.calibrator.predict_proba(x)[0, 1])
        else:
            risk = float(self.classifier.predict_proba(x)[0, 1])

        is_alert = risk >= self.optimal_threshold
        return is_alert, risk


def split_episodes(
    episodes: List[List[Dict[str, Any]]],
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    holdout_scenario: Optional[str] = None,
    seed: int = 42,
) -> Tuple[List[List[Dict[str, Any]]], List[List[Dict[str, Any]]], List[List[Dict[str, Any]]]]:
    """Split dataset by discrete episodes (never random row shuffle).

    If holdout_scenario is specified, all episodes matching that scenario
    are placed exclusively into the test set to evaluate cross-scenario generalization.
    """
    rng = random.Random(seed)

    test_episodes = []
    pool_episodes = []

    for ep in episodes:
        if not ep:
            continue
        ep_scenario = ep[0].get("scenario_type", ep[0].get("episode_id", ""))
        if holdout_scenario and (holdout_scenario in str(ep_scenario)):
            test_episodes.append(ep)
        else:
            pool_episodes.append(ep)

    rng.shuffle(pool_episodes)
    n_pool = len(pool_episodes)
    n_train = int(n_pool * (train_ratio / (train_ratio + val_ratio + (0.2 if not holdout_scenario else 0.0))))
    n_val = int(n_pool * (val_ratio / (train_ratio + val_ratio + (0.2 if not holdout_scenario else 0.0))))

    train_episodes = pool_episodes[:n_train]
    val_episodes = pool_episodes[n_train : n_train + n_val]

    if not holdout_scenario:
        test_episodes.extend(pool_episodes[n_train + n_val :])

    return train_episodes, val_episodes, test_episodes


def _extract_dataset_matrices(
    episodes: List[List[Dict[str, Any]]],
    feature_names: List[str],
    pipeline: RollingFeaturePipeline,
    drop_post_failure: bool = True,
    warmup_skip_sec: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Transform episodes into feature matrices X, label vector y, and episode group IDs."""
    x_list = []
    y_list = []
    groups = []

    for ep in episodes:
        if not ep:
            continue
        feats = pipeline.transform_episode(ep)
        ep_id = str(ep[0].get("episode_id", "unknown"))
        base_t = float(ep[0].get("timestamp", 0.0))

        for s, f in zip(ep, feats):
            ts = float(s.get("timestamp", 0.0))
            is_post = bool(s.get("is_post_failure", False))

            # Drop post-failure / post-recovery samples
            if drop_post_failure and is_post:
                continue

            # Skip initial warmup period if configured
            if (ts - base_t) < warmup_skip_sec:
                continue

            row = [f.get(k, 0.0) for k in feature_names]
            x_list.append(row)
            y_list.append(int(s.get("target_fail_within_H", 0)))
            groups.append(ep_id)

    if not x_list:
        return np.empty((0, len(feature_names))), np.empty((0,), dtype=int), []

    return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=int), groups


def train_models(
    episodes: List[List[Dict[str, Any]]],
    include_lag: bool = True,
    holdout_scenario: Optional[str] = None,
    output_model_dir: str = "models",
    seed: int = 42,
) -> Dict[str, ModelArtifact]:
    """Train calibrated Logistic Regression and HistGradientBoosting models."""
    pipeline = RollingFeaturePipeline()
    feature_names = RollingFeaturePipeline.get_feature_names(include_lag=include_lag)

    train_eps, val_eps, test_eps = split_episodes(
        episodes,
        train_ratio=0.60,
        val_ratio=0.20,
        holdout_scenario=holdout_scenario,
        seed=seed,
    )

    print(
        f"[offGuard Trainer] Episode split: Train={len(train_eps)}, Val={len(val_eps)}, Test={len(test_eps)} "
        f"(Holdout scenario: {holdout_scenario or 'None'}, Include lag: {include_lag})"
    )

    x_train, y_train, _ = _extract_dataset_matrices(train_eps, feature_names, pipeline)
    x_val, y_val, _ = _extract_dataset_matrices(val_eps, feature_names, pipeline)

    artifacts: Dict[str, ModelArtifact] = {}

    # ---------------- 1. Logistic Regression ----------------
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train) if len(x_train) > 0 else x_train
    x_val_scaled = scaler.transform(x_val) if len(x_val) > 0 else x_val

    lr_base = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
    )
    if len(x_train) > 0 and len(np.unique(y_train)) > 1:
        lr_base.fit(x_train_scaled, y_train)
        # Platt scaling on validation fold
        lr_calibrator = None
        if len(x_val) > 0 and len(np.unique(y_val)) > 1:
            try:
                lr_calibrator = CalibratedClassifierCV(estimator=lr_base, method="sigmoid", cv="prefit")
                lr_calibrator.fit(x_val_scaled, y_val)
            except Exception:
                lr_calibrator = None
    else:
        lr_calibrator = None

    artifacts["logistic_regression"] = ModelArtifact(
        model_name="logistic_regression",
        feature_names=feature_names,
        windows_sec=DEFAULT_CONFIG.window_sizes_sec,
        include_lag=include_lag,
        scaler=scaler,
        classifier=lr_base,
        calibrator=lr_calibrator,
        optimal_threshold=0.50,
        training_metadata={"train_samples": len(x_train), "val_samples": len(x_val)},
    )

    # ---------------- 2. HistGradientBoostingClassifier ----------------
    hgb_base = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_depth=4,
        max_iter=100,
        early_stopping=True,
        random_state=seed,
    )
    if len(x_train) > 0 and len(np.unique(y_train)) > 1:
        hgb_base.fit(x_train, y_train)
        hgb_calibrator = None
        if len(x_val) > 0 and len(np.unique(y_val)) > 1:
            try:
                hgb_calibrator = CalibratedClassifierCV(estimator=hgb_base, method="sigmoid", cv="prefit")
                hgb_calibrator.fit(x_val, y_val)
            except Exception:
                hgb_calibrator = None
    else:
        hgb_calibrator = None

    artifacts["gradient_boosting"] = ModelArtifact(
        model_name="gradient_boosting",
        feature_names=feature_names,
        windows_sec=DEFAULT_CONFIG.window_sizes_sec,
        include_lag=include_lag,
        scaler=None,  # Tree models don't require feature scaling
        classifier=hgb_base,
        calibrator=hgb_calibrator,
        optimal_threshold=0.50,
        training_metadata={"train_samples": len(x_train), "val_samples": len(x_val)},
    )

    # Save artifacts to models/ directory
    os.makedirs(output_model_dir, exist_ok=True)
    suffix = "_no_lag" if not include_lag else ""
    for name, art in artifacts.items():
        save_path = os.path.join(output_model_dir, f"{name}{suffix}.joblib")
        joblib.dump(art, save_path)
        print(f"[offGuard Trainer] Exported model artifact -> '{save_path}'")

    return artifacts


def run_benchmark_comparison(
    test_episodes: List[List[Dict[str, Any]]],
    artifacts: Dict[str, ModelArtifact],
    horizon_sec: float = 120.0,
    min_effective_lead_time_sec: float = 60.0,
    hysteresis_ticks: int = 3,
) -> Dict[str, EvaluationReport]:
    """Evaluate baselines and ML models on identical unseen test episodes via hysteresis state machine."""
    reports: Dict[str, EvaluationReport] = {}

    # Baseline 1: Static Threshold
    static_b = StaticThresholdBaseline()
    reports["static_threshold"] = evaluate_detector(
        predict_fn=lambda s: static_b.predict_sample(s).is_alert,
        episodes=test_episodes,
        horizon_sec=horizon_sec,
        min_effective_lead_time_sec=min_effective_lead_time_sec,
        reset_fn=static_b.reset,
    )

    # Baseline 2: Time to Exhaustion
    tte_b = TimeToExhaustionBaseline()
    reports["time_to_exhaustion"] = evaluate_detector(
        predict_fn=lambda s: tte_b.predict_sample(s, horizon_sec=horizon_sec).is_alert,
        episodes=test_episodes,
        horizon_sec=horizon_sec,
        min_effective_lead_time_sec=min_effective_lead_time_sec,
        reset_fn=tte_b.reset,
    )

    # ML Models through hysteresis state machine
    for name, artifact in artifacts.items():
        pipeline = RollingFeaturePipeline()
        streak = 0

        def _predict_with_hysteresis(sample: Dict[str, Any]) -> bool:
            nonlocal streak
            feat = pipeline.push_sample(sample)
            is_alert, _ = artifact.predict_sample(feat)
            if is_alert:
                streak += 1
            else:
                streak = 0
            return streak >= hysteresis_ticks

        def _reset_ml():
            nonlocal streak
            streak = 0
            pipeline.reset()

        reports[name] = evaluate_detector(
            predict_fn=_predict_with_hysteresis,
            episodes=test_episodes,
            horizon_sec=horizon_sec,
            min_effective_lead_time_sec=min_effective_lead_time_sec,
            reset_fn=_reset_ml,
        )

    # Print summary table
    print("\n" + "=" * 90)
    print(f" offGuard Benchmark Comparison on {len(test_episodes)} Test Episodes")
    print("=" * 90)
    print(
        f"{'MODEL / BASELINE':<25} | {'RECALL':<8} | {'EFF RECALL(>=60s)':<18} | "
        f"{'MEDIAN LEAD':<12} | {'FA / HOUR':<10}"
    )
    print("-" * 90)

    for name, rep in reports.items():
        print(
            f"{name:<25} | {rep.episode_recall:<8.2%} | {rep.effective_recall_ge_60s:<18.2%} | "
            f"{rep.lead_time_median_s:<10.1f} s | {rep.false_alarms_per_hour:<10.2f}"
        )
    print("=" * 90 + "\n")

    return reports

