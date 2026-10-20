"""Train candidate classifiers, track them in MLflow, and freeze the serving bundle."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from pulseguard.config import (
    CV_FOLDS,
    DISCLAIMER,
    EXPERIMENT_NAME,
    FIGURES_DIR,
    HIGH_RISK_THRESHOLD,
    MLFLOW_DIR,
    MODEL_VERSION,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from pulseguard.data.preprocess import load_clean_frame, prepare_model_frame
from pulseguard.features.builder import build_pipeline
from pulseguard.models.selection import choose_threshold, screening_utility
from pulseguard.viz.style import PALETTE, style_axes


def candidate_grids() -> dict:
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            {"model__C": [0.2, 1.0, 4.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [180, 320],
                "model__max_depth": [4, 8, None],
                "model__min_samples_leaf": [2, 4],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__max_iter": [180],
            },
        ),
    }


def _cv_metrics(estimator, features, target) -> dict:
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    probabilities = cross_val_predict(
        estimator,
        features,
        target,
        cv=splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(target, probabilities))
    recall = float(recall_score(target, predictions, zero_division=0))
    return {
        "cv_roc_auc": roc_auc,
        "cv_recall": recall,
        "cv_precision": float(precision_score(target, predictions, zero_division=0)),
        "cv_accuracy": float(accuracy_score(target, predictions)),
        "cv_f1": float(f1_score(target, predictions, zero_division=0)),
        "utility": screening_utility(roc_auc, recall),
        "oof_probability": probabilities,
    }


def _plot_comparison(rows: list[dict], path) -> None:
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(frame))
    width = 0.25
    ax.bar(x - width, frame["cv_roc_auc"], width, label="CV ROC-AUC", color=PALETTE["teal"])
    ax.bar(x, frame["cv_recall"], width, label="CV recall", color=PALETTE["coral"])
    ax.bar(x + width, frame["utility"], width, label="Screening utility", color=PALETTE["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(frame["name"], rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("PulseGuard candidate comparison on train/validation CV")
    ax.legend(frameon=False)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_roc(y_true, probability, auc, path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probability)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, color=PALETTE["teal"], lw=2.2, label=f"Held-out ROC-AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], ls="--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Held-out ROC curve")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_confusion(matrix, path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    image = ax.imshow(matrix, cmap="BrBG")
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title("Held-out confusion matrix at tuned threshold")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, int(matrix[row, col]), ha="center", va="center", color=PALETTE["ink"])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_importance(pipeline, path, features=None, target=None) -> None:
    model = pipeline.named_steps["model"]
    preprocess = pipeline.named_steps["preprocess"]
    if hasattr(model, "feature_importances_"):
        names = list(preprocess.get_feature_names_out())
        values = np.asarray(model.feature_importances_)
        title = "Impurity importance of encoded features"
        xlabel = "Relative contribution"
    elif hasattr(model, "coef_"):
        names = list(preprocess.get_feature_names_out())
        values = np.abs(np.asarray(model.coef_).ravel())
        title = "Absolute logistic coefficients"
        xlabel = "Absolute coefficient"
    else:
        from sklearn.inspection import permutation_importance

        sample = features.sample(n=min(160, len(features)), random_state=RANDOM_STATE)
        scored = permutation_importance(
            pipeline,
            sample,
            target.loc[sample.index],
            n_repeats=4,
            random_state=RANDOM_STATE,
            scoring="roc_auc",
            n_jobs=1,
        )
        names = list(sample.columns)
        values = np.asarray(scored.importances_mean)
        title = "Permutation importance on model inputs"
        xlabel = "Drop in ROC-AUC"
    order = np.argsort(values)[-12:]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.barh(np.array(names)[order], values[order], color=PALETTE["teal"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _risk_band(probability: float, decision_threshold: float) -> str:
    if probability >= HIGH_RISK_THRESHOLD:
        return "high"
    if probability >= decision_threshold:
        return "moderate"
    return "low"


def train_and_select() -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_clean_frame()
    features, target = prepare_model_frame(frame)
    sites = frame.loc[features.index, "source_site"]
    x_train, x_test, y_train, y_test, site_train, site_test = train_test_split(
        features,
        target,
        sites,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    candidate_rows = []
    run_index = []
    fitted = {}
    for name, (estimator, grid) in candidate_grids().items():
        pipeline = build_pipeline(estimator)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring="roc_auc",
            cv=splitter,
            n_jobs=-1,
            refit=True,
        )
        search.fit(x_train, y_train)
        metrics = _cv_metrics(search.best_estimator_, x_train, y_train)
        row = {
            "name": name,
            "best_params": search.best_params_,
            "search_best_roc_auc": float(search.best_score_),
            "cv_roc_auc": metrics["cv_roc_auc"],
            "cv_recall": metrics["cv_recall"],
            "cv_precision": metrics["cv_precision"],
            "cv_accuracy": metrics["cv_accuracy"],
            "cv_f1": metrics["cv_f1"],
            "utility": metrics["utility"],
        }
        candidate_rows.append(row)
        fitted[name] = {
            "estimator": search.best_estimator_,
            "oof_probability": metrics["oof_probability"],
        }
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({key: str(value) for key, value in search.best_params_.items()})
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("candidate", name)
            mlflow.log_param("selection_metric", "screening_utility")
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in row.items()
                    if key not in {"best_params", "name"}
                }
            )
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
            run_index.append({"name": name, "run_id": run.info.run_id, "utility": row["utility"]})
        print(
            f"{name}: utility={row['utility']:.4f} auc={row['cv_roc_auc']:.4f} "
            f"recall={row['cv_recall']:.4f} params={search.best_params_}"
        )

    candidate_rows.sort(key=lambda item: item["utility"], reverse=True)
    best_name = candidate_rows[0]["name"]
    best_estimator = fitted[best_name]["estimator"]
    decision_threshold = choose_threshold(y_train, fitted[best_name]["oof_probability"])
    test_probability = best_estimator.predict_proba(x_test)[:, 1]
    test_prediction = (test_probability >= decision_threshold).astype(int)
    test_metrics = {
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "f1": float(f1_score(y_test, test_prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "pr_auc": float(average_precision_score(y_test, test_probability)),
    }
    per_site = {}
    for site in sorted(site_test.unique()):
        mask = site_test == site
        if int(mask.sum()) < 12 or y_test[mask].nunique() < 2:
            per_site[site] = {"rows": int(mask.sum()), "roc_auc": None}
            continue
        per_site[site] = {
            "rows": int(mask.sum()),
            "roc_auc": float(roc_auc_score(y_test[mask], test_probability[mask])),
        }

    comparison_path = FIGURES_DIR / "model_comparison.png"
    roc_path = FIGURES_DIR / "roc_curve.png"
    confusion_path = FIGURES_DIR / "confusion_matrix.png"
    importance_path = FIGURES_DIR / "feature_importance.png"
    _plot_comparison(candidate_rows, comparison_path)
    _plot_roc(y_test, test_probability, test_metrics["roc_auc"], roc_path)
    _plot_confusion(confusion_matrix(y_test, test_prediction), confusion_path)
    _plot_importance(best_estimator, importance_path, x_train, y_train)

    with mlflow.start_run(run_name=f"selected-{best_name}"):
        mlflow.log_param("selected_model", best_name)
        mlflow.log_param("decision_threshold", decision_threshold)
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        for artifact in (comparison_path, roc_path, confusion_path, importance_path):
            if artifact.exists():
                mlflow.log_artifact(str(artifact), artifact_path="figures")

    card = {
        "model_name": best_name,
        "model_version": MODEL_VERSION,
        "random_state": RANDOM_STATE,
        "decision_threshold": decision_threshold,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "test_metrics": test_metrics,
        "candidates": [
            {key: value for key, value in row.items() if key != "best_params"}
            | {"best_params": {k: _jsonable(v) for k, v in row["best_params"].items()}}
            for row in candidate_rows
        ],
        "disclaimer": DISCLAIMER,
    }
    bundle = {"pipeline": best_estimator, "card": card}
    model_path = MODELS_DIR / "pulseguard_pipeline.joblib"
    joblib.dump(bundle, model_path)

    predictions = pd.DataFrame(
        {
            "source_site": site_test.to_numpy(),
            "y_true": y_test.to_numpy(),
            "y_probability": test_probability,
            "y_pred": test_prediction,
            "risk_band": [_risk_band(value, decision_threshold) for value in test_probability],
        }
    )
    predictions.to_csv(REPORTS_DIR / "heldout_predictions.csv", index=False)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "random_state": RANDOM_STATE,
        "selection_policy": "Maximize 0.70 * OOF ROC-AUC + 0.30 * OOF recall on the train split. Threshold tuned by F1.5 on OOF scores. Test split used once.",
        "rows": int(len(frame)),
        "positive_rate": float(target.mean()),
        "best_model": best_name,
        "decision_threshold": decision_threshold,
        "high_risk_threshold": HIGH_RISK_THRESHOLD,
        "candidates": card["candidates"],
        "test_metrics": test_metrics,
        "per_site_test_auc": per_site,
        "mlflow_runs": run_index,
        "model_path": str(model_path.relative_to(model_path.parents[1])),
        "disclaimer": DISCLAIMER,
    }
    (REPORTS_DIR / "experiment_summary.json").write_text(json.dumps(summary, indent=2))
    _write_model_card(card, summary)
    print(json.dumps({"best_model": best_name, "test_metrics": test_metrics}, indent=2))
    return summary


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write_model_card(card: dict, summary: dict) -> None:
    metrics = card["test_metrics"]
    lines = [
        "# PulseGuard model card",
        "",
        f"- Model: `{card['model_name']}`",
        f"- Version: `{card['model_version']}`",
        f"- Trained at: {card['trained_at']}",
        f"- Random seed: {card['random_state']}",
        f"- Decision threshold: {card['decision_threshold']:.3f}",
        f"- High-risk band: probability >= {card['high_risk_threshold']:.2f}",
        "",
        "## Held-out metrics",
        "",
        f"- Accuracy: {metrics['accuracy']:.3f}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        f"- ROC-AUC: {metrics['roc_auc']:.3f}",
        f"- PR-AUC: {metrics['pr_auc']:.3f}",
        "",
        "## Selection policy",
        "",
        summary["selection_policy"],
        "",
        "## Intended use",
        "",
        card["disclaimer"],
        "",
        "Inputs are the 13 UCI clinical fields plus an optional `source_site`.",
        "The saved object is a scikit-learn Pipeline, so scaling and encoding travel with the model.",
        "",
    ]
    (MODELS_DIR / "MODEL_CARD.md").write_text("\n".join(lines))


def main() -> None:
    train_and_select()


if __name__ == "__main__":
    main()
