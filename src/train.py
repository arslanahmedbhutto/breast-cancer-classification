"""Compare classifiers on the WDBC dataset and save the best one.

Usage:
    python -m src.train

Every model is wrapped in a Pipeline with its scaler so scaling is fit on
training folds only (no leakage from the test set). Models are ranked by
mean 5-fold cross-validated F1 on the training split; the held-out 20% test
split is only used to report final numbers.
"""

import json
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import FEATURES, LABELS, ROOT, load_data

SEED = 42
MODEL_PATH = ROOT / "models" / "model.joblib"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Reference palette (light mode): categorical slots 1-3, text inks, surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def candidate_models() -> dict:
    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    return {
        "Logistic Regression": scaled(LogisticRegression(max_iter=5000, random_state=SEED)),
        "SVM (RBF)": scaled(CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED), ensemble=False)),
        "K-Nearest Neighbors": scaled(KNeighborsClassifier(n_neighbors=7)),
        "Naive Bayes": scaled(GaussianNB()),
        "MLP": scaled(MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=SEED)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=5, random_state=SEED)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, random_state=SEED)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=SEED)),
    }


def test_metrics(model, X, y) -> dict:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_comparison(results: pd.DataFrame):
    df = results.sort_values("cv_f1_mean")
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.barh(df.index, df["cv_f1_mean"], height=0.55, color=SERIES[0],
            xerr=df["cv_f1_std"], error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
    for i, v in enumerate(df["cv_f1_mean"]):
        ax.text(v + df["cv_f1_std"].iloc[i] + 0.004, i, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0.8, 1.0)
    ax.set_xlabel("F1 (malignant), 5-fold CV mean ± std")
    ax.set_title("Model comparison", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_roc(fitted: dict, top: list[str], X_test, y_test):
    fig, ax = plt.subplots(figsize=(5.5, 5), facecolor=SURFACE)
    _style(ax)
    for name, color in zip(top, SERIES):
        proba = fitted[name].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, color=color, linewidth=2, drawstyle="steps-post",
                label=f"{name} (AUC {roc_auc_score(y_test, proba):.3f})")
    # Every model is near-perfect, so zoom into the top-left corner where they differ.
    ax.set_xlim(-0.01, 0.3)
    ax.set_ylim(0.8, 1.005)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (malignant recall)")
    ax.set_title("ROC curves, top 3 models (test set, zoomed)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    ax.grid(color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(model, name: str, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.5, 4), facecolor=SURFACE)
    _style(ax)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#104281"]))
    names = [LABELS[0].title(), LABELS[1].title()]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, v, ha="center", va="center", fontsize=14,
                color="white" if v > cm.max() / 2 else INK)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {name}", loc="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def write_markdown(results: pd.DataFrame, best: str):
    cols = ["cv_f1_mean", "cv_roc_auc_mean", "test_accuracy", "test_precision",
            "test_recall", "test_f1", "test_roc_auc"]
    header = ["Model", "CV F1", "CV ROC-AUC", "Test Acc", "Test Precision",
              "Test Recall", "Test F1", "Test ROC-AUC"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in results[cols].iterrows():
        label = f"**{name}**" if name == best else name
        lines.append("| " + " | ".join([label] + [f"{row[c]:.3f}" for c in cols]) + " |")
    (REPORTS / "model_comparison.md").write_text(
        "# Model comparison\n\n"
        "Ranked by mean 5-fold CV F1 on the training split (80%). "
        "Test columns are on the held-out 20% split (114 samples). "
        "Positive class = malignant.\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, fitted = {}, {}
    for name, model in candidate_models().items():
        scores = cross_validate(model, X_train, y_train, cv=cv, scoring=["f1", "roc_auc"])
        model.fit(X_train, y_train)
        fitted[name] = model
        rows[name] = {
            "cv_f1_mean": scores["test_f1"].mean(),
            "cv_f1_std": scores["test_f1"].std(),
            "cv_roc_auc_mean": scores["test_roc_auc"].mean(),
            **{f"test_{k}": v for k, v in test_metrics(model, X_test, y_test).items()},
        }
        print(f"{name:<22} CV F1 {rows[name]['cv_f1_mean']:.3f}  test F1 {rows[name]['test_f1']:.3f}")

    results = pd.DataFrame.from_dict(rows, orient="index").sort_values(
        ["cv_f1_mean", "cv_roc_auc_mean"], ascending=False
    )
    best = results.index[0]
    print(f"\nBest model: {best}")

    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    results.round(4).to_json(REPORTS / "metrics.json", orient="index", indent=2)
    write_markdown(results, best)
    plot_comparison(results)
    plot_roc(fitted, list(results.index[:3]), X_test, y_test)
    plot_confusion(fitted[best], best, X_test, y_test)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(
        {
            "pipeline": fitted[best],
            "model_name": best,
            "features": FEATURES,
            "metrics": {k: round(float(v), 4) for k, v in results.loc[best].items()},
            "sklearn_version": sklearn.__version__,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        MODEL_PATH,
    )
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
