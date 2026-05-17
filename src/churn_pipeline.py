from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer churn prediction pipeline")
    parser.add_argument(
        "--data",
        default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        help="Path to the Telco Customer Churn CSV file.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def ensure_directories(base_dir: Path) -> Dict[str, Path]:
    output_root = base_dir / "outputs"
    paths = {
        "figures": output_root / "figures",
        "metrics": output_root / "metrics",
        "models": output_root / "models",
    }
    for path in [base_dir / "data", *paths.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. Download the Kaggle Telco churn CSV "
            "and place it in the data directory."
        )
    return pd.read_csv(csv_path)


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data.columns = [column.strip() for column in data.columns]
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["ChurnFlag"] = data["Churn"].map({"No": 0, "Yes": 1})

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"],
    )
    active_service_flags = []
    for column in SERVICE_COLUMNS:
        service_flag = data[column].astype(str).str.lower().isin(["yes", "fiber optic", "dsl"])
        active_service_flags.append(service_flag.astype(int))
    data["ServiceCount"] = np.sum(active_service_flags, axis=0)
    data["AvgMonthlySpend"] = np.where(
        data["tenure"] > 0,
        data["TotalCharges"] / data["tenure"],
        data["MonthlyCharges"],
    )
    data["IsLongTermCustomer"] = np.where(data["tenure"] >= 24, "Yes", "No")
    data["ChargesPerService"] = np.where(
        data["ServiceCount"] > 0,
        data["MonthlyCharges"] / data["ServiceCount"],
        data["MonthlyCharges"],
    )

    # Missing TotalCharges values correspond to new customers with zero tenure in this dataset.
    data["TotalCharges"] = data["TotalCharges"].fillna(0.0)
    data["TenureGroup"] = data["TenureGroup"].astype(str)
    return data


def save_distribution_plots(df: pd.DataFrame, figure_dir: Path) -> None:
    numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges", "ServiceCount", "AvgMonthlySpend"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for index, column in enumerate(numeric_columns):
        axes[index].hist(df[column], bins=30, color="#1f77b4", edgecolor="white", alpha=0.85)
        axes[index].set_title(f"{column} Distribution")
        axes[index].set_xlabel(column)
        axes[index].set_ylabel("Count")

    axes[-1].axis("off")
    fig.suptitle("Customer Distribution Plots", fontsize=16)
    fig.tight_layout()
    fig.savefig(figure_dir / "distribution_plots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, figure_dir: Path) -> None:
    corr_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "ServiceCount",
        "AvgMonthlySpend",
        "ChargesPerService",
        "ChurnFlag",
    ]
    corr = df[corr_columns].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    matrix = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Correlation Heatmap")

    for row in range(corr.shape[0]):
        for col in range(corr.shape[1]):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(matrix, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(figure_dir / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_segment_analysis(df: pd.DataFrame, figure_dir: Path) -> pd.DataFrame:
    segment_columns = ["Contract", "InternetService", "PaymentMethod", "TenureGroup"]
    summary_frames = []
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for axis, column in zip(axes, segment_columns):
        summary = (
            df.groupby(column, dropna=False)["ChurnFlag"]
            .agg(churn_rate="mean", customers="count")
            .sort_values("churn_rate", ascending=False)
            .reset_index()
        )
        summary["segment"] = column
        summary_frames.append(summary)

        axis.bar(summary[column].astype(str), summary["churn_rate"], color="#d62728", alpha=0.85)
        axis.set_title(f"Churn Rate by {column}")
        axis.set_xlabel(column)
        axis.set_ylabel("Churn Rate")
        axis.tick_params(axis="x", rotation=25)
        axis.set_ylim(0, min(1.0, summary["churn_rate"].max() * 1.25 + 0.05))

    fig.tight_layout()
    fig.savefig(figure_dir / "churn_by_segment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    segment_summary = pd.concat(summary_frames, ignore_index=True)
    return segment_summary


def get_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    target = "ChurnFlag"
    drop_columns = ["customerID", "Churn", target]
    features = [column for column in df.columns if column not in drop_columns]
    numeric_features = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in features if column not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def model_registry(random_state: int) -> Dict[str, Tuple[object, Dict[str, List[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, solver="liblinear", random_state=random_state),
            {
                "classifier__C": [0.1, 1.0, 5.0],
                "classifier__penalty": ["l1", "l2"],
                "smote__k_neighbors": [3, 5],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 8, 16],
                "classifier__min_samples_split": [2, 5],
                "smote__k_neighbors": [3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
                n_estimators=300,
                learning_rate=0.05,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [3, 5, 7],
                "classifier__learning_rate": [0.03, 0.05, 0.1],
                "classifier__subsample": [0.8, 1.0],
                "classifier__colsample_bytree": [0.8, 1.0],
                "smote__k_neighbors": [3, 5],
            },
        ),
    }


def evaluate_model(
    model_name: str,
    estimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    figure_dir: Path,
) -> Dict[str, float]:
    y_pred = estimator.predict(x_test)
    y_score = estimator.predict_proba(x_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score),
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{model_name} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{slugify(model_name)}_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}", color="#2ca02c")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"{model_name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figure_dir / f"{slugify(model_name)}_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics


def plot_model_comparison(results_df: pd.DataFrame, figure_dir: Path) -> None:
    metric_columns = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    x = np.arange(len(results_df))
    width = 0.16

    fig, ax = plt.subplots(figsize=(13, 7))
    for index, metric in enumerate(metric_columns):
        ax.bar(x + index * width, results_df[metric], width=width, label=metric)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df["model"], rotation=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "model_performance_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def extract_feature_importance(best_estimator, feature_names: np.ndarray) -> pd.DataFrame:
    classifier = best_estimator.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
    else:
        raise ValueError("Selected model does not expose feature importance or coefficients.")

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def plot_feature_importance(importance_df: pd.DataFrame, figure_dir: Path) -> None:
    top_features = importance_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_features["feature"], top_features["importance"], color="#9467bd")
    ax.set_title("Top 20 Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(figure_dir / "feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_business_insights(segment_summary: pd.DataFrame, importance_df: pd.DataFrame, metrics_df: pd.DataFrame, metrics_dir: Path) -> None:
    top_segments = segment_summary.sort_values("churn_rate", ascending=False).head(10)
    top_features = importance_df.head(10)["feature"].tolist()
    best_model_row = metrics_df.sort_values(["roc_auc", "recall", "accuracy"], ascending=False).iloc[0]

    lines = [
        f"Best performing model: {best_model_row['model']}",
        f"Accuracy: {best_model_row['accuracy']:.3f}",
        f"ROC-AUC: {best_model_row['roc_auc']:.3f}",
        "",
        "High-risk customer segments:",
    ]
    for _, row in top_segments.iterrows():
        segment_value = row[row["segment"]]
        lines.append(
            f"- {row['segment']} = {segment_value}: churn rate {row['churn_rate']:.2%} across {int(row['customers'])} customers"
        )

    lines.extend(
        [
            "",
            "Most influential features:",
            *[f"- {feature}" for feature in top_features],
            "",
            "Suggested retention actions:",
            "- Offer proactive discounts or loyalty bundles to month-to-month customers before renewal windows.",
            "- Prioritize onboarding and service-quality outreach for early-tenure customers.",
            "- Bundle security, backup, or tech-support add-ons for customers with lower service adoption.",
            "- Review pricing and support experience for high-charge internet customers, especially fiber optic segments.",
        ]
    )

    (metrics_dir / "business_insights.txt").write_text("\n".join(lines), encoding="utf-8")


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_paths = ensure_directories(project_root)

    raw_df = load_data(project_root / args.data)
    df = clean_and_engineer_features(raw_df)

    save_distribution_plots(df, output_paths["figures"])
    save_correlation_heatmap(df, output_paths["figures"])
    segment_summary = save_segment_analysis(df, output_paths["figures"])
    segment_summary.to_csv(output_paths["metrics"] / "segment_churn_summary.csv", index=False)

    numeric_features, categorical_features = get_feature_lists(df)
    x = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    y = df["ChurnFlag"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    registry = model_registry(args.random_state)

    all_results = []
    best_estimators = {}
    report_payload = {}

    for model_name, (classifier, parameter_grid) in registry.items():
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("smote", SMOTE(random_state=args.random_state)),
                ("classifier", classifier),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(x_train, y_train)

        best_estimators[model_name] = search.best_estimator_
        results = evaluate_model(model_name, search.best_estimator_, x_test, y_test, output_paths["figures"])
        results["best_params"] = json.dumps(search.best_params_)
        results["cv_best_roc_auc"] = search.best_score_
        all_results.append(results)

        report_payload[model_name] = classification_report(
            y_test,
            search.best_estimator_.predict(x_test),
            target_names=["No Churn", "Churn"],
            output_dict=True,
        )

    metrics_df = pd.DataFrame(all_results).sort_values(
        ["roc_auc", "recall", "accuracy"], ascending=False
    ).reset_index(drop=True)
    metrics_df.to_csv(output_paths["metrics"] / "model_metrics.csv", index=False)
    (output_paths["metrics"] / "classification_reports.json").write_text(
        json.dumps(report_payload, indent=2),
        encoding="utf-8",
    )

    plot_model_comparison(metrics_df, output_paths["figures"])

    best_model_name = metrics_df.iloc[0]["model"]
    best_estimator = best_estimators[best_model_name]
    feature_names = best_estimator.named_steps["preprocessor"].get_feature_names_out()
    importance_df = extract_feature_importance(best_estimator, feature_names)
    importance_df.to_csv(output_paths["metrics"] / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df, output_paths["figures"])

    model_summary = {
        "best_model": best_model_name,
        "best_model_metrics": {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in metrics_df.iloc[0].to_dict().items()
        },
        "data_path": str(project_root / args.data),
        "random_state": args.random_state,
        "test_size": args.test_size,
    }
    (output_paths["models"] / "best_model_summary.json").write_text(
        json.dumps(model_summary, indent=2),
        encoding="utf-8",
    )

    save_business_insights(segment_summary, importance_df, metrics_df, output_paths["metrics"])

    print("Pipeline completed successfully.")
    print(f"Best model: {best_model_name}")
    print(metrics_df[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
