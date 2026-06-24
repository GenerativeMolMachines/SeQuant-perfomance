import os
import numpy as np
import pandas as pd

from tqdm import tqdm

from scipy.stats import sem

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import (
    StratifiedKFold,
    KFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBRegressor


# CONFIG
RANDOM_STATE = 42
N_SPLITS = 5

DATASETS_CLASSIFICATION = [
    "antimic",
    "antidia",
    "antiinf",
    "antiox",
    "hemo",
    "nf",
    "sol",
]

DATASETS_REGRESSION = [
    "mic"
]

BENCHMARK_ENCODERS = [
    "random92",
    "scrambled92",
    "one_hot",
    "threemers",
    "blosum62",
    "protbert",
    "prott5",
    "esmc",
    "ankh_large",
    "pepmlm",
    "peptideclm",
]

SEQUANT_ENCODERS = [
    "bilstm",
    "cbilstm",
    "dcbilstm",
]

ALL_ENCODERS = BENCHMARK_ENCODERS + SEQUANT_ENCODERS


ENCODED_DIR = "data/encoded"
SEQUANT_DIR = "data/embeddings_SeQuant"
OUTPUT_DIR = "results"
FOLD_RESULTS_DIR = os.path.join(OUTPUT_DIR, "fold_metrics")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FOLD_RESULTS_DIR, exist_ok=True)

np.random.seed(RANDOM_STATE)


# File loading
def load_embedding_dataset(dataset, encoder):
    if encoder in SEQUANT_ENCODERS:
        path = os.path.join(SEQUANT_DIR, f"{dataset}_{encoder}.csv")
    else:
        path = os.path.join(ENCODED_DIR, f"{dataset}_{encoder}.csv")

    return pd.read_csv(path)


# Preprocessing
def scale_features(df, target_col):

    features = df.drop(columns=["seq", target_col])

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled, columns=features.columns, index=df.index)

    scaled_df["seq"] = df["seq"]
    scaled_df[target_col] = df[target_col]

    return scaled_df


# Classification
def evaluate_classification(dataset, encoder):

    print(
        f"\n[CLASSIFICATION] "
        f"{dataset} | {encoder}"
    )

    df = load_embedding_dataset(dataset, encoder)
    df = scale_features(df, target_col="label")

    X = df.drop(columns=["seq", "label"])
    y = df["label"]

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    fold_metrics = []

    for train_idx, test_idx in tqdm(cv.split(X, y)):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fold_metrics.append({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred),
            "mcc": matthews_corrcoef(y_test, y_pred),
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        })

    save_fold_metrics(dataset, encoder, fold_metrics)

    return summarize_metrics(dataset, encoder, fold_metrics)


# Regression
def evaluate_regression(dataset, encoder):

    print(
        f"\n[REGRESSION] "
        f"{dataset} | {encoder}"
    )

    df = load_embedding_dataset(dataset,encoder)
    df = scale_features(df, target_col="label")

    X = df.drop(columns=["seq", "label"])
    y = df["label"]

    cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    model = XGBRegressor(random_state=RANDOM_STATE)

    fold_metrics = []

    for train_idx, test_idx in tqdm(cv.split(X)):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)

        fold_metrics.append({

            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mse,
            "rmse": np.sqrt(mse),
            "r2": r2_score(y_test, y_pred)
        })

    save_fold_metrics(dataset, encoder, fold_metrics)

    return summarize_metrics(dataset, encoder, fold_metrics)


# Metric aggregation
def save_fold_metrics(dataset, encoder, fold_metrics):

    fold_df = pd.DataFrame(fold_metrics)

    fold_df.insert(0,"fold", np.arange(len(fold_df)))
    fold_df.insert(0,"encoder", encoder)
    fold_df.insert(0,"dataset", dataset)

    fold_df.to_csv(os.path.join(FOLD_RESULTS_DIR, f"{dataset}_{encoder}.csv"), index=False)


def summarize_metrics(dataset, encoder, fold_metrics):

    summary = {
        "dataset": dataset,
        "encoder": encoder
    }

    for metric in fold_metrics[0]:
        values = [fold[metric] for fold in fold_metrics]
        summary[f"mean_{metric}"] = np.mean(values)
        summary[f"stderr_{metric}"] = sem(values)

    return summary


def main():

    classification_results = []

    for dataset in DATASETS_CLASSIFICATION:
        for encoder in tqdm(ALL_ENCODERS):
            try:
                result = evaluate_classification(dataset, encoder)
                classification_results.append(result)

            except FileNotFoundError:
                print(f"Missing file: {dataset}_{encoder}")

    classification_df = pd.DataFrame(classification_results)

    classification_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "classification_metrics.csv"
        ),
        index=False
    )

    print("\nClassification benchmarking complete")

    regression_results = []

    for dataset in DATASETS_REGRESSION:
        for encoder in tqdm(ALL_ENCODERS):
            try:
                result = evaluate_regression(dataset, encoder)
                regression_results.append(result)

            except FileNotFoundError:
                print(f"Missing file: {dataset}_{encoder}")

    regression_df = pd.DataFrame(regression_results)

    regression_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "regression_metrics.csv"
        ),
        index=False
    )

    print("\nRegression benchmarking complete")


if __name__ == "__main__":
    main()