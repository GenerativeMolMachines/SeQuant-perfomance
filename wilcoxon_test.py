import os
import pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon


# CONFIG
FOLD_RESULTS_DIR = "results/fold_metrics"
TARGET_MODEL = "dcbilstm"

BASELINES = [
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
    "bilstm",
    "cbilstm",
]

CLASSIFICATION_DATASETS = [
    "antimic",
    "antidia",
    "antiinf",
    "antiox",
    "hemo",
    "nf",
    "sol",
]

REGRESSION_DATASETS = [
    "mic"
]

METRIC_CLASSIFICATION = "mcc"
METRIC_REGRESSION = "r2"

OUTPUT = "results/wilcoxon_results.csv"


def load_fold_values(dataset, encoder, metric):
    path = os.path.join(FOLD_RESULTS_DIR, f"{dataset}_{encoder}.csv")
    df = pd.read_csv(path)

    return df[metric].values


def compare_models(dataset, reference, baseline, metric):
    ref_values = load_fold_values(dataset, reference, metric)
    base_values = load_fold_values(dataset, baseline, metric)

    # paired test
    stat, p = wilcoxon(ref_values, base_values, alternative="two-sided")

    return {
        "dataset": dataset,
        "reference": reference,
        "baseline": baseline,
        "metric": metric,
        "reference_mean": ref_values.mean(),
        "baseline_mean": base_values.mean(),
        "difference": ref_values.mean() - base_values.mean(),
        "wilcoxon_statistic": stat,
        "p_value": p
    }


def main():
    results = []

    for dataset in CLASSIFICATION_DATASETS:
        print(f"Processing dataset {dataset}...")
        for baseline in tqdm(BASELINES):
            try:
                result = compare_models(dataset, TARGET_MODEL, baseline, METRIC_CLASSIFICATION)
                results.append(result)

            except FileNotFoundError:
                print(f"Missing: {dataset} {baseline}")

    for dataset in REGRESSION_DATASETS:
        print(f"Processing dataset {dataset}...")
        for baseline in tqdm(BASELINES):
            try:
                result = compare_models(dataset, TARGET_MODEL, baseline, METRIC_REGRESSION)
                results.append(result)

            except FileNotFoundError:
                print(f"Missing: {dataset} {baseline}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT, index=False)

    print(df)


if __name__ == "__main__":
    main()