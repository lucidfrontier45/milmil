import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from milmil.datasets import make_mil_data
from milmil.models import AlternatingOptimizationMIL, InstanceExpansionMIL
from milmil.types import LabeledBag, labeled_bags_to_arrays


def split_bags(
    bags: list[LabeledBag], test_ratio: float, seed: int
) -> tuple[list[LabeledBag], list[LabeledBag]]:
    n = len(bags)
    indices = np.arange(n)
    np.random.seed(seed)
    np.random.shuffle(indices)
    split_point = int(n * (1 - test_ratio))
    train_idx, test_idx = indices[:split_point], indices[split_point:]
    return [bags[i] for i in train_idx], [bags[i] for i in test_idx]


def evaluate_baseline(
    train_bags: list[LabeledBag], test_bags: list[LabeledBag], seed: int
) -> float:
    x_train = np.array([b.features.flatten() for b in train_bags])
    y_train = np.array([b.label for b in train_bags])

    x_test = np.array([b.features.flatten() for b in test_bags])
    y_test = np.array([b.label for b in test_bags])

    rf = RandomForestClassifier(random_state=seed, n_estimators=100)
    rf.fit(x_train, y_train)
    y_proba = rf.predict_proba(x_test)[:, 1]

    return average_precision_score(y_test, y_proba)


def evaluate_instance_expansion_mil(
    train_bags: list[LabeledBag], test_bags: list[LabeledBag], seed: int
) -> float:
    x_train, y_train, z_train = labeled_bags_to_arrays(train_bags)
    x_test, y_test, z_test = labeled_bags_to_arrays(test_bags)

    rf = RandomForestClassifier(random_state=seed, n_estimators=100)
    model = InstanceExpansionMIL(estimator=rf)
    model.fit(x_train, y_train, z_train)
    y_proba = model.predict_proba(x_test, z_test)

    return average_precision_score(y_test, y_proba)


def evaluate_alternating_optimization_mil(
    train_bags: list[LabeledBag], test_bags: list[LabeledBag], seed: int
) -> float:
    x_train, y_train, z_train = labeled_bags_to_arrays(train_bags)
    x_test, y_test, z_test = labeled_bags_to_arrays(test_bags)

    rf = RandomForestClassifier(random_state=seed, n_estimators=100)
    model = AlternatingOptimizationMIL(estimator=rf)
    model.fit(x_train, y_train, z_train)
    y_proba = model.predict_proba(x_test, z_test)

    return average_precision_score(y_test, y_proba)


def evaluate_single_seed(seed: int) -> tuple[int, float, float, float]:
    bags = make_mil_data(
        n_bags=1000,
        n_features=50,
        n_instances=(3, 3),
        random_state=seed,
        informative_ratio=0.1,
        redundant_ratio=0.3,
        n_clusters_per_class=5,
        class_sep=0.1,
        noise=0.2,
    )
    train_bags, test_bags = split_bags(bags, test_ratio=0.2, seed=seed)

    baseline_ap = evaluate_baseline(train_bags, test_bags, seed)
    exp_ap = evaluate_instance_expansion_mil(train_bags, test_bags, seed)
    alt_ap = evaluate_alternating_optimization_mil(train_bags, test_bags, seed)

    return seed, baseline_ap, exp_ap, alt_ap


def run_benchmark():
    seeds = [42, 123, 456, 789, 1024, 412, 588, 999, 2024, 3000]

    print("Benchmark: Baseline vs InstanceExpansionMIL vs AlternatingOptimizationMIL")
    print("Data: n_bags=1000, n_features=50, n_instances=(3, 3)")
    print("Split: 80/20 train/test")
    print("Metric: Average Precision")
    print("=" * 70)

    results = Parallel(n_jobs=-1)(delayed(evaluate_single_seed)(seed) for seed in seeds)

    baseline_scores = []
    exp_scores = []
    alt_scores = []
    for seed, baseline_ap, exp_ap, alt_ap in sorted(results, key=lambda x: x[0]):
        baseline_scores.append(baseline_ap)
        exp_scores.append(exp_ap)
        alt_scores.append(alt_ap)
        print(f"\nSeed {seed}:")
        print(f"  Baseline (Concat+RF):           AP = {baseline_ap:.4f}")
        print(f"  InstanceExpansionMIL+RF:        AP = {exp_ap:.4f}")
        print(f"  AlternatingOptimizationMIL+RF:  AP = {alt_ap:.4f}")

    print("\n" + "=" * 70)
    print("Summary (mean ± std):")
    bm = np.mean(baseline_scores)
    bs = np.std(baseline_scores)
    em = np.mean(exp_scores)
    es = np.std(exp_scores)
    am = np.mean(alt_scores)
    as_ = np.std(alt_scores)
    print(f"  Baseline:                       AP = {bm:.4f} ± {bs:.4f}")
    print(f"  InstanceExpansionMIL+RF:        AP = {em:.4f} ± {es:.4f}")
    print(f"  AlternatingOptimizationMIL+RF:  AP = {am:.4f} ± {as_:.4f}")


if __name__ == "__main__":
    run_benchmark()
