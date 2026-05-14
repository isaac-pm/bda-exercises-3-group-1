import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_scores(scores_path: Path):
    rows = []
    with scores_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append((int(row["k"]), float(row["avg_distance"])))
            except (KeyError, ValueError):
                continue
    return sorted(rows, key=lambda item: item[0])


def plot_scores(scores, output_path: Path):
    ks = [k for k, _ in scores]
    distances = [distance for _, distance in scores]
    plt.figure(figsize=(7, 5))
    plt.plot(ks, distances, marker="o", linewidth=1.5)
    plt.title("Average Distance to Centroid vs. k")
    plt.xlabel("k")
    plt.ylabel("Average distance")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_scores = (script_dir / "outputs" / "cluster_scores.csv").resolve()
    default_output = (script_dir / "outputs" / "cluster_scores.png").resolve()
    parser = argparse.ArgumentParser(
        description="Plot average distance vs. k from cluster_scores.csv."
    )
    parser.add_argument(
        "--scores-path",
        default=str(default_scores),
        help="Path to cluster_scores.csv.",
    )
    parser.add_argument(
        "--output-path",
        default=str(default_output),
        help="Path to save the elbow plot image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    scores_path = Path(args.scores_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = read_scores(scores_path)
    if not scores:
        raise ValueError(f"No scores found in {scores_path}")
    plot_scores(scores, output_path)


if __name__ == "__main__":
    main()
