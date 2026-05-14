import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

try:
    from umap import UMAP
except ImportError:
    UMAP = None


def load_scored_sample(sample_path: Path):
    return pd.read_csv(sample_path)


def plot_pca(sample_df: pd.DataFrame, output_path: Path, outlier_count: int):
    non_feature_cols = {"point_id", "source_file", "prediction", "distance"}
    sensor_cols = [col for col in sample_df.columns if col not in non_feature_cols]
    if not sensor_cols:
        raise ValueError("No sensor columns found in scored sample.")

    if "prediction" not in sample_df.columns or "distance" not in sample_df.columns:
        raise ValueError("Sample CSV must contain prediction and distance columns.")

    features = sample_df[sensor_cols].values
    if UMAP is None:
        # UMAP gives better separation for non-linear sensor data, but is optional.
        reducer = PCA(n_components=2, random_state=0)
    else:
        reducer = UMAP(n_components=2, random_state=0)
    coords = reducer.fit_transform(features)

    labels = sample_df["prediction"].values
    distances = sample_df["distance"].values
    outlier_idx = distances.argsort()[-outlier_count:]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        s=10,
        alpha=0.7,
        linewidths=0,
    )
    plt.scatter(
        coords[outlier_idx, 0],
        coords[outlier_idx, 1],
        facecolors="none",
        edgecolors="black",
        s=40,
        linewidths=0.7,
        label=f"Top {outlier_count} outliers",
    )
    title = "Gearbox Scored Sample UMAP (2D)" if UMAP is not None else "Gearbox Scored Sample PCA (2D)"
    plt.title(title)
    plt.xlabel("UMAP-1" if UMAP is not None else "PC1")
    plt.ylabel("UMAP-2" if UMAP is not None else "PC2")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_sample_path = (
        script_dir / "outputs" / "scored_sample.csv"
    ).resolve()
    default_output_path = (script_dir / "outputs" / "gearbox_pca.png").resolve()

    parser = argparse.ArgumentParser(
        description="Plot PCA scatter from scored sample CSV."
    )
    parser.add_argument(
        "--sample-path",
        default=str(default_sample_path),
        help="Path to scored sample CSV from gearbox_outliers.py.",
    )
    parser.add_argument(
        "--outlier-count",
        type=int,
        default=25,
        help="Number of outliers to highlight.",
    )
    parser.add_argument(
        "--output-path",
        default=str(default_output_path),
        help="Path to save PCA plot image.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sample_df = load_scored_sample(Path(args.sample_path))
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_pca(sample_df, output_path, outlier_count=args.outlier_count)


if __name__ == "__main__":
    main()
