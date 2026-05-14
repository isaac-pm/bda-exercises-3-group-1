import argparse
import csv
import math
import time
from glob import glob
from pathlib import Path

from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, col, desc, input_file_name, monotonically_increasing_id, udf
from pyspark.sql.types import DoubleType


def build_spark(
    app_name: str,
    master: str | None,
    driver_memory: str | None,
    executor_memory: str | None,
    shuffle_partitions: int | None,
    driver_max_result_size: str | None,
    offheap_enabled: bool,
    offheap_size: str | None,
):
    builder = SparkSession.builder.appName(app_name)
    if master:
        builder = builder.master(master)
    if driver_memory:
        builder = builder.config("spark.driver.memory", driver_memory)
    if executor_memory:
        builder = builder.config("spark.executor.memory", executor_memory)
    if shuffle_partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    if driver_max_result_size:
        builder = builder.config("spark.driver.maxResultSize", driver_max_result_size)
    if offheap_enabled:
        builder = builder.config("spark.memory.offHeap.enabled", "true")
        if offheap_size:
            builder = builder.config("spark.memory.offHeap.size", offheap_size)
    else:
        builder = builder.config("spark.memory.offHeap.enabled", "false")
    return builder.getOrCreate()


def resolve_paths(data_path: str):
    if any(token in data_path for token in ["*", "?", "["]):
        matches = sorted(glob(data_path))
        if matches:
            return matches

    path = Path(data_path)
    if path.exists() and path.is_dir():
        matches = sorted(str(item) for item in path.glob("*.csv"))
        if matches:
            return matches

    return data_path


def load_gearbox_data(spark: SparkSession, data_path: str):
    resolved = resolve_paths(data_path)
    df = spark.read.csv(resolved, header=False, inferSchema=True)
    sensor_cols = [f"sensor_{idx + 1}" for idx in range(len(df.columns))]
    df = df.toDF(*sensor_cols)
    df = df.withColumn("source_file", input_file_name())
    df = df.withColumn("point_id", monotonically_increasing_id())
    assembler = VectorAssembler(inputCols=sensor_cols, outputCol="features_raw")
    assembled = assembler.transform(df)
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withMean=True,
        withStd=True,
    )
    scaler_model = scaler.fit(assembled)
    scaled = scaler_model.transform(assembled)
    return (
        scaled.select("point_id", "source_file", *sensor_cols, "features"),
        sensor_cols,
    )


def compute_outliers(
    data_df,
    sensor_cols,
    k: int,
    seed: int,
    max_iter: int,
    viz_sample_size: int,
    viz_sample_seed: int,
):
    kmeans = KMeans(
        k=k,
        seed=seed,
        maxIter=max_iter,
        featuresCol="features",
        predictionCol="prediction",
    )
    start = time.perf_counter()
    model = kmeans.fit(data_df)

    centers = [center.tolist() for center in model.clusterCenters()]
    centers_broadcast = data_df.sparkSession.sparkContext.broadcast(centers)

    def distance_to_centroid(prediction, features_array):
        if prediction is None or features_array is None:
            return None
        center = centers_broadcast.value[int(prediction)]
        total = 0.0
        for idx, value in enumerate(features_array):
            diff = value - center[idx]
            total += diff * diff
        return math.sqrt(total)

    distance_udf = udf(distance_to_centroid, DoubleType())
    scored = model.transform(data_df).withColumn(
        "features_array", vector_to_array(col("features"))
    )
    scored = scored.withColumn(
        "distance", distance_udf(col("prediction"), col("features_array"))
    ).drop("features_array")
    scored = scored.persist(StorageLevel.MEMORY_AND_DISK)

    avg_distance = (
        scored.select(avg("distance").alias("avg_distance")).first()["avg_distance"]
    )

    top100 = scored.orderBy(desc("distance")).select("distance").limit(100)
    top100_rows = [row["distance"] for row in top100.collect()]
    anomaly_threshold = top100_rows[-1] if top100_rows else None
    total_count = scored.count()
    if anomaly_threshold is None or total_count == 0:
        anomaly_fraction = 0.0
    else:
        anomaly_count = scored.filter(col("distance") > anomaly_threshold).count()
        anomaly_fraction = (anomaly_count / total_count) * 100.0

    viz_rows = None
    if viz_sample_size and viz_sample_size > 0:
        sample_rows = (
            scored.select("point_id", "source_file", *sensor_cols, "prediction", "distance")
            .rdd.takeSample(False, viz_sample_size, viz_sample_seed)
        )
        viz_rows = [row.asDict() for row in sample_rows]

    top25 = (
        scored.orderBy(desc("distance"))
        # Global ranking by distance to own centroid (not per-cluster).
        .select("point_id", "source_file", "prediction", "distance", *sensor_cols)
        .limit(25)
    )

    top25_rows = [row.asDict() for row in top25.collect()]
    scored.unpersist()
    runtime = time.perf_counter() - start
    return runtime, top25_rows, avg_distance, anomaly_threshold, anomaly_fraction, viz_rows


def write_csv(rows, path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_txt(
    runtimes, scores, anomalies, outliers_by_k, path: Path, sensor_cols
):
    lines = [
        "Runtimes (seconds)",
        "k\truntime_seconds",
    ]
    for k, runtime in runtimes:
        lines.append(f"{k}\t{runtime:.3f}")

    lines.extend(["", "Average distance to centroid", "k\tavg_distance"])
    for k, avg_distance in scores:
        lines.append(f"{k}\t{avg_distance:.6f}")

    lines.extend(["", "Anomaly threshold and fraction", "k\tthreshold\tfraction_percent"])
    for k, threshold, fraction in anomalies:
        threshold_value = f"{threshold:.6f}" if threshold is not None else "NA"
        lines.append(f"{k}\t{threshold_value}\t{fraction:.2f}")

    for k, rows in outliers_by_k.items():
        lines.append("")
        lines.append(f"Top 25 outliers for k={k}")
        header_fields = [
            "point_id",
            "source_file",
            "prediction",
            "distance",
            *sensor_cols,
        ]
        lines.append("\t".join(header_fields))
        for row in rows:
            values = [
                row.get("point_id"),
                row.get("source_file"),
                row.get("prediction"),
                f"{row.get('distance', 0.0):.6f}",
            ] + [row.get(col_name) for col_name in sensor_cols]
            lines.append("\t".join(str(value) for value in values))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def parse_args():
    script_dir = Path(__file__).resolve().parent
    default_data_path = (
        script_dir / ".." / ".." / "data" / "problem_3" / "gearbox" / "*.csv"
    ).resolve()
    default_output_dir = (script_dir / "outputs").resolve()

    parser = argparse.ArgumentParser(
        description="Run KMeans (k=2..12) and report top-25 outliers per k."
    )
    parser.add_argument(
        "--data-path",
        default=str(default_data_path),
        help="Path or glob to gearbox CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir),
        help="Directory to write runtimes and outlier CSVs.",
    )
    parser.add_argument("--k-min", type=int, default=2, help="Minimum k value.")
    parser.add_argument("--k-max", type=int, default=12, help="Maximum k value.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=50, help="Max iterations.")
    parser.add_argument(
        "--master",
        default="local[*]",
        help="Spark master URL (default: local[*]).",
    )
    parser.add_argument(
        "--driver-memory",
        default="16g",
        help="Spark driver memory (e.g., 8g).",
    )
    parser.add_argument(
        "--executor-memory",
        default=None,
        help="Spark executor memory (e.g., 8g).",
    )
    parser.add_argument(
        "--shuffle-partitions",
        type=int,
        default=200,
        help="Spark SQL shuffle partitions.",
    )
    parser.add_argument(
        "--driver-max-result-size",
        default="2g",
        help="spark.driver.maxResultSize (e.g., 2g).",
    )
    parser.add_argument(
        "--offheap-enabled",
        default="true",
        choices=["true", "false"],
        help="Enable off-heap memory (true/false).",
    )
    parser.add_argument(
        "--offheap-size",
        default="4g",
        help="spark.memory.offHeap.size (e.g., 4g).",
    )
    parser.add_argument(
        "--repartition",
        type=int,
        default=None,
        help="Repartition the input DataFrame to this number of partitions.",
    )
    parser.add_argument(
        "--persist-level",
        default="DISK_ONLY",
        choices=[
            "NONE",
            "DISK_ONLY",
            "MEMORY_ONLY",
            "MEMORY_AND_DISK",
            "MEMORY_AND_DISK_SER",
        ],
        help="Storage level for the input DataFrame.",
    )
    parser.add_argument(
        "--viz-k",
        type=int,
        default=None,
        help="If set, export a scored sample for this k (defaults to k-min when --viz-output is set).",
    )
    parser.add_argument(
        "--viz-sample-size",
        type=int,
        default=5000,
        help="Rows to export for visualization (0 disables).",
    )
    parser.add_argument(
        "--viz-sample-seed",
        type=int,
        default=42,
        help="Seed for visualization sample.",
    )
    parser.add_argument(
        "--viz-output",
        default=str((script_dir / "outputs" / "scored_sample.csv").resolve()),
        help="CSV path for scored visualization sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    offheap_enabled = args.offheap_enabled.lower() == "true"
    spark = build_spark(
        "GearboxOutliers",
        master=args.master,
        driver_memory=args.driver_memory,
        executor_memory=args.executor_memory,
        shuffle_partitions=args.shuffle_partitions,
        driver_max_result_size=args.driver_max_result_size,
        offheap_enabled=offheap_enabled,
        offheap_size=args.offheap_size,
    )

    data_df, sensor_cols = load_gearbox_data(spark, args.data_path)
    if args.repartition:
        data_df = data_df.repartition(args.repartition)
    if args.persist_level != "NONE":
        storage_level = getattr(StorageLevel, args.persist_level)
        data_df = data_df.persist(storage_level)
        data_df.count()

    runtimes = []
    scores = []
    anomalies = []
    outliers_by_k = {}
    outlier_fields = ["point_id", "source_file", "prediction", "distance", *sensor_cols]

    output_dir = Path(args.output_dir)
    viz_output = Path(args.viz_output).resolve() if args.viz_output else None
    if viz_output is not None and args.viz_k is None:
        args.viz_k = args.k_min

    for k in range(args.k_min, args.k_max + 1):
        sample_size = 0
        if args.viz_k is not None and k == args.viz_k:
            sample_size = args.viz_sample_size
        (
            runtime,
            outliers,
            avg_distance,
            anomaly_threshold,
            anomaly_fraction,
            viz_rows,
        ) = compute_outliers(
            data_df,
            sensor_cols,
            k,
            args.seed,
            args.max_iter,
            sample_size,
            args.viz_sample_seed,
        )
        runtimes.append((k, runtime))
        scores.append((k, avg_distance))
        anomalies.append((k, anomaly_threshold, anomaly_fraction))
        if anomaly_threshold is None:
            print(f"k={k}: anomaly threshold=NA, anomaly fraction=0.00%")
        else:
            print(
                f"k={k}: anomaly threshold={anomaly_threshold:.6f}, "
                f"anomaly fraction={anomaly_fraction:.2f}%"
            )
        outliers_by_k[k] = outliers
        write_csv(
            outliers,
            output_dir / f"outliers_k{k}.csv",
            fieldnames=outlier_fields,
        )
        if viz_rows is not None and viz_output is not None:
            write_csv(
                viz_rows,
                viz_output,
                fieldnames=[
                    "point_id",
                    "source_file",
                    *sensor_cols,
                    "prediction",
                    "distance",
                ],
            )

    runtime_rows = [{"k": k, "runtime_seconds": runtime} for k, runtime in runtimes]
    write_csv(
        runtime_rows,
        output_dir / "runtimes.csv",
        fieldnames=["k", "runtime_seconds"],
    )
    score_rows = [{"k": k, "avg_distance": avg_distance} for k, avg_distance in scores]
    write_csv(
        score_rows,
        output_dir / "cluster_scores.csv",
        fieldnames=["k", "avg_distance"],
    )
    write_summary_txt(
        runtimes,
        scores,
        anomalies,
        outliers_by_k,
        output_dir / "summary.txt",
        sensor_cols,
    )

    spark.stop()


if __name__ == "__main__":
    main()
