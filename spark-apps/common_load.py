from __future__ import annotations

from typing import List, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.utils import AnalysisException


def _assert_paths_exist(spark: SparkSession, paths: List[str]) -> None:
    """
    Validate paths exist using Hadoop FS (works for local, docker volume, HDFS, s3a, etc.).
    Raises FileNotFoundError with a clear message if any are missing.
    """
    jvm = spark._jvm
    hconf = spark._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hconf)

    missing = []
    for p in paths:
        if not fs.exists(jvm.org.apache.hadoop.fs.Path(p)):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "These input paths do not exist or are not accessible from this Spark driver/executor:\n"
            + "\n".join(missing)
        )


# --- Ingestion-time loader (node-local CSV) ---
def load_local_split_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    """
    Use ONLY for ingestion, when you are inside the container that owns the file.
    """
    _assert_paths_exist(spark, [csv_path])
    return spark.read.csv(csv_path, header=True, inferSchema=True)


# --- Training-time loaders (shared Parquet) ---
def load_split_parquet(spark: SparkSession, base_path: str, split_tag: str) -> DataFrame:
    """
    Load one split from shared storage.
    split_tag: "master" | "w1" | "w2" (or whatever tags you used at ingestion)
    """
    path = f"{base_path}/ingested/{split_tag}"
    _assert_paths_exist(spark, [path])
    return spark.read.parquet(path)


def load_integrated_parquet(spark: SparkSession, base_path: str, split_tags: Optional[List[str]] = None) -> DataFrame:
    """
    Load and union all splits from shared storage.
    """
    if split_tags is None:
        split_tags = ["master", "w1", "w2"]

    paths = [f"{base_path}/ingested/{t}" for t in split_tags]
    _assert_paths_exist(spark, paths)

    dfs = [spark.read.parquet(p) for p in paths]
    out = dfs[0]
    for d in dfs[1:]:
        out = out.unionByName(d)

    return out
