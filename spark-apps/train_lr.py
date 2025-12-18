import argparse, time, json
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from common_load import load_split_parquet, load_integrated_parquet
from common_preprocess import preprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["integrated", "split"], required=True)

    # For split mode: which ingested parquet to use (master / w1 / w2)
    p.add_argument("--split_tag", default="master")

    # Shared base path where ingested parquet lives
    p.add_argument("--shared_base", default="/opt/spark-shared")

    p.add_argument("--sample_frac", type=float, default=1.0)
    p.add_argument("--out", default="/opt/spark-apps/results/lr_result.json")
    args = p.parse_args()

    spark = SparkSession.builder.appName("Train-LR").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load data
    if args.mode == "integrated":
        df = load_integrated_parquet(spark, args.shared_base)
    else:
        df = load_split_parquet(spark, args.shared_base, args.split_tag)

    if args.sample_frac < 1.0:
        df = df.sample(False, args.sample_frac, seed=42)

    data = preprocess(df, repartition_n=12).cache()
    data.count()  # materialize cache

    train, test = data.randomSplit([0.8, 0.2], seed=42)

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label_hit",
        maxIter=50
    )

    # -------------------
    # Train model
    # -------------------
    t0 = time.time()
    model = lr.fit(train)
    train_time = time.time() - t0

    preds = model.transform(test)

    # -------------------
    # Evaluators
    # -------------------
    acc_eval = MulticlassClassificationEvaluator(
        labelCol="label_hit",
        predictionCol="prediction",
        metricName="accuracy"
    )

    precision_eval = MulticlassClassificationEvaluator(
        labelCol="label_hit",
        predictionCol="prediction",
        metricName="weightedPrecision"
    )

    f1_eval = MulticlassClassificationEvaluator(
        labelCol="label_hit",
        predictionCol="prediction",
        metricName="f1"
    )

    accuracy = acc_eval.evaluate(preds)
    precision = precision_eval.evaluate(preds)
    f1_score = f1_eval.evaluate(preds)

    # -------------------
    # Results
    # -------------------
    result = {
        "algorithm": "LogisticRegression",
        "mode": args.mode,
        "split_tag": args.split_tag if args.mode == "split" else None,
        "rows_total": data.count(),
        "train_time_sec": round(train_time, 3),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "f1_score": round(f1_score, 4)
    }

    print("RESULT:", result)

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    spark.stop()


if __name__ == "__main__":
    main()
