from pyspark.sql import SparkSession
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Local split CSV path (inside container)")
    parser.add_argument("--out", required=True, help="Shared output base path (inside container)")
    parser.add_argument("--tag", required=True, help="Which node wrote this (master/w1/w2)")
    args = parser.parse_args()

    spark = SparkSession.builder.appName(f"IngestSplit-{args.tag}").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.csv(args.input, header=True, inferSchema=True)

    # Write each split into its own parquet folder to avoid overwrite conflicts
    out_path = f"{args.out}/ingested/{args.tag}"
    df.write.mode("overwrite").parquet(out_path)

    print(f"Wrote parquet split to: {out_path} (rows={df.count()})")
    spark.stop()

if __name__ == "__main__":
    main()
