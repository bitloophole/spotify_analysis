from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, Word2Vec, VectorAssembler
)
from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import (
    RegressionEvaluator, BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)
import time

# 0. Setup
spark = SparkSession.builder \
    .appName("SpotifySongPopularity") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")


# 1. Load dataset
data_path = "/opt/spark-apps/data/spotify_songs.csv" 
df = spark.read.csv(data_path, header=True, inferSchema=True)
df.printSchema()
df.show(5, truncate=False)

# 2. Inspect relevant columns
# From Kaggle description: we expect columns like 'track_name', 'artist_name', 'popularity', 
# and audio features such as 'danceability', 'energy', 'tempo', etc.
# Check columns:
print(df.columns)

# 3. Filter & clean
df = df.dropna(subset=["track_name", "track_popularity"])
df = df.withColumn("track_popularity", col("track_popularity").cast("int"))

# Create binary hit label (e.g., popularity >= 70)
HIT_THRESHOLD = 70
df = df.withColumn(
    "label_hit",
    when(col("track_popularity") >= HIT_THRESHOLD, 1).otherwise(0)
)

# -----------------------------------------
# CAST AUDIO FEATURES TO DOUBLE
# -----------------------------------------
audio_features = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms"
]

# Keep only existing columns
audio_features = [c for c in audio_features if c in df.columns]

for c in audio_features:
    df = df.withColumn(c, col(c).cast("double"))

# (Optional) Drop rows where any audio feature is null after casting
df = df.dropna(subset=audio_features)

df.printSchema()


# 4. Text preprocessing on track titles + keyword flags
df = df.withColumn(
    "title_clean",
    lower(regexp_replace(col("track_name"), "[^a-zA-Z0-9\\s]", " "))
)

df = df.withColumn(
    "is_remix",
    when(col("title_clean").contains("remix"), 1.0).otherwise(0.0)
)

df = df.withColumn(
    "is_live",
    when(col("title_clean").contains("live"), 1.0).otherwise(0.0)
)

# If there is a genre column (check df.columns). Suppose column 'genre' exists.
if "genre" in df.columns:
    df = df.withColumn(
        "genre_clean",
        lower(regexp_replace(col("genre"), "[^a-zA-Z0-9\\s]", " "))
    )
else:
    # If no genre: create blank column
    df = df.withColumn("genre_clean", lower(regexp_replace(col("track_name"), "[^a-zA-Z0-9\\s]", " ")))

# 5. NLP pipeline
title_tokenizer = Tokenizer(inputCol="title_clean", outputCol="title_tokens")
title_stop = StopWordsRemover(inputCol="title_tokens", outputCol="title_tokens_clean")
title_hashing_tf = HashingTF(inputCol="title_tokens_clean", outputCol="title_tf", numFeatures=2**12)
title_idf = IDF(inputCol="title_tf", outputCol="title_tfidf")
title_w2v = Word2Vec(inputCol="title_tokens_clean", outputCol="title_w2v", vectorSize=100, minCount=5)

genre_tokenizer = Tokenizer(inputCol="genre_clean", outputCol="genre_tokens")
genre_hashing_tf = HashingTF(inputCol="genre_tokens", outputCol="genre_tf", numFeatures=2**10)
genre_idf = IDF(inputCol="genre_tf", outputCol="genre_tfidf")

# # Select audio features you have in dataset, for example:
# audio_features = []
# for feat in ["danceability", "energy", "loudness", "tempo", "valence", "acousticness", "instrumentalness"]:
#     if feat in df.columns:
#         audio_features.append(feat)
#     else:
#         print(f"Warning: {feat} not in dataset columns")

assembler_inputs = ["title_tfidf", "title_w2v", "genre_tfidf", "is_remix", "is_live"] + audio_features

feature_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

preprocess_pipeline = Pipeline(stages=[
    title_tokenizer, title_stop, title_hashing_tf, title_idf, title_w2v,
    genre_tokenizer, genre_hashing_tf, genre_idf,
    feature_assembler
])

preprocess_model = preprocess_pipeline.fit(df)
data_prepared = preprocess_model.transform(df)
data_prepared.select("track_name", "track_popularity", "label_hit", "features").show(5, truncate=False)

# 6. Split into train/test
train, test = data_prepared.randomSplit([0.8, 0.2], seed=42)
print("Train count:", train.count(), "Test count:", test.count())

# 7. Regression: popularity
reg_evaluator_rmse = RegressionEvaluator(labelCol="track_popularity", predictionCol="prediction", metricName="rmse")
reg_evaluator_mae  = RegressionEvaluator(labelCol="track_popularity", predictionCol="prediction", metricName="mae")

results = []

def train_and_eval_regressor(model, name):
    start = time.time()
    m = model.fit(train)
    duration = time.time() - start
    preds = m.transform(test)
    rmse = reg_evaluator_rmse.evaluate(preds)
    mae  = reg_evaluator_mae.evaluate(preds)

    results.append({
        "model": name,
        "type": "regression",
        "time_sec": round(duration, 2),
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
    })

    print(f"{name} => time: {duration:.2f}s, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    return m

lr = LinearRegression(featuresCol="features", labelCol="track_popularity", maxIter=50, regParam=0.1)
train_and_eval_regressor(lr, "LinearRegression")

rf_reg = RandomForestRegressor(featuresCol="features", labelCol="track_popularity", numTrees=50, maxDepth=5, maxBins=32, subsamplingRate=0.7, featureSubsetStrategy="sqrt" )
train_and_eval_regressor(rf_reg, "RandomForestRegressor")

gbt_reg = GBTRegressor(featuresCol="features", labelCol="track_popularity", maxIter=50, maxDepth=7)
train_and_eval_regressor(gbt_reg, "GBTRegressor")

# 8. Classification: hit or not
bin_eval_auc = BinaryClassificationEvaluator(labelCol="label_hit", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
multi_eval_acc = MulticlassClassificationEvaluator(labelCol="label_hit", predictionCol="prediction", metricName="accuracy")
multi_eval_f1  = MulticlassClassificationEvaluator(labelCol="label_hit", predictionCol="prediction", metricName="f1")

def train_and_eval_classifier(model, name):
    start = time.time()
    m = model.fit(train)
    duration = time.time() - start
    preds = m.transform(test)
    auc = bin_eval_auc.evaluate(preds)
    acc = multi_eval_acc.evaluate(preds)
    f1  = multi_eval_f1.evaluate(preds)

    results.append({
        "model": name,
        "type": "classification",
        "time_sec": round(duration, 2),
        "AUC": round(auc, 4),
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4)
    })

    print(f"{name} => time: {duration:.2f}s, AUC: {auc:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
    return m

log_reg = LogisticRegression(featuresCol="features", labelCol="label_hit", maxIter=50)
train_and_eval_classifier(log_reg, "LogisticRegression")

rf_clf = RandomForestClassifier(featuresCol="features", labelCol="label_hit", numTrees=100, maxDepth=10)
train_and_eval_classifier(rf_clf, "RandomForestClassifier")

gbt_clf = GBTClassifier(featuresCol="features", labelCol="label_hit", maxIter=50, maxDepth=7)
train_and_eval_classifier(gbt_clf, "GBTClassifier")

print("\n==================== FINAL MODEL RESULTS ====================\n")
for r in results:
    print(r)
print("\n=============================================================\n")

# Stop Spark when done
spark.stop()

