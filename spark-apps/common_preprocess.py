from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF, VectorAssembler
)
from pyspark.ml import Pipeline

HIT_THRESHOLD = 70

AUDIO_FEATURES = [
    "danceability","energy","key","loudness","mode","speechiness","acousticness",
    "instrumentalness","liveness","valence","tempo","duration_ms"
]

def sql_clean_and_label(df: DataFrame) -> DataFrame:
    # Spark SQL requirement: do cleaning/labeling with SQL
    df.createOrReplaceTempView("songs_raw")

    # 1) select + cast
    df = df.sparkSession.sql(f"""
        SELECT
            track_name,
            CAST(track_popularity AS INT) AS track_popularity,
            CASE WHEN CAST(track_popularity AS INT) >= {HIT_THRESHOLD} THEN 1 ELSE 0 END AS label_hit,
            LOWER(REGEXP_REPLACE(track_name, '[^a-zA-Z0-9\\s]', ' ')) AS title_clean,
            -- audio features (cast to double)
            CAST(danceability AS DOUBLE) AS danceability,
            CAST(energy AS DOUBLE) AS energy,
            CAST(key AS DOUBLE) AS key,
            CAST(loudness AS DOUBLE) AS loudness,
            CAST(mode AS DOUBLE) AS mode,
            CAST(speechiness AS DOUBLE) AS speechiness,
            CAST(acousticness AS DOUBLE) AS acousticness,
            CAST(instrumentalness AS DOUBLE) AS instrumentalness,
            CAST(liveness AS DOUBLE) AS liveness,
            CAST(valence AS DOUBLE) AS valence,
            CAST(tempo AS DOUBLE) AS tempo,
            CAST(duration_ms AS DOUBLE) AS duration_ms
        FROM songs_raw
        WHERE track_name IS NOT NULL AND track_popularity IS NOT NULL
    """)

    df.createOrReplaceTempView("songs_cast")

    # 2) SQL feature flags
    df = df.sparkSession.sql("""
        SELECT *,
            CASE WHEN title_clean LIKE '%remix%' THEN 1.0 ELSE 0.0 END AS is_remix,
            CASE WHEN title_clean LIKE '%live%'  THEN 1.0 ELSE 0.0 END AS is_live
        FROM songs_cast
    """)

    # Drop rows with null audio features after casting
    for c in AUDIO_FEATURES:
        df = df.filter(col(c).isNotNull())

    return df


def build_text_audio_pipeline(df: DataFrame) -> PipelineModel:
    title_tokenizer = Tokenizer(inputCol="title_clean", outputCol="title_tokens")
    title_stop = StopWordsRemover(inputCol="title_tokens", outputCol="title_tokens_clean")
    title_tf = HashingTF(inputCol="title_tokens_clean", outputCol="title_tf", numFeatures=2**12)
    title_idf = IDF(inputCol="title_tf", outputCol="title_tfidf")

    assembler_inputs = ["title_tfidf", "is_remix", "is_live"] + AUDIO_FEATURES
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    pipe = Pipeline(stages=[title_tokenizer, title_stop, title_tf, title_idf, assembler])
    return pipe.fit(df)


def preprocess(df: DataFrame, repartition_n: int = 12) -> DataFrame:
    df = sql_clean_and_label(df)

    # explicit repartition requirement
    df = df.repartition(repartition_n)

    model = build_text_audio_pipeline(df)
    out = model.transform(df).select("features", "label_hit")
    return out
