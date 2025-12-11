# Dockerfile
FROM apache/spark:3.5.0

USER root

# Install pip and Python deps (pandas + pyarrow)
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip && \
    pip3 install --no-cache-dir pandas pyarrow && \
    rm -rf /var/lib/apt/lists/*

# Make sure Spark uses Python 3 for driver & executors
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER spark
