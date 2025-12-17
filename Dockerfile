FROM apache/spark:3.5.0

USER root

# System deps + Python deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3-pip build-essential curl && \
    pip3 install --no-cache-dir pandas pyarrow synapseml && \
    rm -rf /var/lib/apt/lists/*

# Create ivy cache dir (prevents Ivy errors)
RUN mkdir -p /home/spark/.ivy2/cache && \
    chown -R spark:spark /home/spark

# ---- ADD SYNAPSEML JARS INTO SPARK CLASSPATH ----
# Download SynapseML jars from the SynapseML Maven repo
RUN curl -L -o /opt/spark/jars/synapseml-core_2.12-1.1.0.jar \
      https://mmlspark.azureedge.net/maven/com/microsoft/azure/synapseml-core_2.12/1.1.0/synapseml-core_2.12-1.1.0.jar && \
    curl -L -o /opt/spark/jars/synapseml-lightgbm_2.12-1.1.0.jar \
      https://mmlspark.azureedge.net/maven/com/microsoft/azure/synapseml-lightgbm_2.12/1.1.0/synapseml-lightgbm_2.12-1.1.0.jar

# Make sure Spark uses Python 3
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

USER spark
