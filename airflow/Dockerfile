FROM apache/airflow:2.10.5-python3.12

USER root

RUN python -m venv /opt/ml-venv \
    && chown -R airflow:root /opt/ml-venv

USER airflow

COPY requirements.txt /tmp/requirements.txt

RUN /opt/ml-venv/bin/pip install --no-cache-dir \
    -r /tmp/requirements.txt