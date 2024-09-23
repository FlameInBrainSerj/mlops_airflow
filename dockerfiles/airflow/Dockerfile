FROM apache/airflow:latest as development
COPY requirements.txt /tmp


RUN pip install jupyter-core
RUN pip install jupyter
RUN pip install --no-cache-dir -r /tmp/requirements.txt


RUN chmod -R 775 /home/airflow/.local/share/jupyter
