# MLOps (Airflow + MLFlow)

The task is hosted at: [sskrivosheev.ru](http://sskrivosheev.ru/)

Services can accessed following way:
- Airflow:  
    * Url:          [sskrivosheev.ru/airflow](http://sskrivosheev.ru/airflow)
    * Login:        airflow
    * Password:     airflow
- MLFlow:  
    * Url:          [sskrivosheev.ru/mlflow](http://sskrivosheev.ru/mlflow)
- Minio (S3):  
    * Url:          [sskrivosheev.ru/minio](http://sskrivosheev.ru/minio)
    * Login:        airflow
    * Password:     airflow123
- Jupyter Noteobook:  
    * Url:          [sskrivosheev.ru/jupyter](http://sskrivosheev.ru/jupyter)
    * Token:        airflow

Variables in Airflow:
- S3_BUCKET:  
    * Key: S3_BUCKET   
    * Val: airflow

Connections in Airflow:
- s3_connection:  
    * aws_access_key_id:        airflow
    * aws_secret_access_key:    airflow123
    * endpoint_url:             http://minio:9000 (docker)

HWs:
- HW1: (done)      
    - Файл: HWs/airflow/dags/dags_hw1.py
    - Бакет: airflow
- HW2: (done)      
    - Файл: HWs/mlflow/runs_hw2.py
    - Бакет: mlflow
- Project: (not yet)  
    - Файл: HWs/airflow/dags/dags_project.py
    - Бакет: lizvladi-mlops
