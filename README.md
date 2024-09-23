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
    * aws_access_key_id:        <generated_in_minio_ak>
    * aws_secret_access_key:    <generated_in_minio_ak>
    * endpoint_url:             http://minio:9000 (docker)

DAGS_Files:
- HW1:      dags/dags_hw1.py
- HW2:      TBA
- Project:  TBA