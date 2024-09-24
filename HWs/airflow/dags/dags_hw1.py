import json
import time
from datetime import timedelta
from io import StringIO

import pandas as pd
from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

s3_conn = "s3_connection"
s3_bucket = Variable.get("S3_BUCKET")
s3_path_prefix = "SergeyKrivosheev"

default_args = {
    "owner": "Sergey Krivosheev",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}


def init_model(model_name, **kwargs):
    timestamp = time.time()

    metrics = {
        "timestamp": timestamp,
        "model_name": model_name,
    }

    kwargs["ti"].xcom_push(
        key="init_metrics",
        value=metrics,
    )


def get_data(model_name, **kwargs):
    start_time = time.time()

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    end_time = time.time()

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "dataset_size": df.shape,
    }

    kwargs["ti"].xcom_push(
        key="data_metrics",
        value=metrics,
    )

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3 = S3Hook(s3_conn)
    path = f"{s3_path_prefix}/{model_name}/datasets/data.csv"
    s3.load_string(
        string_data=csv_buffer.getvalue(),
        bucket_name=s3_bucket,
        key=path,
        replace=True,
    )


def prepare_data(model_name, **kwargs):
    start_time = time.time()

    s3 = S3Hook(s3_conn)
    path = f"{s3_path_prefix}/{model_name}/datasets/data.csv"
    data = pd.read_csv(s3.download_file(key=path, bucket_name=s3_bucket))

    X = data.drop(columns=["target"])
    y = data["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    prepared_data = pd.DataFrame(X_scaled, columns=X.columns)
    prepared_data["target"] = y

    end_time = time.time()

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "features": X.columns.tolist(),
    }

    kwargs["ti"].xcom_push(
        key="prepare_metrics",
        value=metrics,
    )

    csv_buffer = StringIO()
    prepared_data.to_csv(csv_buffer, index=False)

    path = f"{s3_path_prefix}/{model_name}/datasets/prepared_data.csv"
    s3.load_string(
        string_data=csv_buffer.getvalue(),
        bucket_name=s3_bucket,
        key=path,
        replace=True,
    )


def train_model(model, model_name, **kwargs):
    start_time = time.time()

    s3 = S3Hook(s3_conn)
    path = f"{s3_path_prefix}/{model_name}/datasets/prepared_data.csv"
    data = pd.read_csv(s3.download_file(key=path, bucket_name=s3_bucket))

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_mse = mean_squared_error(y_train, train_preds)
    train_r2 = r2_score(y_train, train_preds)

    test_mse = mean_squared_error(y_test, test_preds)
    test_r2 = r2_score(y_test, test_preds)

    end_time = time.time()

    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "train_metrics": {"mse": train_mse, "r2": train_r2},
        "test_metrics": {"mse": test_mse, "r2": test_r2},
    }
    kwargs["ti"].xcom_push(
        key="train_metrics",
        value=metrics,
    )


def save_results(model_name, **kwargs):
    init_metrics = kwargs["ti"].xcom_pull(key="init_metrics")
    data_metrics = kwargs["ti"].xcom_pull(key="data_metrics")
    prepare_metrics = kwargs["ti"].xcom_pull(key="prepare_metrics")
    train_metrics = kwargs["ti"].xcom_pull(key="train_metrics")

    all_metrics = {
        "init": init_metrics,
        "get_data": data_metrics,
        "prepare_data": prepare_metrics,
        "train_model": train_metrics,
    }

    s3 = S3Hook(s3_conn)
    path = f"{s3_path_prefix}/{model_name}/results/metrics.json"

    s3.load_string(
        string_data=json.dumps(all_metrics, indent=4),
        bucket_name=s3_bucket,
        key=path,
        replace=True,
    )


def create_dag(dag_id, model_instance, model_name, schedule, default_args):
    with DAG(
        dag_id,
        tags=["mlops"],
        default_args=default_args,
        schedule_interval=schedule,
        start_date=days_ago(1),
    ) as dag:
        init = PythonOperator(
            task_id="init",
            python_callable=init_model,
            op_kwargs={
                "model_name": model_name,
            },
        )

        get_data_task = PythonOperator(
            task_id="get_data",
            python_callable=get_data,
            provide_context=True,
            op_kwargs={
                "model_name": model_name,
            },
        )

        prepare_data_task = PythonOperator(
            task_id="prepare_data",
            python_callable=prepare_data,
            provide_context=True,
            op_kwargs={
                "model_name": model_name,
            },
        )

        train_model_task = PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={
                "model": model_instance,
                "model_name": model_name,
            },
            provide_context=True,
        )

        save_results_task = PythonOperator(
            task_id="save_results",
            python_callable=save_results,
            provide_context=True,
            op_kwargs={
                "model_name": model_name,
            },
        )

        (
            init
            >> get_data_task
            >> prepare_data_task
            >> train_model_task
            >> save_results_task
        )

    return dag


models = {
    "linear_regression": LinearRegression(),
    "decision_tree": DecisionTreeRegressor(),
    "random_forest": RandomForestRegressor(),
}

for model_name, model_instance in models.items():
    dag_id = f"SergeyKrivosheev_{model_name}_dag"
    globals()[dag_id] = create_dag(
        dag_id=dag_id,
        model_name=model_name,
        model_instance=model_instance,
        schedule="0 1 * * *",
        default_args=default_args,
    )
