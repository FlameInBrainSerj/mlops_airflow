import json
import os
import time
from datetime import timedelta
from io import StringIO

import mlflow
import pandas as pd
from airflow.models import DAG, Variable
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.utils.dates import days_ago
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

S3_CONN = "s3_connection"
S3_BUCKET = Variable.get("S3_BUCKET_Project")
S3_PATH_PREFIX = "SergeyKrivosheev"

DAG_NAME = "SergeyKrivosheev_project_dag"
DEFAULT_ARGS = {
    "owner": "Sergey Krivosheev",
    "retries": 3,
    "retry_delay": timedelta(minutes=1),
}

MODELS = dict(
    zip(
        ["RandomForest", "LinearRegression", "HistGB"],
        [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()],
    )
)

EXPERIMENT_NAME = "sergey_krivosheev_finalproject"
PARENT_RUN_NAME = "FlameInBrain"


def configure_mlflow():
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)


def get_experiment_id(experiment_name: str) -> int:
    client = MlflowClient()

    if client.get_experiment_by_name(experiment_name):
        exp_id = mlflow.set_experiment(experiment_name).experiment_id
    else:
        exp_id = mlflow.create_experiment(
            name=experiment_name, artifact_location=f"s3://{S3_BUCKET}"
        )

    return exp_id


def create_buffer_and_load_data_to_s3(data: pd.DataFrame, s3_hook: S3Hook, path: str):
    csv_buffer = StringIO()
    data.to_csv(csv_buffer, index=False)

    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        bucket_name=S3_BUCKET,
        key=path,
        replace=True,
    )


def load_data_from_s3(s3_hook: S3Hook, path: str) -> pd.DataFrame:
    data = pd.read_csv(s3_hook.download_file(key=path, bucket_name=S3_BUCKET))
    return data


def init(**kwargs):
    # Init step logs
    timestamp = time.time()
    kwargs["ti"].xcom_push(
        key="init_metrics",
        value={"timestamp": timestamp},
    )

    # Init Airflow env variables
    configure_mlflow()
    # Get experiment ID in MLFlow
    exp_id = get_experiment_id(EXPERIMENT_NAME)
    # Init parent_run
    parent_run = mlflow.start_run(
        run_name=PARENT_RUN_NAME,
        experiment_id=exp_id,
        description="parent",
    )
    # Push exp_info to XCom
    kwargs["ti"].xcom_push(
        key="mlflow_exp_info",
        value={
            "exp_name": EXPERIMENT_NAME,
            "exp_id": exp_id,
            "parent_run_name": PARENT_RUN_NAME,
            "parent_run_id": parent_run.info.run_uuid,
        },
    )


def get_data(**kwargs):
    start_time = time.time()

    # Get data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    end_time = time.time()

    # Get_data step logs
    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "dataset_size": df.shape,
    }
    kwargs["ti"].xcom_push(
        key="data_metrics",
        value=metrics,
    )

    # Upload initial data to S3
    s3 = S3Hook(S3_CONN)
    create_buffer_and_load_data_to_s3(
        data=df,
        s3_hook=s3,
        path=f"{S3_PATH_PREFIX}/datasets/data.csv",
    )


def prepare_data(**kwargs):
    start_time = time.time()

    # Get initial data from S3
    s3 = S3Hook(S3_CONN)
    data = load_data_from_s3(s3_hook=s3, path=f"{S3_PATH_PREFIX}/datasets/data.csv")

    X = data.drop(columns=["target"])
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=0.5,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns,
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X.columns,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns,
    )

    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)

    X_train_scaled["target"] = y_train
    X_val_scaled["target"] = y_val

    end_time = time.time()

    # Prepare_data step logs
    metrics = {
        "start_time": start_time,
        "end_time": end_time,
        "features": X.columns.tolist(),
    }
    kwargs["ti"].xcom_push(
        key="prepare_metrics",
        value=metrics,
    )

    # Upload splitted data to S3
    s3 = S3Hook(S3_CONN)
    create_buffer_and_load_data_to_s3(
        data=X_train_scaled,
        s3_hook=s3,
        path=f"{S3_PATH_PREFIX}/datasets/data_train.csv",
    )
    create_buffer_and_load_data_to_s3(
        data=X_val_scaled,
        s3_hook=s3,
        path=f"{S3_PATH_PREFIX}/datasets/data_val.csv",
    )
    create_buffer_and_load_data_to_s3(
        data=X_test_scaled,
        s3_hook=s3,
        path=f"{S3_PATH_PREFIX}/datasets/data_test.csv",
    )


def train_and_log_model(model, model_name, **kwargs):
    start_time = time.time()

    # Get experiment info
    exp_info = kwargs["ti"].xcom_pull(key="mlflow_exp_info")

    # Get data from S3
    s3 = S3Hook(S3_CONN)
    data_train = load_data_from_s3(
        s3_hook=s3, path=f"{S3_PATH_PREFIX}/datasets/data_train.csv"
    )
    data_val = load_data_from_s3(
        s3_hook=s3, path=f"{S3_PATH_PREFIX}/datasets/data_val.csv"
    )
    X_test = load_data_from_s3(
        s3_hook=s3, path=f"{S3_PATH_PREFIX}/datasets/data_test.csv"
    )

    # Fit model
    X_train = data_train.drop(columns=["target"])
    y_train = data_train["target"]
    model.fit(X_train, y_train)

    end_time = time.time()

    # Train_and_log_model step logs
    metrics = {
        "start_time": start_time,
        "end_time": end_time,
    }
    kwargs["ti"].xcom_push(
        key="train_metrics",
        value=metrics,
    )

    # Validation dataset
    eval_df = data_val.copy()

    # # MLFlow child run
    with mlflow.start_run(
        run_name=model_name,
        experiment_id=exp_info["exp_id"],
        nested=True,
        parent_run_id=exp_info["parent_run_id"],
    ) as child_run:

        # Log model
        signature = infer_signature(X_test, model.predict(X_test))
        model_info = mlflow.sklearn.log_model(
            model,
            model_name,
            signature=signature,
            registered_model_name=f"sk-learn-{model_name}-project-model",
        )

        # Evaluate model
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_df,
            targets="target",
            model_type="regressor",
            evaluators=["default"],
        )

    kwargs["ti"].xcom_push(
        key=f"model_info-{model_name}",
        value={
            "model_uri": model_info.model_uri,
            "model_name": model_name,
        },
    )


def save_results(models_names, **kwargs) -> None:
    init_metrics = kwargs["ti"].xcom_pull(key="init_metrics")
    data_metrics = kwargs["ti"].xcom_pull(key="data_metrics")
    prepare_metrics = kwargs["ti"].xcom_pull(key="prepare_metrics")
    train_metrics = kwargs["ti"].xcom_pull(key="train_metrics")
    mlflow_exp_info = kwargs["ti"].xcom_pull(key="mlflow_exp_info")

    # Load info about all models
    for model_name in models_names:
        model_info = kwargs["ti"].xcom_pull(key=f"model_info-{model_name}")

        all_metrics = {
            "dag_name": DAG_NAME,
            "mlflow_exp_info": mlflow_exp_info,
            "model_info": model_info,
            "init": init_metrics,
            "get_data": data_metrics,
            "prepare_data": prepare_metrics,
            "train_model": train_metrics,
        }

        s3 = S3Hook(S3_CONN)
        path = f"{S3_PATH_PREFIX}/{model_info['model_name']}/results/metrics.json"

        s3.load_string(
            string_data=json.dumps(all_metrics, indent=4),
            bucket_name=S3_BUCKET,
            key=path,
            replace=True,
        )


dag = DAG(
    DAG_NAME,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
    schedule_interval="0 1 * * *",
    start_date=days_ago(1),
)

task_init = PythonOperator(
    task_id="init",
    python_callable=init,
    dag=dag,
)

task_get_data = PythonOperator(
    task_id="get_data",
    python_callable=get_data,
    provide_context=True,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

training_model_tasks = [
    PythonOperator(
        task_id=f"train_model_{model_name}",
        python_callable=train_and_log_model,
        provide_context=True,
        op_kwargs={"model": model_instance, "model_name": model_name},
        dag=dag,
    )
    for model_name, model_instance in MODELS.items()
]


task_save_results = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    provide_context=True,
    dag=dag,
    op_kwargs={"models_names": MODELS.keys()},
)

(
    task_init
    >> task_get_data
    >> task_prepare_data
    >> training_model_tasks
    >> task_save_results
)
