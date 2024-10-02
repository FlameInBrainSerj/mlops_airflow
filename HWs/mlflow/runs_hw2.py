import mlflow
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

ARTIFACT_ROOT = "s3://mlflow"
EXPERIMENT_NAME = "sergey_krivosheev_test"
PARENT_RUN_NAME = "FlameInBrain"

MODELS = dict(
    zip(
        ["RandomForest", "LinearRegression", "HistGB"],
        [RandomForestRegressor(), LinearRegression(), HistGradientBoostingRegressor()],
    )
)


def get_experiment_id(experiment_name: str) -> int:
    client = MlflowClient()

    if client.get_experiment_by_name(experiment_name):
        exp_id = mlflow.set_experiment(experiment_name).experiment_id
    else:
        exp_id = mlflow.create_experiment(
            name=experiment_name, artifact_location="s3://mlflow"
        )

    return exp_id


def get_data() -> None:
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    return df


def prepare_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    return X_train_scaled, X_test_scaled, X_val_scaled, y_train, y_val


def train_and_log_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame,
) -> None:

    model.fit(X_train, y_train)

    # Validation dataset
    eval_df = X_val.copy()
    eval_df["target"] = y_val

    # Log model
    signature = infer_signature(X_test, model.predict(X_test))
    model_info = mlflow.sklearn.log_model(
        model,
        model_name,
        signature=signature,
        registered_model_name=f"sk-learn-{model_name}-model",
    )

    # Evaluate model
    mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_df,
        targets="target",
        model_type="regressor",
        evaluators=["default"],
    )


if __name__ == "__main__":

    exp_id = get_experiment_id(EXPERIMENT_NAME)

    with mlflow.start_run(
        run_name=PARENT_RUN_NAME,
        experiment_id=exp_id,
        description="parent",
    ) as parent_run:

        for model_name, model_instance in MODELS.items():

            with mlflow.start_run(
                run_name=model_name,
                experiment_id=exp_id,
                nested=True,
            ) as child_run:

                # Get data
                data = get_data()
                # Prepare data
                X_train, X_test, X_val, y_train, y_val = prepare_data(data=data)
                # Train and log model
                train_and_log_model(
                    model=model_instance,
                    model_name=model_name,
                    X_train=X_train,
                    X_test=X_test,
                    X_val=X_val,
                    y_train=y_train,
                    y_val=y_val,
                )
