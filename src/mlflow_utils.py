import os
import re
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def configure_mlflow(experiment_name):
    try:
        import mlflow
        from mlflow.exceptions import MlflowException
    except ModuleNotFoundError:
        print("MLflow is not installed; skipping MLflow logging.")
        return None

    root = project_root()
    tracking_uri = f"sqlite:///{root.as_posix()}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException as exc:
        message = str(exc)
        if "out-of-date database schema" in message or "Can't locate revision" in message:
            print(
                "MLflow tracking is unavailable because the local mlflow.db schema "
                "does not match this MLflow installation. Saved CSV reports will "
                "still be used. To repair tracking manually, back up mlflow.db and "
                f"run: mlflow db upgrade {tracking_uri}"
            )
            return None

        print(f"MLflow tracking is unavailable; skipping MLflow logging. Details: {exc}")
        return None

    return mlflow


def safe_metric_name(name):
    name = name.replace("@", "_at_")
    return re.sub(r"[^A-Za-z0-9_.:-]", "_", name)


def log_params(mlflow, params):
    if mlflow is None:
        return
    mlflow.log_params({key: value for key, value in params.items() if value is not None})


def log_metrics(mlflow, metrics, step=None, prefix=None):
    if mlflow is None:
        return

    for key, value in metrics.items():
        if value is None:
            continue
        metric_name = safe_metric_name(f"{prefix}_{key}" if prefix else key)
        try:
            mlflow.log_metric(metric_name, float(value), step=step)
        except (TypeError, ValueError):
            continue


def log_artifacts(mlflow, paths, artifact_path=None):
    if mlflow is None:
        return

    for path in paths:
        if path and os.path.exists(path):
            mlflow.log_artifact(path, artifact_path=artifact_path)
