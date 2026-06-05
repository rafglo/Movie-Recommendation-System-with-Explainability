import os
import re
from pathlib import Path


def project_root():
    return Path(__file__).resolve().parents[1]


def configure_mlflow(experiment_name):
    try:
        import mlflow
    except ModuleNotFoundError:
        print("MLflow is not installed; skipping MLflow logging.")
        return None

    root = project_root()
    tracking_uri = f"sqlite:///{root.as_posix()}/mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
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
