"""Tests for monitoring run logger backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow

from pyTorchAutoForge.monitoring import (
    CompositeRunLogger,
    MetricSplit,
    MlflowRunLogger,
    TensorBoardRunLogger,
)


@dataclass
class DummyMetrics:
    """Metric container exposing scalar and non-scalar values."""

    loss: float = 1.0
    mean_abs_err: tuple[float, float] = (2.0, 3.0)
    sample_values: tuple[float, ...] = (4.0, 5.0)
    label: str = "ignored"

    def as_dict(self) -> dict[str, Any]:
        """Return metric mapping for logger protocol."""
        return {
            "loss": self.loss,
            "mean_abs_err": self.mean_abs_err,
            "sample_values": self.sample_values,
            "label": self.label,
        }


def _reset_mlflow() -> None:
    """Close active MLflow run if one is open."""
    if mlflow.active_run() is not None:
        mlflow.end_run()


def test_mlflow_run_logger_logs_params_metrics_and_terminal_state(tmp_path: Path) -> None:
    """MLflow logger writes params, metrics, and terminal status."""
    _reset_mlflow()
    old_tracking_uri = mlflow.get_tracking_uri()
    tracking_uri = f"file://{tmp_path / 'mlruns'}"

    try:
        mlflow.set_tracking_uri(tracking_uri)
        experiment = mlflow.set_experiment("monitoring-test")
        run = mlflow.start_run(run_name="logger-test")

        logger = MlflowRunLogger(
            run_id=run.info.run_id,
            experiment_id=experiment.experiment_id,
            tracking_uri=tracking_uri,
        )
        logger.log_run_params({"batch_size": 8, "model/name": "demo"})
        logger.log_epoch_data(
            epoch_num=7,
            train_metrics=DummyMetrics(loss=0.5),
            valid_metrics={"loss": 0.25, "offset": (
                1, 2), "sample_debug": 9.0},
            current_lr=1e-3,
            best_valid_loss=0.25,
        )
        logger.mark_run_finished()

        fetched_run = mlflow.MlflowClient(
            tracking_uri=tracking_uri).get_run(run.info.run_id)
        assert fetched_run.info.status == "FINISHED"
        assert fetched_run.data.params["batch_size"] == "8"
        assert fetched_run.data.params["model/name"] == "demo"
        assert fetched_run.data.metrics["train/loss"] == 0.5
        assert fetched_run.data.metrics["train/mean_abs_err_x"] == 2.0
        assert fetched_run.data.metrics["train/mean_abs_err_y"] == 3.0
        assert "train/sample_values" not in fetched_run.data.metrics
        assert fetched_run.data.metrics["valid/loss"] == 0.25
        assert fetched_run.data.metrics["valid/offset_x"] == 1.0
        assert fetched_run.data.metrics["valid/offset_y"] == 2.0
        assert "valid/sample_debug" not in fetched_run.data.metrics
        assert fetched_run.data.metrics["train/lr"] == 1e-3
        assert fetched_run.data.metrics["valid/best_loss"] == 0.25
    finally:
        _reset_mlflow()
        mlflow.set_tracking_uri(old_tracking_uri)


def test_tensorboard_run_logger_writes_run_directory_and_event_file(tmp_path: Path) -> None:
    """TensorBoard logger creates run directory and event file."""
    logger = TensorBoardRunLogger(
        run_id="run-1",
        experiment_id="exp-1",
        root_dir=str(tmp_path),
        flush_secs=1,
        max_queue=1,
    )

    logger.log_run_params({"batch_size": 4, "flag": True, "name": "demo"})
    logger.log_epoch_data(
        epoch_num=2,
        train_metrics=DummyMetrics(loss=0.75),
        valid_metrics={"loss": 0.5},
        current_lr=1e-4,
        best_valid_loss=0.5,
    )
    logger.mark_run_finished()

    log_dir = tmp_path / "exp-1" / "run-1"
    assert log_dir.is_dir()
    assert list(log_dir.glob("events.out.tfevents.*"))


def test_composite_logger_fans_out_to_backends() -> None:
    """Composite logger forwards calls to each backend."""

    class SpyLogger:
        """Small spy backend used to verify fan-out behavior."""

        def __init__(self) -> None:
            """Create empty call recorder."""
            self.params: list[dict[str, Any]] = []
            self.epochs: list[int] = []
            self.closed = False

        def log_run_params(self, params_dict: dict[str, Any], **_: Any) -> None:
            """Record parameter payload."""
            self.params.append(params_dict)

        def log_epoch_data(self,
                           epoch_num: int,
                           train_metrics: Any,
                           valid_metrics: Any | None = None,
                           current_lr: float | None = None,
                           best_valid_loss: float | None = None,
                           **_: Any,
                           ) -> None:
            """Record epoch number."""
            self.epochs.append(epoch_num)

        def mark_run_finished(self, **_: Any) -> None:
            """Record terminal marker."""
            self.epochs.append(-1)

        def close(self) -> None:
            """Record close call."""
            self.closed = True

    first = SpyLogger()
    second = SpyLogger()
    logger = CompositeRunLogger(first, second)

    logger.log_run_params({"x": 1})
    logger.log_epoch_data(epoch_num=3, train_metrics={"loss": 1.0})
    logger.mark_run_finished()
    logger.close()

    assert first.params == [{"x": 1}]
    assert second.params == [{"x": 1}]
    assert first.epochs == [3, -1]
    assert second.epochs == [3, -1]
    assert first.closed is True
    assert second.closed is True


def test_monitoring_exports_are_available_from_top_level() -> None:
    """Top-level package exposes monitoring symbols through lazy export map."""
    import pyTorchAutoForge

    assert pyTorchAutoForge.MetricSplit is MetricSplit
    assert pyTorchAutoForge.MlflowRunLogger is MlflowRunLogger
    assert pyTorchAutoForge.TensorBoardRunLogger is TensorBoardRunLogger
