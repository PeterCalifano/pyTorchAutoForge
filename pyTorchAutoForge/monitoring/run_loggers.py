"""Generic MLflow and TensorBoard run loggers."""

from __future__ import annotations

import json
from numbers import Number
from pathlib import Path
from typing import Any, Mapping

import mlflow
from mlflow import MlflowClient
from torch.utils.tensorboard import SummaryWriter

from .types import MetricInput, MetricSplit, MetricsAsDict


# Internal helpers
def _metrics_to_mapping(metrics: MetricInput) -> Mapping[str, Any]:
    """Normalize metric containers to mapping form.

    Parameters
    ----------
    metrics
        Mapping or object exposing ``as_dict``.

    Returns
    -------
    Mapping[str, Any]
        Metric values keyed by metric name.
    """
    if isinstance(metrics, Mapping):
        return metrics

    if isinstance(metrics, MetricsAsDict):
        return metrics.as_dict()

    raise TypeError(
        "metrics must be a mapping or expose an as_dict() method. "
        f"Got {type(metrics)!r}."
    )


def _iter_scalar_metrics(metrics: MetricInput) -> dict[str, float]:
    """Extract scalar metrics accepted by MLflow and TensorBoard.

    Parameters
    ----------
    metrics
        Mapping or dataclass-like metric container.

    Returns
    -------
    dict[str, float]
        Scalar metric map. Two-number tuples are expanded into ``*_x`` and
        ``*_y`` entries. Keys prefixed with ``sample_`` are skipped because
        they describe per-sample debug payloads.
    """
    scalar_metrics_: dict[str, float] = {}

    # Keep only values each backend can store as scalar time-series data.
    for key_, value_ in _metrics_to_mapping(metrics).items():
        if key_.startswith("sample_"):
            continue

        if isinstance(value_, Number):
            scalar_metrics_[key_] = float(value_)
            continue

        if (
            isinstance(value_, tuple)
            and len(value_) == 2
            and all(isinstance(component_, Number) for component_ in value_)
        ):
            scalar_metrics_[f"{key_}_x"] = float(value_[0])
            scalar_metrics_[f"{key_}_y"] = float(value_[1])

    return scalar_metrics_


def _normalize_split(split: MetricSplit | str) -> str:
    """Return metric namespace string from enum or caller-provided value."""
    if isinstance(split, MetricSplit):
        return split.value
    return str(split)


class MlflowRunLogger:
    """Logger for MLflow experiment tracking."""

    VALID_TERMINAL_STATUS: tuple[str, ...] = ("FINISHED", "FAILED", "KILLED")

    # Public API
    def __init__(self,
                 run_id: str | None = None,
                 experiment_id: str | None = None,
                 tracking_uri: str | None = None,
                 mlflow_client: MlflowClient | None = None,
                 ) -> None:
        """Create MLflow logger bound to optional run context.

        Parameters
        ----------
        run_id
            Explicit MLflow run identifier. If omitted, active run is used.
        experiment_id
            Optional MLflow experiment identifier.
        tracking_uri
            Optional tracking URI. Defaults to current MLflow tracking URI.
        mlflow_client
            Optional preconfigured client, useful for tests.
        """
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.tracking_uri = tracking_uri if tracking_uri is not None else mlflow.get_tracking_uri()
        self.mlflow_client = (
            mlflow_client
            if mlflow_client is not None
            else MlflowClient(tracking_uri=self.tracking_uri)
        )

    @staticmethod
    def flatten_dataclass_fields(obj: Any, prefix: str = "") -> dict[str, Any]:
        """Unpack dataclass-like fields into a one-level flat dictionary."""
        out_: dict[str, Any] = {}
        fields_ = vars(obj) if hasattr(obj, "__dict__") else {}
        for field_name_, value_ in fields_.items():
            key_ = f"{prefix}/{field_name_}" if prefix else field_name_
            out_[key_] = value_
        return out_

    def log_run_params(self,
                       params_dict: Mapping[str, Any],
                       run_id: str | None = None,
                       experiment_id: str | None = None,
                       ) -> None:
        """Log run parameters to MLflow."""
        if not params_dict:
            print(
                "\033[38;5;208mWarning: Empty params_dict passed to log_run_params. Nothing logged.\033[0m"
            )
            return

        resolved_run_id_, _ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="run-params logging",
        )
        if resolved_run_id_ is None:
            return

        mlflow.log_params(
            {key_: str(value_) for key_, value_ in params_dict.items()},
            run_id=resolved_run_id_,
        )

    def log_epoch_data(self,
                       epoch_num: int,
                       train_metrics: MetricInput,
                       valid_metrics: MetricInput | None = None,
                       current_lr: float | None = None,
                       best_valid_loss: float | None = None,
                       run_id: str | None = None,
                       experiment_id: str | None = None,
                       ) -> None:
        """Log one training epoch to MLflow."""
        # Keep learning-rate and metric streams independently optional.
        if current_lr is not None:
            self._log_current_lr(
                current_lr=current_lr,
                epoch_num=epoch_num,
                run_id=run_id,
                experiment_id=experiment_id,
            )

        self._log_metrics(
            metrics_set_tag=MetricSplit.TRAIN,
            metrics=train_metrics,
            epoch_num=epoch_num,
            run_id=run_id,
            experiment_id=experiment_id,
        )

        if valid_metrics is None:
            return

        self._log_metrics(
            metrics_set_tag=MetricSplit.VALID,
            metrics=valid_metrics,
            epoch_num=epoch_num,
            run_id=run_id,
            experiment_id=experiment_id,
        )

        if best_valid_loss is not None:
            self._log_best_valid_loss(
                best_valid_loss=best_valid_loss,
                epoch_num=epoch_num,
                run_id=run_id,
                experiment_id=experiment_id,
            )

    def mark_run_finished(self,
                          run_id: str | None = None,
                          experiment_id: str | None = None,
                          use_end_run: bool = True,
                          ) -> None:
        """Mark MLflow run as finished."""
        self._set_run_status("FINISHED", run_id, experiment_id, use_end_run)

    def mark_run_failed(self,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        use_end_run: bool = True,
                        ) -> None:
        """Mark MLflow run as failed."""
        self._set_run_status("FAILED", run_id, experiment_id, use_end_run)

    def mark_run_killed(self,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        use_end_run: bool = True,
                        ) -> None:
        """Mark MLflow run as killed."""
        self._set_run_status("KILLED", run_id, experiment_id, use_end_run)

    # Internal methods
    def _resolve_run_context(self,
                             run_id: str | None = None,
                             experiment_id: str | None = None,
                             operation_name: str = "logging",
                             ) -> tuple[str | None, str | None]:
        """Resolve explicit or active MLflow run context."""
        # Explicit call-level context has precedence over logger defaults.
        resolved_run_id_ = run_id or self.run_id
        resolved_experiment_id_ = experiment_id or self.experiment_id

        if resolved_run_id_ is not None:
            return resolved_run_id_, resolved_experiment_id_

        active_run_ = mlflow.active_run()
        if active_run_ is None:
            print(
                f"\033[38;5;208mWarning: Skipping MLflow {operation_name}. "
                "No run_id provided and no active MLflow run.\033[0m"
            )
            return None, None

        resolved_run_id_ = active_run_.info.run_id
        if resolved_experiment_id_ is None:
            resolved_experiment_id_ = active_run_.info.experiment_id
        return resolved_run_id_, resolved_experiment_id_

    def _log_current_lr(self,
                        current_lr: float,
                        epoch_num: int,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        ) -> None:
        """Log learning rate for one epoch."""
        resolved_run_id_, _ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="learning-rate logging",
        )
        if resolved_run_id_ is None:
            return

        mlflow.log_metric(
            "train/lr",
            float(current_lr),
            step=epoch_num,
            run_id=resolved_run_id_,
        )

    def _log_best_valid_loss(self,
                             best_valid_loss: float,
                             epoch_num: int,
                             run_id: str | None = None,
                             experiment_id: str | None = None,
                             ) -> None:
        """Log best validation loss for one epoch."""
        resolved_run_id_, _ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="best-valid-loss logging",
        )
        if resolved_run_id_ is None:
            return

        mlflow.log_metric(
            "valid/best_loss",
            float(best_valid_loss),
            step=epoch_num,
            run_id=resolved_run_id_,
        )

    def _log_metrics(self,
                     metrics_set_tag: MetricSplit | str,
                     metrics: MetricInput,
                     epoch_num: int,
                     run_id: str | None = None,
                     experiment_id: str | None = None,
                     ) -> None:
        """Log scalar metrics under train, validation, or custom namespace."""
        prefix_ = _normalize_split(metrics_set_tag)
        prefixed_epoch_metrics_ = {
            f"{prefix_}/{key_}": value_
            for key_, value_ in _iter_scalar_metrics(metrics).items()
        }

        if not prefixed_epoch_metrics_:
            print(
                f"\033[38;5;208mWarning: No metrics logged for epoch {epoch_num} with tag "
                f"'{prefix_}' - all metrics were either sample-level or non-numeric.\033[0m"
            )
            return

        resolved_run_id_, _ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name=f"metrics logging ({prefix_})",
        )
        if resolved_run_id_ is None:
            return

        mlflow.log_metrics(
            prefixed_epoch_metrics_,
            step=epoch_num,
            run_id=resolved_run_id_,
        )

    def _set_run_status(self,
                        status: str,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        use_end_run: bool = True,
                        ) -> None:
        """Set terminal MLflow run status."""
        normalized_status_ = status.strip().upper()
        if normalized_status_ not in self.VALID_TERMINAL_STATUS:
            print(
                f"\033[38;5;208mWarning: Skipping MLflow run termination. "
                f"Invalid status '{status}'. Valid values: {self.VALID_TERMINAL_STATUS}.\033[0m"
            )
            return

        resolved_run_id_, _ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name=f"setting terminated state ({normalized_status_})",
        )
        if resolved_run_id_ is None:
            return

        if use_end_run:
            active_run_ = mlflow.active_run()
            if active_run_ is None:
                mlflow.start_run(run_id=resolved_run_id_)
                mlflow.end_run(status=normalized_status_)
                return

            active_run_id_ = active_run_.info.run_id
            if active_run_id_ == resolved_run_id_:
                mlflow.end_run(status=normalized_status_)
                return

            print(
                f"\033[38;5;208mWarning: Requested end_run for run_id "
                f"'{resolved_run_id_}', but active run is '{active_run_id_}'. "
                "Falling back to client.update_run.\033[0m"
            )

        self.mlflow_client.update_run(
            run_id=resolved_run_id_, status=normalized_status_)


class TensorBoardRunLogger:
    """TensorBoard logger with MLflow-like run context."""

    VALID_TERMINAL_STATUS: tuple[str, ...] = ("FINISHED", "FAILED", "KILLED")

    # Public API
    def __init__(self,
                 run_id: str | None = None,
                 experiment_id: str | None = None,
                 root_dir: str = "runs/tensorboard",
                 flush_secs: int = 30,
                 max_queue: int = 50,
                 run_context: Any | None = None,
                 ) -> None:
        """Create TensorBoard logger bound to optional run context."""
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.root_dir = root_dir
        self.flush_secs = flush_secs
        self.max_queue = max_queue
        self._writer_by_run_key: dict[tuple[str, str], SummaryWriter] = {}

        if run_context is not None:
            self.set_run_context_from_object(run_context)

    def set_run_context_from_object(self, run_context: Any) -> None:
        """Update logger run context from object exposing ``.info``."""
        resolved_run_id_, resolved_experiment_id_ = self._extract_run_info(
            run_context)
        if resolved_run_id_ is not None:
            self.run_id = resolved_run_id_
        if resolved_experiment_id_ is not None:
            self.experiment_id = resolved_experiment_id_

    def Set_run_context_from_object(self, run_context: Any) -> None:
        """Backward-compatible alias for older callers."""
        self.set_run_context_from_object(run_context)

    def log_run_params(self,
                       params_dict: Mapping[str, Any],
                       run_id: str | None = None,
                       experiment_id: str | None = None,
                       ) -> None:
        """Log run parameters to TensorBoard text and scalar streams."""
        writer_, _, _ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="run-params logging",
        )
        if writer_ is None:
            return

        for key_, value_ in params_dict.items():
            if isinstance(value_, bool):
                writer_.add_scalar(f"params/{key_}", int(value_), global_step=0)
            elif isinstance(value_, Number):
                writer_.add_scalar(f"params/{key_}", float(value_), global_step=0)

        params_json_ = json.dumps(params_dict, indent=2, default=str, sort_keys=True)
        writer_.add_text("params/all", params_json_, global_step=0)

    def log_epoch_data(self,
                       epoch_num: int,
                       train_metrics: MetricInput,
                       valid_metrics: MetricInput | None = None,
                       current_lr: float | None = None,
                       best_valid_loss: float | None = None,
                       run_id: str | None = None,
                       experiment_id: str | None = None,
                       ) -> None:
        """Log one training epoch to TensorBoard."""
        if current_lr is not None:
            self._log_current_lr(current_lr, epoch_num, run_id, experiment_id)

        self._log_metrics(MetricSplit.TRAIN, train_metrics,
                          epoch_num, run_id, experiment_id)

        if valid_metrics is not None:
            self._log_metrics(MetricSplit.VALID, valid_metrics,
                              epoch_num, run_id, experiment_id)

        if best_valid_loss is not None:
            self._log_best_valid_loss(
                best_valid_loss, epoch_num, run_id, experiment_id)

        writer_, _, _ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="flush",
        )
        if writer_ is not None:
            writer_.flush()

    def mark_run_finished(self,
                          run_id: str | None = None,
                          experiment_id: str | None = None,
                          ) -> None:
        """Mark TensorBoard run as finished."""
        self._set_terminate_run("FINISHED", run_id, experiment_id)

    def mark_run_failed(self,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        ) -> None:
        """Mark TensorBoard run as failed."""
        self._set_terminate_run("FAILED", run_id, experiment_id)

    def mark_run_killed(self,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        ) -> None:
        """Mark TensorBoard run as killed."""
        self._set_terminate_run("KILLED", run_id, experiment_id)

    def close(self) -> None:
        """Flush and close all open TensorBoard writers."""
        for writer_ in self._writer_by_run_key.values():
            writer_.flush()
            writer_.close()
        self._writer_by_run_key.clear()

    # Internal methods
    @staticmethod
    def _extract_run_info(run_context: Any) -> tuple[str | None, str | None]:
        """Extract ``run_id`` and ``experiment_id`` from MLflow-like object."""
        run_info_ = getattr(run_context, "info", None)
        if run_info_ is None:
            return None, None
        return (
            getattr(run_info_, "run_id", None),
            getattr(run_info_, "experiment_id", None),
        )

    def _resolve_run_context(self,
                             run_id: str | None = None,
                             experiment_id: str | None = None,
                             operation_name: str = "logging",
                             ) -> tuple[str | None, str | None]:
        """Resolve TensorBoard run context from call overrides or defaults."""
        resolved_run_id_ = run_id or self.run_id
        resolved_experiment_id_ = experiment_id or self.experiment_id

        if resolved_run_id_ is not None:
            return resolved_run_id_, resolved_experiment_id_

        print(
            f"\033[38;5;208mWarning: Skipping TensorBoard {operation_name}. "
            "No run_id provided. Set one explicitly or provide an MLflow-like run context object.\033[0m"
        )
        return None, None

    @staticmethod
    def _is_uri_path(path_str: str) -> bool:
        """Return True for URI-like root paths."""
        return "://" in path_str

    def _build_log_dir(self, resolved_run_id: str, resolved_experiment_id: str | None) -> str:
        """Build TensorBoard log directory for run context."""
        experiment_component_ = resolved_experiment_id or "default_experiment"
        if self._is_uri_path(self.root_dir):
            root_dir_ = self.root_dir.rstrip("/")
            return f"{root_dir_}/{experiment_component_}/{resolved_run_id}"

        # Local paths are created immediately so SummaryWriter can open event files.
        log_dir_ = str(Path(self.root_dir) / experiment_component_ / resolved_run_id)
        Path(log_dir_).mkdir(parents=True, exist_ok=True)
        return log_dir_

    def _resolve_writer(self,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        operation_name: str = "logging",
                        ) -> tuple[SummaryWriter | None, str | None, str | None]:
        """Return cached writer or create one for resolved run context."""
        resolved_run_id_, resolved_experiment_id_ = self._resolve_run_context(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name=operation_name,
        )
        if resolved_run_id_ is None:
            return None, None, None

        run_key_ = (resolved_experiment_id_ or "default_experiment", resolved_run_id_)
        writer_ = self._writer_by_run_key.get(run_key_)
        if writer_ is not None:
            return writer_, resolved_run_id_, resolved_experiment_id_

        # Writer cache avoids multiple event files for repeated epoch calls.
        writer_ = SummaryWriter(
            log_dir=self._build_log_dir(
                resolved_run_id=resolved_run_id_,
                resolved_experiment_id=resolved_experiment_id_,
            ),
            max_queue=self.max_queue,
            flush_secs=self.flush_secs,
        )
        writer_.add_text("run/run_id", resolved_run_id_, global_step=0)
        writer_.add_text(
            "run/experiment_id",
            resolved_experiment_id_ or "default_experiment",
            global_step=0,
        )
        self._writer_by_run_key[run_key_] = writer_
        return writer_, resolved_run_id_, resolved_experiment_id_

    def _log_current_lr(self,
                        current_lr: float,
                        epoch_num: int,
                        run_id: str | None = None,
                        experiment_id: str | None = None,
                        ) -> None:
        """Log learning rate scalar to TensorBoard."""
        writer_, _, _ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="learning-rate logging",
        )
        if writer_ is not None:
            writer_.add_scalar("train/lr", float(current_lr), epoch_num)

    def _log_best_valid_loss(self,
                             best_valid_loss: float,
                             epoch_num: int,
                             run_id: str | None = None,
                             experiment_id: str | None = None,
                             ) -> None:
        """Log best validation loss scalar to TensorBoard."""
        writer_, _, _ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name="best-valid-loss logging",
        )
        if writer_ is not None:
            writer_.add_scalar("valid/best_loss", float(best_valid_loss), epoch_num)

    def _log_metrics(self,
                     metrics_set_tag: MetricSplit | str,
                     metrics: MetricInput,
                     epoch_num: int,
                     run_id: str | None = None,
                     experiment_id: str | None = None,
                     ) -> None:
        """Log scalar metrics under TensorBoard namespace."""
        writer_, _, _ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name=f"metrics logging ({_normalize_split(metrics_set_tag)})",
        )
        if writer_ is None:
            return

        prefix_ = _normalize_split(metrics_set_tag)
        logged_any_metric_ = False
        for key_, value_ in _iter_scalar_metrics(metrics).items():
            writer_.add_scalar(f"{prefix_}/{key_}", float(value_), epoch_num)
            logged_any_metric_ = True

        if not logged_any_metric_:
            print(
                f"\033[38;5;208mWarning: No metrics logged for epoch {epoch_num} with tag "
                f"'{prefix_}' - all metrics were either sample-level or non-numeric.\033[0m"
            )

    def _set_terminate_run(self,
                           status: str,
                           run_id: str | None = None,
                           experiment_id: str | None = None,
                           ) -> None:
        """Write terminal run status and close TensorBoard writer."""
        normalized_status_ = status.strip().upper()
        if normalized_status_ not in self.VALID_TERMINAL_STATUS:
            print(
                f"\033[38;5;208mWarning: Skipping TensorBoard run termination. "
                f"Invalid status '{status}'. Valid values: {self.VALID_TERMINAL_STATUS}.\033[0m"
            )
            return

        writer_, resolved_run_id_, resolved_experiment_id_ = self._resolve_writer(
            run_id=run_id,
            experiment_id=experiment_id,
            operation_name=f"setting terminated state ({normalized_status_})",
        )
        if writer_ is None or resolved_run_id_ is None:
            return

        writer_.add_text("run/status", normalized_status_, global_step=0)
        writer_.flush()
        writer_.close()

        run_key_ = (resolved_experiment_id_ or "default_experiment", resolved_run_id_)
        self._writer_by_run_key.pop(run_key_, None)


class CompositeRunLogger:
    """Fan-out logger for multiple run logger backends."""

    # Public API
    def __init__(self, *loggers: Any) -> None:
        """Create composite logger from non-null backends."""
        self.loggers = tuple(logger_ for logger_ in loggers if logger_ is not None)

    def log_run_params(self, params_dict: Mapping[str, Any], **kwargs: Any) -> None:
        """Forward parameter logging to every backend."""
        for logger_ in self.loggers:
            logger_.log_run_params(params_dict, **kwargs)

    def log_epoch_data(self,
                       epoch_num: int,
                       train_metrics: MetricInput,
                       valid_metrics: MetricInput | None = None,
                       current_lr: float | None = None,
                       best_valid_loss: float | None = None,
                       **kwargs: Any,
                       ) -> None:
        """Forward epoch logging to every backend."""
        for logger_ in self.loggers:
            logger_.log_epoch_data(
                epoch_num=epoch_num,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
                current_lr=current_lr,
                best_valid_loss=best_valid_loss,
                **kwargs,
            )

    def mark_run_finished(self, **kwargs: Any) -> None:
        """Mark all backends finished."""
        for logger_ in self.loggers:
            logger_.mark_run_finished(**kwargs)

    def mark_run_failed(self, **kwargs: Any) -> None:
        """Mark all backends failed."""
        for logger_ in self.loggers:
            logger_.mark_run_failed(**kwargs)

    def mark_run_killed(self, **kwargs: Any) -> None:
        """Mark all backends killed."""
        for logger_ in self.loggers:
            logger_.mark_run_killed(**kwargs)

    def close(self) -> None:
        """Close backend loggers that expose ``close``."""
        for logger_ in self.loggers:
            close_ = getattr(logger_, "close", None)
            if close_ is not None:
                close_()
