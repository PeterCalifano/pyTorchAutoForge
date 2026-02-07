import pytest
import torch

from pyTorchAutoForge.datasets.LabelsClasses import PTAF_Datakey
from pyTorchAutoForge.datasets.vector_error_models.VectorErrorsBaseClasses import (
    Vector1dErrorModel,
    Vector1dErrorStackModel,
    DistributionType,
    GaussianParams,
    PoissonParams,
    UniformParams,
    ExponentialParams,
)


def _gaussian_model(**kwargs: object) -> Vector1dErrorModel:
    return Vector1dErrorModel(
        variable_name="x",
        shape=(1,),
        error_type=DistributionType.GAUSSIAN,
        parameters=GaussianParams(mean=0.0, std=1.0),
        **kwargs,
    )


def _uniform_model(low: float, high: float, **kwargs: object) -> Vector1dErrorModel:
    return Vector1dErrorModel(
        variable_name="x",
        shape=(1,),
        error_type=DistributionType.UNIFORM,
        parameters=UniformParams(low=low, high=high),
        **kwargs,
    )


def test_stack_keys_unknown_datakey_raises() -> None:
    model = _gaussian_model()
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(keys=("not_a_key",), error_models=(model,))


def test_key_size_match_allows() -> None:
    model = _gaussian_model()
    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.BBOX_XYWH,),
        error_models=(model,),
        key_sizes=(4,),
    )
    values = torch.zeros(4)
    assert stack.apply(values).shape == values.shape


def test_key_size_mismatch_raises() -> None:
    model = _gaussian_model()
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(
            keys=(PTAF_Datakey.BBOX_XYWH,),
            error_models=(model,),
            key_sizes=(3,),
        )


def test_key_size_required_for_unknown() -> None:
    model = _gaussian_model()
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(keys=(PTAF_Datakey.IMAGE,), error_models=(model,))

    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.IMAGE,),
        error_models=(model,),
        key_sizes=(8,),
    )
    values = torch.zeros(8)
    assert stack.apply(values).shape == values.shape


def test_global_indices_validation() -> None:
    model_ok = _gaussian_model(target_indices=(0, 1, 2, 3))
    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.BBOX_XYWH,), error_models=(model_ok,)
    )
    values = torch.zeros(4)
    assert stack.apply(values).shape == values.shape

    model_too_many = _gaussian_model(target_indices=(0, 1, 2, 3, 4))
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(
            keys=(PTAF_Datakey.BBOX_XYWH,), error_models=(model_too_many,)
        )

    model_oob = _gaussian_model(target_indices=(4,))
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(
            keys=(PTAF_Datakey.BBOX_XYWH,), error_models=(model_oob,)
        )

    model_negative = _gaussian_model(target_indices=(-1,))
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(
            keys=(PTAF_Datakey.BBOX_XYWH,), error_models=(model_negative,)
        )


def test_local_indices_exceed_target_size_raises() -> None:
    model = _gaussian_model(
        target_keys=(PTAF_Datakey.BBOX_XYWH,),
        target_indices=(0, 1, 2, 3, 4),
    )
    with pytest.raises(ValueError):
        Vector1dErrorStackModel(
            keys=(PTAF_Datakey.BBOX_XYWH, PTAF_Datakey.RANGE_TO_COM),
            error_models=(model,),
        )


def test_local_indices_mapping() -> None:
    model = _uniform_model(
        low=1.0,
        high=1.0,
        target_keys=(PTAF_Datakey.BBOX_XYWH,),
        target_indices=(1, 3),
    )
    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.BBOX_XYWH, PTAF_Datakey.RANGE_TO_COM),
        error_models=(model,),
    )
    values = torch.zeros(5)
    out, err = stack.apply(values, return_error=True)

    expected = torch.zeros(5)
    expected[1] = 1.0
    expected[3] = 1.0

    assert torch.allclose(err, expected)
    assert torch.allclose(out, expected)


def test_target_keys_full_slice() -> None:
    model = _uniform_model(
        low=2.0,
        high=2.0,
        target_keys=(PTAF_Datakey.RANGE_TO_COM,),
    )
    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.BBOX_XYWH, PTAF_Datakey.RANGE_TO_COM),
        error_models=(model,),
    )
    values = torch.zeros(5)
    out = stack.apply(values)

    assert torch.allclose(out[:4], torch.zeros(4))
    assert out[4].item() == 2.0


def test_return_error_shapes() -> None:
    model = _uniform_model(low=1.0, high=1.0)
    stack = Vector1dErrorStackModel(
        keys=(PTAF_Datakey.BBOX_XYWH,), error_models=(model,)
    )
    values = torch.zeros((2, 4))
    out, err = stack.apply(values, return_error=True)

    assert out.shape == values.shape
    assert err.shape == values.shape


def test_param_validation_gaussian_std_negative() -> None:
    with pytest.raises(ValueError):
        Vector1dErrorModel(
            variable_name="x",
            shape=(1,),
            error_type=DistributionType.GAUSSIAN,
            parameters=GaussianParams(mean=0.0, std=-1.0),
        )


def test_param_validation_poisson_rate_negative() -> None:
    with pytest.raises(ValueError):
        Vector1dErrorModel(
            variable_name="x",
            shape=(1,),
            error_type=DistributionType.POISSON,
            parameters=PoissonParams(rate=-1.0),
        )


def test_param_validation_exponential_rate_zero() -> None:
    with pytest.raises(ValueError):
        Vector1dErrorModel(
            variable_name="x",
            shape=(1,),
            error_type=DistributionType.NEG_EXPONENTIAL,
            parameters=ExponentialParams(rate=0.0),
        )


def test_param_validation_uniform_low_gt_high() -> None:
    with pytest.raises(ValueError):
        Vector1dErrorModel(
            variable_name="x",
            shape=(1,),
            error_type=DistributionType.UNIFORM,
            parameters=UniformParams(low=2.0, high=1.0),
        )
