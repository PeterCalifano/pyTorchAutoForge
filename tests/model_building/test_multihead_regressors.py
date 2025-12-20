import pytest
import torch
from pyTorchAutoForge.model_building.ModelAutoBuilder import (
    MultiHeadRegressor,
    CascadedHeadRegressor,
    EnumMultiHeadOutMode,
)

def test_multihead_regressor_concat_and_append_modes():
    """
    NOTE: Test automatically generated
    """
    # Prepare simple linear heads
    in_feat = 4
    head1 = torch.nn.Linear(in_feat, 3)
    head2 = torch.nn.Linear(in_feat, 2)

    # Concatenate mode (default) -> concatenated features along dim=1
    mh_concat = MultiHeadRegressor(model_heads=torch.nn.ModuleList([head1, head2]),
                                   output_mode=EnumMultiHeadOutMode.Concatenate)
    X = torch.randn(5, in_feat)
    out_concat = mh_concat(X)
    
    assert isinstance(out_concat, torch.Tensor)
    assert out_concat.shape == (5, 3 + 2)

    # Append mode -> returns list of outputs
    mh_append = MultiHeadRegressor(model_heads=torch.nn.ModuleList([head1, head2]),
                                   output_mode=EnumMultiHeadOutMode.Append)
    out_list = mh_append(X)
    
    assert isinstance(out_list, list)
    assert len(out_list) == 2
    assert out_list[0].shape == (5, 3)
    assert out_list[1].shape == (5, 2)


def test_cascaded_head_regressor_forward_and_concat_pack():
    """
    NOTE: Test automatically generated
    """
    # Design two heads for cascade:
    # head1 consumes 4 -> outputs 3
    # head2 will consume original 4 + 3 = 7 -> outputs 2
    head1 = torch.nn.Linear(4, 3)
    head2 = torch.nn.Linear(7, 2)

    casc = CascadedHeadRegressor(model_heads=torch.nn.ModuleList([head1, head2]),
                                 output_mode=EnumMultiHeadOutMode.Concatenate,
                                 concat_dim=1)
    X = torch.randn(6, 4)
    out = casc(X)
    
    # Predictions concatenated: 3 + 2 = 5 features
    assert isinstance(out, torch.Tensor)
    assert out.shape == (6, 5)


def test_headregressor_accepts_moduledict_and_single_module():
    """
    NOTE: Test automatically generated
    """
    # Test ModuleDict input and single module input
    head_single = torch.nn.Linear(3, 2)
    hr_single = MultiHeadRegressor(
        model_heads=head_single, output_mode=EnumMultiHeadOutMode.Append)
    
    inp = torch.randn(2, 3)
    out_single = hr_single(inp)

    assert isinstance(out_single, list)
    assert out_single[0].shape == (2, 2)

    head_dict = torch.nn.ModuleDict(
        {"a": torch.nn.Linear(3, 1), "b": torch.nn.Linear(3, 1)})
    
    hr_dict = MultiHeadRegressor(
        model_heads=head_dict, output_mode=EnumMultiHeadOutMode.Append)
    out_dict = hr_dict(inp)

    assert isinstance(out_dict, list)
    assert len(out_dict) == 2
    assert all(isinstance(t, torch.Tensor) for t in out_dict)
