import torch
import pytest
from pyTorchAutoForge.model_building.fullyConnectedBlocks import (
    FullyConnectedBlock,
    FullyConnectedBlockConfig,
    FullyConnectedBlocksStack,
    FullyConnectedResidualBlockStack,
)

# Tests for FullyConnectedBlock construction
def test_forward_default():
    block = FullyConnectedBlock(10, 5)
    x = torch.randn(2, 10)
    out = block(x)
    assert out.shape == (2, 5)


def test_activation_relu():
    block = FullyConnectedBlock(10, 5, activ_type="relu")
    x = torch.randn(3, 10)
    out = block(x)
    assert torch.all(
        out >= 0), "ReLU activation should output non-negative values."


def test_activation_prelu_unique():
    block = FullyConnectedBlock(
        10, 5, activ_type="prelu", prelu_params="unique")
    x = torch.randn(3, 10)
    out = block(x)
    assert out.shape == (3, 5)


def test_activation_sigmoid():
    block = FullyConnectedBlock(10, 5, activ_type="sigmoid")
    x = torch.randn(3, 10)
    out = block(x)
    assert torch.all((out >= 0) & (out <= 1)
                     ), "Sigmoid activation should output values in [0, 1]."


def test_activation_tanh():
    block = FullyConnectedBlock(10, 5, activ_type="tanh")
    x = torch.randn(3, 10)
    out = block(x)
    assert torch.all((out >= -1) & (out <= 1)
                     ), "Tanh activation should output values in [-1, 1]."


def test_regularizer_dropout():
    block = FullyConnectedBlock(
        10, 5, regularizer_type="dropout", regularizer_param=0.5)
    block.train()
    x = torch.ones(4, 10)
    out = block(x)
    assert (out == 0).any(), "Dropout should zero out some elements."


def test_regularizer_batchnorm():
    block = FullyConnectedBlock(10, 4, regularizer_type="batchnorm")
    x = torch.randn(3, 10)
    out = block(x)
    assert out.shape == (3, 4)


def test_regularizer_groupnorm():
    block = FullyConnectedBlock(
        10, 6, regularizer_type="groupnorm", regularizer_param=2)
    x = torch.randn(3, 10)
    out = block(x)
    assert out.shape == (3, 6)

def test_invalid_activation():
    with pytest.raises(ValueError):
        FullyConnectedBlock(10, 5, activ_type="invalid")  # type: ignore

def test_invalid_regularizer():
    with pytest.raises(ValueError):
        FullyConnectedBlock(10, 5, regularizer_type="invalid")  # type: ignore


# Test initialization methods
@pytest.mark.parametrize("init_method", ["xavier_uniform", "kaiming_uniform", "xavier_normal", "kaiming_normal", "orthogonal"])
def test_init_methods(init_method):
    block = FullyConnectedBlock(10, 5, init_method=init_method)
    x = torch.randn(3, 10)
    out = block(x)
    assert out.shape == (3, 5)

# Test FullyConnectedBlocksStack with multiple blocks
def test_blocks_stack_add_and_forward():
    cfgs = [FullyConnectedBlockConfig(
        in_channels=4, out_channels=4) for _ in range(2)]
    stack = FullyConnectedBlocksStack(cfgs)
    assert len(stack.blocks) == 2
    x = torch.randn(3, 4)
    out = stack(x)
    assert out.shape == (3, 4)


def test_blocks_stack_add_block_method():
    cfg = FullyConnectedBlockConfig(in_channels=2, out_channels=2)
    stack = FullyConnectedBlocksStack()
    stack.add_block(cfg)
    assert len(stack.blocks) == 1
    x = torch.randn(5, 2)
    out = stack(x)
    assert out.shape == (5, 2)

# Tests for FullyConnectedResidualBlockStack
@pytest.mark.parametrize("bad_input", ["a", 1, None])
def test_residual_stack_invalid_input_skip_indices_type(bad_input):
    with pytest.raises(TypeError):
        FullyConnectedResidualBlockStack(
            input_skip_indices=bad_input, output_skip_indices=[0])

@pytest.mark.parametrize("bad_input", [[1, "b"], [None], [1.5]])
def test_residual_stack_invalid_input_skip_indices_contents(bad_input):
    with pytest.raises(TypeError):
        FullyConnectedResidualBlockStack(
            input_skip_indices=bad_input, output_skip_indices=[0])

@pytest.mark.parametrize("bad_output", ["a", 1, None])
def test_residual_stack_invalid_output_skip_indices_type(bad_output):
    with pytest.raises(TypeError):
        FullyConnectedResidualBlockStack(
            input_skip_indices=[0], output_skip_indices=bad_output)


@pytest.mark.parametrize("bad_output", [[1, "b"], [None], [2.5]])
def test_residual_stack_invalid_output_skip_indices_contents(bad_output):
    with pytest.raises(TypeError):
        FullyConnectedResidualBlockStack(
            input_skip_indices=[0], output_skip_indices=bad_output)

def test_residual_stack_buffers_and_forward_shape():
    input_indices = [0, 2]
    output_indices = [1, 3]
    cfg = FullyConnectedBlockConfig(in_channels=4, out_channels=4)
    stack = FullyConnectedResidualBlockStack(
        input_skip_indices=input_indices,
        output_skip_indices=output_indices,
        block_cfg_list=[cfg],
    )

    print(stack._buffers)
    
    # Verify buffers
    assert torch.all(stack._buffers["input_skip_indices"] == torch.tensor(input_indices))
    assert torch.all(stack._buffers["output_skip_indices"] == torch.tensor(output_indices))

    # Forward pass shape
    x = torch.randn(5, 4)
    out = stack(x)
    assert out.shape == (5, 4)

