from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import enable_custom_op
from vllm_ascend.utils import is_310p as is_310p_hw
from vllm_ascend.ops.layernorm import AscendGemmaRMSNorm, AscendRMSNorm

enable_custom_op()


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 8, dtype=torch.float16)


def mock_rms_norm(x, weight, eps):
    return x + 1, None


def mock_add_rms_norm(x, residual, weight, eps):
    return 2 * x, None, 2 * residual


def mock_add_rms_norm_bias(x, residual, weight, bias, eps):
    if bias is None:
        return 2 * x, None, 2 * residual
    else:
        return 2 * x + bias, None, 2 * residual


@pytest.fixture(autouse=True)
def default_vllm_config():
    mock_config = MagicMock()
    mock_config.compilation_config.custom_ops = ["all"]

    with set_current_vllm_config(mock_config):
        yield mock_config


@pytest.mark.skip("Skip as register_kernels has NPU SocName checking in CANN 8.5.0.")
@pytest.mark.skipif(is_310p_hw(), reason="non_310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float32)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm, create=True)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias, create=True)
def test_RMSNorm_forward(
    mock_add_rms_norm_bias, mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config
):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rms_norm_bias.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1

        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@pytest.mark.skipif(not is_310p_hw(), reason="310P device unittest case.")
@pytest.mark.parametrize("residual", [None, torch.randn(4, 8, dtype=torch.float16)])
@patch("torch_npu.npu_rms_norm", side_effect=mock_rms_norm)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm)
def test_RMSNorm_forward_310p(mock_add_rmsnorm, mock_rmsnorm, residual, dummy_tensor, default_vllm_config):
    layer = RMSNorm(hidden_size=8, eps=1e-05)
    if residual is not None:
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = 2 * dummy_tensor
        expected_out_residual = 2 * residual
        mock_add_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)
        assert torch.allclose(out_residual, expected_out_residual)
    else:
        out_x = layer.forward_oot(dummy_tensor, residual)
        expected_out_x = dummy_tensor + 1
        mock_rmsnorm.assert_called_once()
        assert torch.allclose(out_x, expected_out_x)


@pytest.mark.parametrize(
    "layer_cls, weight_getter",
    [
        (AscendRMSNorm, lambda layer: layer.weight),
        (AscendGemmaRMSNorm, lambda layer: 1.0 + layer.weight),
    ],
)
@patch("vllm_ascend.ops.layernorm.enable_custom_op", return_value=True)
@patch("torch.compiler.is_compiling", return_value=True)
@patch("torch_npu.npu_add_rms_norm", side_effect=mock_add_rms_norm, create=True)
@patch("torch.ops._C_ascend.npu_add_rms_norm_bias", side_effect=mock_add_rms_norm_bias, create=True)
def test_rmsnorm_residual_compile_uses_torch_npu_fallback(
    mock_add_rms_norm_bias,
    mock_add_rmsnorm,
    mock_is_compiling,
    mock_enable_custom_op,
    layer_cls,
    weight_getter,
    dummy_tensor,
    default_vllm_config,
):
    residual = torch.randn(4, 8, dtype=torch.float16)
    layer = layer_cls(hidden_size=8, eps=1e-05)

    with patch("torch.ops.vllm.maybe_chunk_residual", side_effect=lambda _x, res: res):
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

    expected_out_x = 2 * dummy_tensor
    expected_out_residual = 2 * residual
    mock_add_rms_norm_bias.assert_not_called()
    mock_add_rmsnorm.assert_called_once()
    call_args = mock_add_rmsnorm.call_args.args
    assert call_args[0] is dummy_tensor
    assert call_args[1] is residual
    torch.testing.assert_close(call_args[2], weight_getter(layer))
    assert call_args[3] == 1e-05
    assert torch.allclose(out_x, expected_out_x)
    assert torch.allclose(out_residual, expected_out_residual)
