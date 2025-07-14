import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import MoEGate
from models.demos.deepseek_v3.tt.moe_gate import TT_MoE_Gate
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
def test_moe_gate(hf_config, mesh_device, batch_size=32):
    torch.manual_seed(1000)
    torch_input = torch.randn(1, batch_size, 7168)
    reference_model = MoEGate(hf_config)
    # torch_input = torch.stack([torch.randperm(7168).float()*0.001 for _ in range(batch_size)], dim=0).unsqueeze(0)
    top8_experts_indices, top8_experts_weights = reference_model(torch_input)
    assert top8_experts_weights.shape == (batch_size, 8)

    state_dict = reference_model.state_dict()

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_moe_gate = TT_MoE_Gate(hf_config, state_dict, mesh_device, batch_size)
    tt_top8_experts_weights, tt_top8_experts_indices = tt_moe_gate(tt_input)

    # compare pcc of tt and torch top8_experts_weights
    tt_top8_experts_weights = ttnn.to_torch(ttnn.get_device_tensors(tt_top8_experts_weights)[0]).squeeze(0).squeeze(0)
    passing, pcc_message = comp_pcc(tt_top8_experts_weights, top8_experts_weights, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"top8_experts_weights output does not meet PCC requirement 0.99: {pcc_message}"

    # compare pcc of tt and torch top8_experts_indices
    tt_top8_experts_indices = ttnn.to_torch(ttnn.get_device_tensors(tt_top8_experts_indices)[0]).squeeze(0).squeeze(0)
    passing, pcc_message = comp_pcc(tt_top8_experts_indices, top8_experts_indices, 0.99)
    logger.info(f"PCC: {pcc_message}")
    assert passing, f"top8_experts_indices output does not meet PCC requirement 0.99: {pcc_message}"
