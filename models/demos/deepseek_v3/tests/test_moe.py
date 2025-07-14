import os

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tt.moe import TT_MoE
from models.utility_functions import comp_pcc


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing."""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    return config


@pytest.mark.parametrize(
    "device_params",
    [
        {"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_2D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (4, 8)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
def test_moe(hf_config, mesh_device, batch_size=32):
    torch.manual_seed(1000)
    reference_model = DeepseekV3MoE(hf_config)
    torch_input = torch.randn(1, batch_size, 7168)
    torch_input = torch_input.repeat(1, 4, 1)
    torch_output = reference_model(torch_input)

    state_dict = reference_model.state_dict()

    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(2, None), mesh_shape=list(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_moe = TT_MoE(hf_config, state_dict, mesh_device, batch_size)
    tt_output = tt_moe(tt_input)
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=list(mesh_device.shape)),
    )
    # last all reduce in torch
    tt_output_torch = tt_output_torch.sum(dim=0).squeeze(0)
    # Compare outputs
    pcc_required = 0.99  # Embedding should be exact match (just lookup)
    passing, pcc_message = comp_pcc(torch_output, tt_output_torch, pcc_required)
    assert passing, f"MoE output does not meet PCC requirement {pcc_required}: {pcc_message}"
