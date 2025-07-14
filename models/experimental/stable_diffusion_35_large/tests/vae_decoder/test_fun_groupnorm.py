# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.vae_decoder.fun_group_norm import TtGroupNormParameters, vae_group_norm
from ...tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.max()}, {data_.min()}]"


# @pytest.mark.parametrize("device_params", [{"trace_region_size": 40960}], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize(
    ("batch", "channels", "height", "width", "group_count", "cores_y", "cores_x"),
    [
        # (8, 512, 32, 32, 32, 8, 8),
        (1, 128, 32, 32, 32, 8, 8),  # Group norm observed edge case
        # (512, 256, 256, 32),
        # (256, 512, 512, 32),
        # (512, 512, 512, 32),
        # (128, 1024, 1024, 32),
        # (256, 1024, 1024, 32),
    ],
)
def test_group_norm(
    *,
    mesh_device: ttnn.MeshDevice,
    batch: int,
    channels: int,
    height: int,
    width: int,
    group_count: int,
    cores_y: int,
    cores_x: int,
) -> None:
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    torch_model.eval()

    inp = torch.randn((batch, channels, height, width), dtype=torch_dtype)

    parameters = TtGroupNormParameters.from_torch(
        torch_model,
        core_grid=ttnn.CoreGrid(x=cores_x, y=cores_y),
        device=mesh_device,
    )

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=mesh_device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = vae_group_norm(tt_inp, parameters, torch_model.eps)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=mesh_device))
    logger.info(print_stats("in", inp))
