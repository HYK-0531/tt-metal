# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger
import os

from ..tt.group_norm import TtGroupNorm, TtGroupNormParameters
from ..tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc


def print_stats(label, data: torch.Tensor):
    data_ = data if isinstance(data, torch.Tensor) else ttnn.to_torch(data)
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.max()}, {data_.min()}]"


# @pytest.mark.parametrize("device_params", [{"trace_region_size": 40960}], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize(
    ("batch", "channels", "height", "width", "group_count", "num_out_blocks", "cores_y", "cores_x"),
    [
        (8, 768, 1, 512, 32, 2, 8, 8),
        # (512, 256, 256, 32),
        # (256, 512, 512, 32),
        # (512, 512, 512, 32),
        # (128, 1024, 1024, 32),
        # (256, 1024, 1024, 32),
    ],
)
def test_group_norm(
    *,
    device: ttnn.Device,
    batch: int,
    channels: int,
    height: int,
    width: int,
    group_count: int,
    num_out_blocks: int,
    cores_y: int,
    cores_x: int,
) -> None:
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)

    # torch_model = torch.nn.GroupNorm(num_groups=group_count, num_channels=channels)
    # torch_model.eval()

    # torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    inp = torch.randn((batch, channels, height, width), dtype=torch_dtype)
    torch_weight = torch.rand((channels,), dtype=torch.bfloat16)
    torch_bias = torch.rand((channels,), dtype=torch.bfloat16)

    f_path = "/localdev/proj_sw/user_dev/sadesoye/tt-metal/models/experimental/stable_diffusion_35_large/tests"

    torch.save(inp, os.path.join(f_path, "torch_input_tensor.pt"))
    torch.save(torch_weight, os.path.join(f_path, "torch_weight.pt"))
    torch.save(torch_bias, os.path.join(f_path, "torch_bias.pt"))

    # inp = torch.load(os.path.join(f_path,"torch_input_tensor.pt"))
    # torch_weight = torch.load(os.path.join(f_path,"torch_weight.pt"))
    # torch_bias = torch.load(os.path.join(f_path,"torch_bias.pt"))

    t_model_state = {"weight": torch_weight, "bias": torch_bias}
    out = torch.nn.functional.group_norm(inp, group_count, weight=torch_weight, bias=torch_bias)
    # t_model_state = torch_model.state_dict()

    parameters = TtGroupNormParameters.from_torch(
        t_model_state,
        input_width=width,
        input_height=height,
        num_channels=channels,
        num_groups=group_count,
        num_out_blocks=num_out_blocks,
        core_grid=ttnn.CoreGrid(x=cores_x, y=cores_y),
        device=device,
    )

    tt_model = TtGroupNorm(parameters, eps=0.0)
    # tt_model = TtGroupNorm(parameters, eps=torch_model.eps)

    # inp = torch.randn([batch, channels, height, width], dtype=torch_dtype)

    logger.info(f"device {device}")
    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp))

    # with torch.no_grad():
    #   out = torch_model(inp)

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    tt_out = tt_model(tt_inp)
    # breakpoint()

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch))
