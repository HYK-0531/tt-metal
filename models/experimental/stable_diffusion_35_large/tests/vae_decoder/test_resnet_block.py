# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.vae_decoder.fun_resnet_block import resnet_block, TtResnetBlock2DParameters
from ...reference.vae_decoder import ResnetBlock2D
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
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize(
    ("batch", "in_channels", "out_channels", "height", "width", "num_groups", "num_out_blocks", "cores_y", "cores_x"),
    [
        (1, 3, 32, 480, 640, 32, 2, 8, 8),
        # (512, 256, 256, 32),
        # (256, 512, 512, 32),
        # (512, 512, 512, 32),
        # (128, 1024, 1024, 32),
        # (256, 1024, 1024, 32),
    ],
)
def test_resnet_block(
    *,
    device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    height: int,
    width: int,
    num_groups: int,
    num_out_blocks: int,
    cores_y: int,
    cores_x: int,
) -> None:
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    logger.info(f"Device: {device}, {device.core_grid}")

    torch_model = ResnetBlock2D(in_channels=in_channels, out_channels=out_channels, groups=num_groups)
    torch_model.eval()

    parameters = TtResnetBlock2DParameters.from_torch(
        resnet_block=torch_model,
        dtype=ttnn_dtype,
        device=device,
        core_grid=ttnn.CoreGrid(x=cores_x, y=cores_y),
        num_out_blocks=num_out_blocks,
    )

    inp = torch.normal(1, 2, (batch, in_channels, height, width))
    torch_input_tensor = torch.nn.functional.pad(
        torch_input_tensor.permute(0, 2, 3, 1), (0, 5)
    )  # channel dimension is padded to 8

    memory_config = ttnn.create_sharded_memory_config(
        [4800, 8],
        core_grid=device.core_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    tt_inp = ttnn.from_torch(
        inp.permute(0, 2, 3, 1),
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = resnet_block(tt_inp, parameters, None)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))
