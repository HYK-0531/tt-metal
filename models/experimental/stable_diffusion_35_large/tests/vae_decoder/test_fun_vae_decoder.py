# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...reference.vae_decoder import VaeDecoder
from ...tt.vae_decoder.fun_vae_decoder import sd_vae_decode, TtVaeDecoderParameters
from ...tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}]"


# @pytest.mark.parametrize("device_params", [{"trace_region_size": 40960}], indirect=True)
# @pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
# @pytest.mark.usefixtures("use_program_cache")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize(
    (
        "batch",
        "in_channels",
        "out_channels",
        "layers_per_block",
        "height",
        "width",
        "norm_num_groups",
        "block_out_channels",
        "cores_y",
        "cores_x",
    ),
    [
        (1, 16, 3, 2, 32, 32, 32, (128, 256, 512, 512), 8, 8),  # slice 128, output blocks 32. Need to parametize
    ],
)
def test_vae_decoder(
    *,
    device: ttnn.Device,
    batch: int,
    in_channels: int,
    out_channels: int,
    layers_per_block: int,
    height: int,
    width: int,
    norm_num_groups: int,
    block_out_channels: list[int] | tuple[int, ...],
    cores_y: int,
    cores_x: int,
) -> None:
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    logger.info(f"Device: {device}, {device.core_grid}")

    torch_model = VaeDecoder(
        block_out_channels=block_out_channels,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        norm_num_groups=norm_num_groups,
    )

    # sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    # print(sd_vae.decoder)
    # torch_model=sd_vae.decoder.mid_block
    torch_model.eval()

    parameters = TtVaeDecoderParameters.from_torch(
        torch_vae_decoder=torch_model,
        dtype=ttnn_dtype,
        device=device,
        core_grid=ttnn.CoreGrid(x=cores_x, y=cores_y),
    )

    # inp = torch.randn(batch, in_channels, height, width)
    inp = torch.normal(1, 2, (batch, in_channels, height, width))

    tt_inp = ttnn.from_torch(inp.permute(0, 2, 3, 1), dtype=ttnn_dtype, device=device)

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    # tt_inp = allocate_tensor_on_device_like(tt_inp_host, device=mesh_device)
    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = sd_vae_decode(tt_inp, parameters, None)

    tt_out_torch = to_torch(tt_out).permute(0, 3, 1, 2)

    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))
    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
