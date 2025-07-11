# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from loguru import logger

from ...tt.vae_decoder.fun_linear import vae_linear, TtLinearParameters
from ...tt.utils import assert_quality, to_torch
from models.utility_functions import comp_allclose, comp_pcc
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


def print_stats(label, data: torch.Tensor, device=None):
    if isinstance(data, torch.Tensor):
        data_ = data
    else:
        data_ = ttnn.to_torch(
            data, mesh_composer=ttnn.ConcatMesh2dToTensor(device, mesh_shape=tuple(device.shape), dims=(0, 1))
        )
    return f"{label}: mean:{data_.mean()} , std:{data_.std()} , range:[{data_.min()}, {data_.max()}]"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}])
@pytest.mark.parametrize(
    ("batch", "in_channels", "out_channels", "height", "width"),
    [
        (1, 512, 512, 256, 256),
        # (512, 256, 256, 32),
        # (256, 512, 512, 32),
        # (512, 512, 512, 32),
        # (128, 1024, 1024, 32),
        # (256, 1024, 1024, 32),
    ],
)
def test_fun_linear(
    *, device: ttnn.Device, batch: int, in_channels: int, out_channels: int, height: int, width: int
) -> None:
    # torch_dtype = torch.float32
    torch_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16
    torch.manual_seed(0)
    logger.info(f"Device: {device}, {device.core_grid}")

    sd_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-3.5-large", subfolder="vae")
    print(sd_vae.decoder.mid_block)

    torch_model = torch.nn.Linear(in_features=in_channels, out_features=out_channels)
    torch_model.eval()

    parameters = TtLinearParameters.from_torch(
        torch_linear=torch_model, dtype=ttnn_dtype, device=device, core_grid=None
    )

    inp = torch.randn(batch, height, width, in_channels)
    # torch_input_padded = inp.permute(0, 2, 3, 1)
    # torch_input_padded = torch.nn.functional.pad(inp.permute(0, 2, 3, 1), (0, 8-in_channels)) #channel dimension is padded to 8

    tt_inp = ttnn.from_torch(inp, dtype=ttnn_dtype, device=device)

    logger.info(print_stats("torch_input", inp))
    logger.info(print_stats("tt_input", tt_inp, device=device))

    logger.info(f" input shape TT: {tt_inp.shape}, Torch: {inp.shape}")
    with torch.no_grad():
        out = torch_model(inp)

    tt_out = vae_linear(tt_inp, parameters)

    tt_out_torch = to_torch(tt_out)  # .permute(0, 3, 1, 2)

    logger.info(print_stats("torch", out))
    logger.info(print_stats("tt", tt_out_torch, device=device))

    assert_quality(out, tt_out_torch, pcc=0.94, ccc=0.94)
    print(comp_allclose(out, tt_out_torch))
    result, output = comp_pcc(out, tt_out_torch)
    logger.info(f"Comparison result Pass:{result}, Output {output}, in: {torch.count_nonzero(tt_out_torch)}")
