# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import timm
import torch
import ttnn
from models.experimental.functional_vovnet.tt.vovnet import TtVoVNet
from models.experimental.functional_vovnet.reference.vovnet import VoVNet
from models.experimental.functional_vovnet.tt.model_preprocessing import custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "model_name, pcc",
    (("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", 0.99),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vovnet_model_inference(device, pcc, imagenet_sample_input, model_name, reset_seeds):
    Timm_model = timm.create_model(model_name, pretrained=True)
    Timm_model.eval()

    state_dict = Timm_model.state_dict()
    PT_model = VoVNet()
    res = PT_model.load_state_dict(state_dict)
    parameters = custom_preprocessor(device, state_dict)

    input = torch.rand(1, 3, 224, 224)
    STAGE_IDX = 2
    with torch.no_grad():
        model_output = PT_model(input, stage_idx=STAGE_IDX)
        # , stage_idx=STAGE_IDX)

    tt_model = TtVoVNet(
        device=device,
        # torch_model=torch_model.state_dict(),
        parameters=parameters,
        base_address="",
    )

    tt_input = ttnn.from_torch(input, dtype=ttnn.bfloat16, device=device)  # , layout = ttnn.TILE_LAYOUT)
    tt_output = tt_model.forward(tt_input, stage_idx=STAGE_IDX)
    # print("Shapes :", tt_output.shape, " ", model_output.shape)
    tt_output_torch = ttnn.to_torch(tt_output)  # .squeeze(0).squeeze(0)
    # tt_output_torch = torch.permute(tt_output_torch, (0, 3, 1, 2))
    # tt_output_torch = torch.reshape(tt_output_torch, model_output.shape)
    assert_with_pcc(model_output, tt_output_torch, 0.99)
