# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc

def test_ttnn_where(device):
    # ttt: tensor, tensor, tensor
    C = torch.ones(4, 4, dtype=torch.bfloat16)
    T = torch.randn(4, 4, dtype=torch.bfloat16)
    F = torch.ones(4, 4, dtype=torch.bfloat16) * 10
    golden_ttt = torch.where(C != 0, T, F)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result_ttt = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    result_ttt = ttnn.to_torch(ttnn_result_ttt)
    assert_with_pcc(result_ttt, golden_ttt)
    print("ttt result:")
    print(result_ttt)
    print("ttt golden:")
    print(golden_ttt)
    print()

    # tts: tensor, tensor, scalar
    scalar_F = 10.0
    golden_tts = torch.where(C != 0, T, scalar_F)
    ttnn_result_tts = ttnn.where(ttnn_C, ttnn_T, scalar_F)
    result_tts = ttnn.to_torch(ttnn_result_tts)
    assert_with_pcc(result_tts, golden_tts)
    print("tts result:")
    print(result_tts)
    print("tts golden:")
    print(golden_tts)
    print()

    # tst: tensor, scalar, tensor
    scalar_T = 5.0
    golden_tst = torch.where(C != 0, scalar_T, F)
    ttnn_result_tst = ttnn.where(ttnn_C, scalar_T, ttnn_F)
    result_tst = ttnn.to_torch(ttnn_result_tst)
    assert_with_pcc(result_tst, golden_tst)
    print("tst result:")
    print(result_tst)
    print("tst golden:")
    print(golden_tst)
    print()


def torch_equal_nan(a, b):
    return torch.all((a == b) | (torch.isnan(a) & torch.isnan(b)))


def test_ttnn_where_nan(device):
    tor_dtype = torch.float32

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1, float("inf"), float("nan")], dtype=tor_dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tor_dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tor_dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, float("nan"), 3.0, float("inf"), -float("inf"), -1.0, 0.0, -0.0, 42.49, -92.42, 111.0, 112.0],
        dtype=tor_dtype,
    )
    false_values = torch.tensor(
        [-1.0, 999.9, float("nan"), -float("inf"), float("inf"), 1.0, -0.0, 0.0, -3.14, 7.84, 222.0, 223.0],
        dtype=tor_dtype,
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)

    print("ttnn res", tt_result1)
    print("torch res", result1)
    print("\nttnn res all true", tt_result2, torch_equal_nan(tt_result2, true_values))
    print("torch res", result2, torch_equal_nan(result2, true_values))
    print("\nttnn res all false", tt_result3, torch_equal_nan(tt_result3, false_values))
    print("torch res", result3, torch_equal_nan(result3, false_values))

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("tor_dtype, ttnn_dtype", [(torch.bfloat16, ttnn.bfloat16), (torch.float32, ttnn.float32)])
def test_ttnn_where_mcw(h, w, tor_dtype, ttnn_dtype, device):
    # tor_dtype = torch.bfloat16
    # ttnn_dtype = ttnn.bfloat16
    C = torch.arange(h * w, dtype=tor_dtype)
    C = (C % 2).float()  # Alternates 0, 1, 0, 1, ...
    C = C.reshape(1, 1, h, w)
    C = C.expand(1, 1, h, w).to(tor_dtype)  # Broadcast to (n, c, h, w)
    T = torch.ones(1, 1, h, w, dtype=tor_dtype) * 4.0
    F = torch.ones(1, 1, h, w, dtype=tor_dtype) * 10.0
    golden = torch.where(C != 0, T, F)

    print("input ", C)

    ttnn_C = ttnn.from_torch(C, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_T = ttnn.from_torch(T, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_F = ttnn.from_torch(F, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_result = ttnn.where(ttnn_C, ttnn_T, ttnn_F)
    # print("ttnn_result", ttnn_result)
    result = ttnn.to_torch(ttnn_result)
    # torch.set_printoptions(linewidth=200, threshold = 10000 , precision=5, sci_mode = False, edgeitems=17)
    print("result", result, result.shape)
    print("golden", golden)
    assert torch.equal(result, golden)


def test_ttnn_where_nan_bf16(device):
    tor_dtype = torch.bfloat16
    ttnn_dtype = ttnn.bfloat16

    condition = torch.tensor([1, 0, -2, 0, 5, 0, 0, 8, 0, -1, float("inf"), float("nan")], dtype=tor_dtype)
    condition_all_ones = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tor_dtype)
    condition_all_zeros = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tor_dtype)

    # true and false value tensors
    true_values = torch.tensor(
        [1.0, float("nan"), 3.0, float("inf"), -float("inf"), -1.0, 0.0, -0.0, 42.49, -92.42, 111.0, 112.0],
        dtype=tor_dtype,
    )
    false_values = torch.tensor(
        [-1.0, 999.9, float("nan"), -float("inf"), float("inf"), 1.0, -0.0, 0.0, -3.14, 7.84, 222.0, 223.0],
        dtype=tor_dtype,
    )

    ttnn_condition = ttnn.from_torch(condition, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_condition_all_ones = ttnn.from_torch(
        condition_all_ones, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_condition_all_zeros = ttnn.from_torch(
        condition_all_zeros, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device
    )

    ttnn_true_values = ttnn.from_torch(true_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_false_values = ttnn.from_torch(false_values, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_result1 = ttnn.where(ttnn_condition, ttnn_true_values, ttnn_false_values)
    ttnn_result2 = ttnn.where(ttnn_condition_all_ones, ttnn_true_values, ttnn_false_values)
    ttnn_result3 = ttnn.where(ttnn_condition_all_zeros, ttnn_true_values, ttnn_false_values)

    # print("ttnn_result1", ttnn_result1)
    # print("ttnn_result2", ttnn_result2)
    # print("ttnn_result3", ttnn_result3)

    tt_result1 = ttnn.to_torch(ttnn_result1)
    tt_result2 = ttnn.to_torch(ttnn_result2)
    tt_result3 = ttnn.to_torch(ttnn_result3)

    # where operation in torch expects condition to be a boolean dtype, in ttnn.where we follow 0's & non-zero's (0's and 1's would be ideal)
    result1 = torch.where(condition.bool(), true_values, false_values)
    result2 = torch.where(condition_all_ones.bool(), true_values, false_values)
    result3 = torch.where(condition_all_zeros.bool(), true_values, false_values)

    print("ttnn res", tt_result1)
    print("torch res", result1)
    print("\nttnn res all true", tt_result2, torch_equal_nan(tt_result2, true_values))
    print("torch res", result2, torch_equal_nan(result2, true_values))
    print("\nttnn res all false", tt_result3, torch_equal_nan(tt_result3, false_values))
    print("torch res", result3, torch_equal_nan(result3, false_values))

    assert torch_equal_nan(tt_result1, result1)
    assert torch_equal_nan(tt_result2, result2)
    assert torch_equal_nan(tt_result3, result3)
