# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import compare_equal
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
import models.utility_functions as util


def print_unique_values(tensor):
    unique_values = torch.unique(tensor.to(torch.float32))
    unique_array = unique_values.numpy()
    print("Unique values:", unique_array)
    print("Min value:", torch.min(tensor).item())
    print("Max value:", torch.max(tensor).item())


def test_plot_binary_op_pow_looping_tensor(device):
    torch_binary_op = torch.pow
    ttnn_op = ttnn.pow
    low = -100
    high = 100
    x = torch.arange(low, high, 0.1, dtype=torch.float32)
    x_bf16 = x.to(torch.bfloat16)

    plot_dir = "accuracy_results/plots/pow_results/"
    csv_dir = "accuracy_results/csvs/pow_results/"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_results.csv")

    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["input_x", "input_y", "torch_result", "ttnn_result", "abs_error", "ulp_error"])

    scalar_values = []
    mean_ulp_errors = []
    max_ulp_errors = []

    for y_scalar in np.arange(1.0, 10.5, 0.5):
        y = torch.full_like(x, fill_value=y_scalar, dtype=torch.float32)
        y_bf16 = y.to(torch.bfloat16)

        torch_out = torch_binary_op(x, y)

        ttnn_x = ttnn.from_torch(x_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn_y = ttnn.from_torch(y_bf16, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn_out_res = ttnn.multiply(
            ttnn_x,
            ttnn_y,
            input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
            activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
            use_legacy=False,
        )
        ttnn_out = ttnn.to_torch(ttnn_out_res).to(torch.float32)

        abs_error = torch.abs(torch_out - ttnn_out)
        ulp_spacing = util.ulp(torch_out.to(torch.bfloat16)).to(torch.float32)
        ulp_error = abs_error / ulp_spacing

        # Filter valid ULP values to avoid NaNs/infs
        valid_mask = torch.isfinite(ulp_error)
        filtered_ulp_error = ulp_error[valid_mask]

        scalar_values.append(y_scalar)
        mean_ulp_errors.append(filtered_ulp_error.mean().item() if filtered_ulp_error.numel() > 0 else float("nan"))
        max_ulp_errors.append(filtered_ulp_error.max().item() if filtered_ulp_error.numel() > 0 else float("nan"))

        # CSV
        with open(csv_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            for i in range(len(x)):
                writer.writerow(
                    [
                        x[i].item(),
                        y_scalar,
                        torch_out[i].item(),
                        ttnn_out[i].item(),
                        abs_error[i].item(),
                        ulp_error[i].item(),
                    ]
                )

        # Output Comparison
        plt.plot(x.numpy(), torch_out.numpy(), label="torch", linewidth=1)
        plt.plot(x.numpy(), ttnn_out.numpy(), label="ttnn", linestyle="--", linewidth=1)
        plt.title(f"Output Comparison: {torch_binary_op.__name__}(x, y={y_scalar})\nInput Range: x ∈ [{low}, {high}]")
        plt.xlabel(f"x (with y = {y_scalar})")
        plt.ylabel(f"{torch_binary_op.__name__}(x, {y_scalar})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        filename = f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_y_{y_scalar}.png"
        save_path = os.path.join(plot_dir, filename)
        plt.savefig(save_path)
        plt.close()
        print(f"\n[x, y={y_scalar}] Output graph saved to {os.path.abspath(save_path)}")

        # ULP
        plt.figure(figsize=(10, 5))
        plt.plot(x.numpy(), ulp_error.numpy(), label="ULP Error", color="red", linewidth=1)
        plt.title(f"ULP Error: {ttnn_op.__name__} vs Torch\nInput Range: x ∈ [{low}, {high}], y = {y_scalar}")
        plt.xlabel(f"x (with y = {y_scalar})")
        plt.ylabel("ULP Error")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        ulp_filename = f"{ttnn_op.__name__}_bf16_range_{int(low)}_{int(high)}_y_{y_scalar}_ulp.png"
        ulp_path = os.path.join(plot_dir, ulp_filename)
        plt.savefig(ulp_path)
        plt.close()
        print(f"[x, y={y_scalar}] ULP graph saved to {os.path.abspath(ulp_path)}")

    plt.figure(figsize=(10, 5))
    plt.plot(scalar_values, mean_ulp_errors, label="Mean ULP Error", marker="o")
    plt.plot(scalar_values, max_ulp_errors, label="Max ULP Error", marker="x")
    plt.title(f"ULP Error Summary across Scalar y ∈ [1.0, 10.0]")
    plt.xlabel("Scalar y Value")
    plt.ylabel("ULP Error")
    plt.xticks(scalar_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    summary_path = os.path.join(plot_dir, f"{ttnn_op.__name__}_ulp_summary_vs_scalar.png")
    plt.savefig(summary_path)
    plt.close()
    print(f"\nULP summary graph saved to {os.path.abspath(summary_path)}")

    print(f"\nAll results saved to CSV: {os.path.abspath(csv_path)}")


@pytest.mark.parametrize(
    "input_shapes",
    ((torch.Size([1, 1, 32, 32])),),
)
@pytest.mark.parametrize("input_val", [-88, -89, -90, 4.3944, 20.7058])
def test_unary_max_fill_val_bf16(input_shapes, input_val, device):
    torch_input = torch.ones(input_shapes, dtype=torch.bfloat16) * input_val

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # tt_in = ttnn.fill(tt_in, input_val)

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    print("\nInput : ", torch_input[0, 0, 0, 0])
    print("\ngolden:", golden[0, 0, 0, 0], "\nTTNN:", result[0, 0, 0, 0])
    # assert_with_pcc(golden, result, 0.999)
    assert assert_with_ulp(golden, result)
    # assert compare_equal([tt_result], [golden])


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        # (torch.Size([1, 2, 64, 120])),
        # (torch.Size([1, 3, 320, 320])),
    ),
)
@pytest.mark.parametrize(
    "low, high",
    [
        (-100, 88),
    ],
)
def test_unary_max_bf16(input_shapes, low, high, device):
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    torch_input = torch.linspace(high, low, num_elements, dtype=torch.bfloat16)
    torch_input = torch_input[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.exp)
    golden = golden_function(torch_input, device=device)

    tt_in = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_result = ttnn.exp(tt_in)
    result = ttnn.to_torch(tt_result)
    # assert_with_pcc(golden, result, 0.999)
    print("\ngolden:", golden, "\nTTNN:", result)
    print_unique_values(golden - result)
    assert_with_ulp(golden, result)


def test_pow_bf16(device):
    torch_input_a = torch.tensor([9.0, 100000, 5.0], dtype=torch.float32)
    torch_input_b = torch.tensor([2.0, 1.7984, 3.0], dtype=torch.float32)

    golden_function = ttnn.get_golden_function(ttnn.pow)
    golden = golden_function(torch_input_a, torch_input_b, device=device)

    tt_ina = ttnn.from_torch(
        torch_input_a,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_inb = ttnn.from_torch(
        torch_input_b,
        dtype=ttnn.float32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_log = ttnn.log(tt_ina)
    print("tt logA - ", tt_log)
    print("torch logA - ", torch.log(torch_input_a))
    tt_mul = ttnn.multiply(tt_log, tt_inb)
    print("tt log mul - ", tt_mul)
    print("torch log mul - ", torch_input_b * torch.log(torch_input_a))
    tt_exp = ttnn.exp(tt_mul)
    print("tt exp - ", tt_exp)

    print("torch exp - ", torch.exp(torch_input_b * torch.log(torch_input_a)))

    tt_result = ttnn.multiply(
        tt_ina,
        tt_inb,
        input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.LOG)],
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.EXP, False)],
        use_legacy=False,
    )

    result = ttnn.to_torch(tt_result)
    # assert_with_pcc(golden, result, 0.999)
    print("\ngolden:", golden, "\nTTNN:", result)
    assert_with_ulp(golden, result)
