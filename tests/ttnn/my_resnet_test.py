# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import time
from loguru import logger

from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra


class Conv2dArgs:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias: bool = bias

        self.bias = -1  # TODO!!!


class MaxPool2dArgs:
    def __init__(self, kernel_size, stride, padding):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class Downsample:
    def __init__(self, device, in_channels, out_channels, kernel_size, stride):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            device=self.device,
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            batch_size=self.conv1.batch_size,
            input_height=self.conv1.input_height,
            input_width=self.conv1.input_width,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation,
            bias_tensor=self.conv1_bias_tensor,
        )
        x = ttnn.batch_norm(x)
        return x


class Bottleneck:
    expansion = 4

    def __init__(self, device, in_channels, out_channels, downsample_layer=None, stride=1):
        self.device = device

        self.conv1 = Conv2dArgs(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1_num_features = out_channels

        self.conv2 = Conv2dArgs(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2_num_features = out_channels

        self.conv3 = Conv2dArgs(
            in_channels=out_channels,
            out_channels=out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3_num_features = out_channels * self.expansion

        # Some Bottlenecks will have a downsample layer
        self.downsample_layer = downsample_layer
        self.stride = stride

    def forward(self, x: ttnn.Tensor):
        y = x  # copy input

        # First block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv1_weight_tensor,
            device=self.device,
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            batch_size=self.conv1.batch_size,
            input_height=self.conv1.input_height,
            input_width=self.conv1.input_width,
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            dilation=self.conv1.dilation,
            bias_tensor=self.conv1_bias_tensor,
        )
        x = ttnn.batch_norm(x)
        x = ttnn.relu(x)

        # Second block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv2_weight_tensor,
            device=self.device,
            in_channels=self.conv2.in_channels,
            out_channels=self.conv2.out_channels,
            batch_size=self.conv2.batch_size,
            input_height=self.conv2.input_height,
            input_width=self.conv2.input_width,
            kernel_size=self.conv2.kernel_size,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            dilation=self.conv2.dilation,
            bias_tensor=self.conv2_bias_tensor,
        )
        x = ttnn.batch_norm(x)
        x = ttnn.relu(x)

        # Third block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv3_weight_tensor,
            device=self.device,
            in_channels=self.conv3.in_channels,
            out_channels=self.conv3.out_channels,
            batch_size=self.conv3.batch_size,
            input_height=self.conv3.input_height,
            input_width=self.conv3.input_width,
            kernel_size=self.conv3.kernel_size,
            stride=self.conv3.stride,
            padding=self.conv3.padding,
            dilation=self.conv3.dilation,
            bias_tensor=self.conv3_bias_tensor,
        )
        x = ttnn.batch_norm(x)

        # Downsample if needed
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)

        # Add identity
        x += y
        x = ttnn.relu(x)

        return x


class Resnet50:
    def __init__(self, device, num_channels=3):
        self.device = device

        self.layer_list = [3, 4, 6, 3]
        self.in_channels = 64

        self.conv1_args = Conv2dArgs(
            in_channels=num_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        bn1_num_features = 64
        # relu

        self.max_pool_args = MaxPool2dArgs(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(self.layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(self.layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(self.layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(self.layer_list[3], planes=512, stride=2)

        # ttnn.global_avg_pool2d

        ttnn.linear_args = ()

    def forward(self, x):
        # TODO
        return x

    def _make_layer(self, blocks, planes, stride=1):
        downsample_layer = None
        layers = []

        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample_layer = Downsample(
                device=self.device,
                in_channels=self.in_channels,
                out_channels=planes * Bottleneck.expansion,
                kernel_size=1,
                stride=stride,
            )

        layers.append(
            Bottleneck(self.device, self.in_channels, planes, downsample_layer=downsample_layer, stride=stride)
        )
        self.in_channels = planes * Bottleneck.expansion

        for i in range(blocks - 1):
            layers.append(Bottleneck(self.device, self.in_channels, planes))

        return layers


def time_run(test_infra, tt_inputs_host, input_mem_config):
    start_time = time.time()
    test_infra.input_tensor = tt_inputs_host.to(test_infra.device, input_mem_config)
    test_infra.run()
    end_time = time.time()
    time_taken = end_time - start_time
    fps = test_infra.batch_size / time_taken
    return time_taken, fps


def run_resnet_50(
    device,
    batch_size,
    act_dtype,
    weight_dtype,
    math_fidelity,
    use_pretrained_weight,
    model_location_generator,
):
    if (device.compute_with_storage_grid_size().x, device.compute_with_storage_grid_size().y) == (8, 7):
        pytest.skip("Test is not supported on n300 (8,7) grid")

    if batch_size > 16 and not is_blackhole():
        pytest.skip("Batch size > 16 is not supported on non-blackhole devices")

    test_infra = create_test_infra(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        math_fidelity,
        use_pretrained_weight,
        model_location_generator=model_location_generator,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    # First run configures convs JIT
    time_taken, fps = time_run(test_infra, tt_inputs_host, input_mem_config)
    print(f"Time taken for initial run: {time_taken:.5f} seconds, FPS: {fps:.5f}")

    # Optimized runs
    for idx in range(10):
        time_taken, fps = time_run(test_infra, tt_inputs_host, input_mem_config)
        print(f"Time taken for optimized run {idx+1}: {time_taken:.5f} seconds, FPS: {fps:.5f}")

    # Begin graph capture
    #
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    test_infra.input_tensor = tt_inputs_host.to(test_infra.device, input_mem_config)
    test_infra.run()
    captured_graph = ttnn.graph.end_graph_capture()
    #
    # End graph capture

    ttnn.graph.pretty_print(captured_graph)
    # ttnn.graph.visualize(captured_graph, file_name="graph.svg")

    # Dump the captured graph
    #
    with open("dump.txt", "w") as f:
        f.write(str(captured_graph))

    passed, message = test_infra.validate()
    assert passed, message


if __name__ == "__main__":
    logger.remove()

    device = ttnn.open_device(device_id=0, l1_small_size=24576)

    run_resnet_50(
        device,
        batch_size=16,
        act_dtype=ttnn.bfloat8_b,
        weight_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        use_pretrained_weight=False,
        model_location_generator=None,
    )
