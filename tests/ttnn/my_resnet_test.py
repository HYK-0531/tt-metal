# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

# import ttnn.database
import time
import math
from loguru import logger

# from models.demos.ttnn_resnet.tests.resnet50_test_infra import create_test_infra

DTYPE = ttnn.bfloat16


def get_input_height(x, batch_size):
    return int(math.sqrt(x.shape[-2] // batch_size))


def get_input_width(x, batch_size):
    return get_input_height(x, batch_size)


class Conv2dArgs:
    def __init__(self, batch_size, in_channels, out_channels, kernel_size, stride, padding, bias):
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.bias: bool = bias

    def __repr__(self):
        return f"Conv2dArgs(batch_size={self.batch_size}, in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, bias={self.bias})"


class MaxPool2dArgs:
    def __init__(self, batch_size, channels, kernel_size, stride, padding, dilation, ceil_mode):
        self.batch_size = batch_size
        self.channels = channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (padding, padding)
        self.dilation = (dilation, dilation)
        self.ceil_mode = ceil_mode


class Downsample:
    def __init__(self, device, batch_size, in_channels, out_channels, kernel_size, stride):
        self.device = device

        self.conv1 = Conv2dArgs(
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False,
        )

    def __call__(self, x):
        y = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.ones(
                shape=[
                    self.conv1.out_channels,
                    self.conv1.in_channels,
                    self.conv1.kernel_size[0],
                    self.conv1.kernel_size[1],
                ],
                dtype=DTYPE,
            ),
            device=self.device,
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            batch_size=self.conv1.batch_size,
            input_height=get_input_height(x, self.conv1.batch_size),
            input_width=get_input_width(x, self.conv1.batch_size),
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            bias_tensor=ttnn.ones(shape=[self.conv1.out_channels], dtype=DTYPE) if self.conv1.bias else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        # x = ttnn.batch_norm(x)
        return y


class Bottleneck:
    expansion = 4

    def __init__(self, device, batch_size, in_channels, out_channels, downsample_layer=None, stride=1):
        self.device = device
        self.batch_size = batch_size

        self.conv1 = Conv2dArgs(
            batch_size=batch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1_num_features = out_channels

        self.conv2 = Conv2dArgs(
            batch_size=batch_size,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2_num_features = out_channels

        self.conv3 = Conv2dArgs(
            batch_size=batch_size,
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

    def __call__(self, x: ttnn.Tensor):
        identity = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # copy input

        # First block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.ones(
                shape=[
                    self.conv1.out_channels,
                    self.conv1.in_channels,
                    self.conv1.kernel_size[0],
                    self.conv1.kernel_size[1],
                ],
                dtype=DTYPE,
            ),
            device=self.device,
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            batch_size=self.conv1.batch_size,
            input_height=get_input_height(x, self.conv1.batch_size),
            input_width=get_input_width(x, self.conv1.batch_size),
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            bias_tensor=ttnn.ones(shape=[self.conv1.out_channels], dtype=DTYPE) if self.conv1.bias else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # x = ttnn.batch_norm(x)
        x = ttnn.relu(x)

        # Second block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.ones(
                shape=[
                    self.conv2.out_channels,
                    self.conv2.in_channels,
                    self.conv2.kernel_size[0],
                    self.conv2.kernel_size[1],
                ],
                dtype=DTYPE,
            ),
            device=self.device,
            in_channels=self.conv2.in_channels,
            out_channels=self.conv2.out_channels,
            batch_size=self.conv2.batch_size,
            input_height=get_input_height(x, self.conv2.batch_size),
            input_width=get_input_width(x, self.conv2.batch_size),
            kernel_size=self.conv2.kernel_size,
            stride=self.conv2.stride,
            padding=self.conv2.padding,
            bias_tensor=ttnn.ones(shape=[self.conv2.out_channels], dtype=DTYPE) if self.conv2.bias else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # x = ttnn.batch_norm(x)
        x = ttnn.relu(x)

        # Third block
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.ones(
                shape=[
                    self.conv3.out_channels,
                    self.conv3.in_channels,
                    self.conv3.kernel_size[0],
                    self.conv3.kernel_size[1],
                ],
                dtype=DTYPE,
            ),
            device=self.device,
            in_channels=self.conv3.in_channels,
            out_channels=self.conv3.out_channels,
            batch_size=self.conv3.batch_size,
            input_height=get_input_height(x, self.conv3.batch_size),
            input_width=get_input_width(x, self.conv3.batch_size),
            kernel_size=self.conv3.kernel_size,
            stride=self.conv3.stride,
            padding=self.conv3.padding,
            bias_tensor=ttnn.ones(shape=[self.conv3.out_channels], dtype=DTYPE) if self.conv3.bias else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # x = ttnn.batch_norm(x)

        # Downsample if needed
        if self.downsample_layer is not None:
            identity = self.downsample_layer(identity)

        # Add identity
        identity = ttnn.to_memory_config(identity, x.memory_config())
        x += identity
        x = ttnn.relu(x)

        return x


class Resnet50:
    def __init__(self, device, batch_size, num_channels=3):
        self.device = device
        self.batch_size = batch_size

        self.layer_list = [3, 4, 6, 3]
        self.in_channels = 64

        self.conv1 = Conv2dArgs(
            batch_size=batch_size,
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
            batch_size=batch_size,
            channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            dilation=1,
            ceil_mode=False,
        )

        self.layer1 = self._make_layer(self.layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(self.layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(self.layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(self.layer_list[3], planes=512, stride=2)

        # ttnn.global_avg_pool2d

        ttnn.linear_args = ()

    def __call__(self, x):
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=ttnn.ones(
                shape=[
                    self.conv1.out_channels,
                    self.conv1.in_channels,
                    self.conv1.kernel_size[0],
                    self.conv1.kernel_size[1],
                ],
                dtype=DTYPE,
            ),
            device=self.device,
            in_channels=self.conv1.in_channels,
            out_channels=self.conv1.out_channels,
            batch_size=self.conv1.batch_size,
            input_height=get_input_height(x, self.conv1.batch_size),
            input_width=get_input_width(x, self.conv1.batch_size),
            kernel_size=self.conv1.kernel_size,
            stride=self.conv1.stride,
            padding=self.conv1.padding,
            bias_tensor=ttnn.ones(shape=[self.conv1.out_channels], dtype=DTYPE) if self.conv1.bias else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # x = ttnn.batch_norm(x)  # TODO
        x = ttnn.relu(x)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.max_pool_args.batch_size,
            input_h=get_input_height(x, self.max_pool_args.batch_size),
            input_w=get_input_width(x, self.max_pool_args.batch_size),
            channels=self.max_pool_args.channels,
            kernel_size=self.max_pool_args.kernel_size,
            stride=self.max_pool_args.stride,
            padding=self.max_pool_args.padding,
            dilation=self.max_pool_args.dilation,
            ceil_mode=self.max_pool_args.ceil_mode,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)

        x = ttnn.global_avg_pool2d(x)
        # ttnn.linear

        return x

    def _make_layer(self, blocks, planes, stride=1):
        downsample_layer = None
        layers = []

        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample_layer = Downsample(
                device=self.device,
                batch_size=self.batch_size,
                in_channels=self.in_channels,
                out_channels=planes * Bottleneck.expansion,
                kernel_size=1,
                stride=stride,
            )

        layers.append(
            Bottleneck(
                self.device, self.batch_size, self.in_channels, planes, downsample_layer=downsample_layer, stride=stride
            )
        )
        self.in_channels = planes * Bottleneck.expansion

        for i in range(blocks - 1):
            layers.append(Bottleneck(self.device, self.batch_size, self.in_channels, planes))

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
):
    resnet = Resnet50(device, batch_size)

    x = ttnn.ones(
        shape=[1, 1, batch_size * 224 * 224, 3],
        dtype=DTYPE,
        device=device,
    )

    out = resnet(x)

    # # Begin graph capture
    # #
    # ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    # # test_infra.input_tensor = tt_inputs_host.to(test_infra.device, input_mem_config)
    # # test_infra.run()
    # captured_graph = ttnn.graph.end_graph_capture()
    # #
    # # End graph capture

    # ttnn.graph.pretty_print(captured_graph)
    # # ttnn.graph.visualize(captured_graph, file_name="graph.svg")

    # # Dump the captured graph
    # #
    # with open("dump.txt", "w") as f:
    #     f.write(str(captured_graph))

    # passed, message = test_infra.validate()
    # assert passed, message

    return out


def test_resnet_50():
    logger.remove()

    # device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=36000)
    # device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=24576)
    # device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=12544)
    device = ttnn.open_device(device_id=0, l1_small_size=12544)

    out = run_resnet_50(
        device,
        batch_size=8,
    )

    print(f"Output shape: {out.shape}")


if __name__ == "__main__":
    test_resnet_50()
