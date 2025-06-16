import ttnn


class MnistModel:
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.mm1_weight = ttnn.ones((784, 512), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
        self.mm1_bias = ttnn.ones((512,), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
        self.mm2_weight = ttnn.ones((512, 256), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
        self.mm2_bias = ttnn.ones((256,), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
        self.mm3_weight = ttnn.ones((256, 10), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
        self.mm3_bias = ttnn.ones((10,), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)

    def forward(self, x):
        y = ttnn.matmul(x, self.mm1_weight)
        ttnn.deallocate(x)
        x = ttnn.add(y, self.mm1_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.mm2_weight)
        x = ttnn.add(x, self.mm2_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.mm3_weight)
        x = ttnn.add(x, self.mm3_bias)
        x = ttnn.softmax(x)

        return x


def main():
    # Begin graph capture
    #
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    device = ttnn.open_device(device_id=0)
    input = ttnn.ones((1, 784), device=device, dtype=ttnn.float32, layout=ttnn.Layout.TILE)
    model = MnistModel(device)
    output = model.forward(input)

    captured_graph = ttnn.graph.end_graph_capture()

    print("test")
    ttnn.graph.visualize(captured_graph, file_name="graph.svg")
    # ttnn.graph.pretty_print(captured_graph)
    #
    # End graph capture

    print()
    print(captured_graph)
    print()

    # print("Output shape:", output.shape)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
