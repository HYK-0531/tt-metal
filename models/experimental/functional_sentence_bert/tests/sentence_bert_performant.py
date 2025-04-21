# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.functional_sentence_bert.tests.sentence_bert_test_infra import create_test_infra

try:
    from tracy import signpost

    use_signpost = True

except ModuleNotFoundError:
    use_signpost = False


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


def buffer_address(tensor):
    addr = []
    for ten in ttnn.get_device_tensors(tensor):
        addr.append(ten.buffer_address())
    return addr


# TODO: Create ttnn apis for this
ttnn.buffer_address = buffer_address


def run_sbert_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
    )

    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()
    # Optimized run
    print("opt start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    print("opt run start")
    test_infra.run()
    print("opt run end")
    test_infra.validate()
    test_infra.dealloc_output()

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    test_infra.run()
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()
    test_infra.dealloc_output()


def run_sbert_trace_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device=device,
        batch_size=device_batch_size,
    )
    tt_inputs_host, input_mem_config = test_infra.setup_l1_sharded_input(device)

    # First run configures convs JIT
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    spec = test_infra.input_tensor.spec
    print("is is1", tt_inputs_host.shape, tt_inputs_host.memory_config(), tt_inputs_host.layout)
    print(
        "is is2", test_infra.input_tensor.shape, test_infra.input_tensor.memory_config(), test_infra.input_tensor.layout
    )
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()

    # Optimized run
    # print("opt start")
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)
    # print("opt run start")
    test_infra.run()
    # print("opt run end")
    test_infra.validate()
    # print("opt valid")

    # Capture
    test_infra.input_tensor = tt_inputs_host.to(device, input_mem_config)

    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    tt_image_res = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(tt_image_res)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    if use_signpost:
        signpost(header="stop")
    test_infra.validate()

    ttnn.release_trace(device, tid)
    test_infra.dealloc_output()


def sbert_trace_2cqs_inference(
    device,
    device_batch_size,
):
    test_infra = create_test_infra(
        device,
        device_batch_size,
    )
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(device)
    tt_image_res = tt_inputs_host.to(device, ttnn.DRAM_MEMORY_CONFIG)
    print("in1", tt_image_res.shape, tt_image_res.layout, tt_image_res.dtype, tt_image_res.memory_config())
    print("in2", tt_inputs_host.shape, tt_inputs_host.layout, tt_inputs_host.dtype, tt_inputs_host.memory_config())
    # ss
    # Initialize the op event so we can write
    op_event = ttnn.record_event(device, 0)

    # First run configures convs JIT
    ttnn.wait_for_event(1, op_event)
    print("before")
    p(tt_inputs_host, "host")
    p(tt_image_res, "dev tensor")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    print("after")
    p(tt_inputs_host, "host")
    p(tt_image_res, "dev tensor")
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    print(
        "before conversion",
        tt_image_res.shape,
        tt_image_res.layout,
        tt_image_res.dtype,
        tt_image_res.memory_config(),
        input_mem_config,
    )
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    spec = test_infra.input_tensor.spec
    op_event = ttnn.record_event(device, 0)
    test_infra.run()
    test_infra.validate()
    test_infra.dealloc_output()
    # Optimized run
    ttnn.wait_for_event(1, op_event)
    print("opt run befwefbwe")
    p(tt_inputs_host, "host")
    p(tt_image_res, "dev tensor")
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    print("uefbewuefbwufe")
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.run()
    test_infra.validate()
    # Capture
    ttnn.wait_for_event(1, op_event)
    ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
    write_event = ttnn.record_event(device, 1)
    ttnn.wait_for_event(0, write_event)
    test_infra.input_tensor = ttnn.to_memory_config(tt_image_res, input_mem_config)
    op_event = ttnn.record_event(device, 0)
    test_infra.dealloc_output()
    trace_input_addr = ttnn.buffer_address(test_infra.input_tensor)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    test_infra.run()
    input_tensor = ttnn.allocate_tensor_on_device(spec, device)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    assert trace_input_addr == ttnn.buffer_address(input_tensor)

    # More optimized run with caching
    if use_signpost:
        signpost(header="start")
    for iter in range(0, 2):
        print("iter is", iter)
        ttnn.wait_for_event(1, op_event)
        ttnn.copy_host_to_device_tensor(tt_inputs_host, tt_image_res, 1)
        write_event = ttnn.record_event(device, 1)
        ttnn.wait_for_event(0, write_event)
        # TODO: Add in place support to ttnn to_memory_config
        if tt_image_res.is_sharded():
            input_tensor = ttnn.reshard(tt_image_res, input_mem_config, input_tensor)
        op_event = ttnn.record_event(device, 0)
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    if use_signpost:
        signpost(header="stop")

    ttnn.release_trace(device, tid)
