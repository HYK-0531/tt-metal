import pytest
import torch
from loguru import logger

import ttnn
from models.demos.t3000.mixtral8x7b.reference.model import TransformerBlock
from models.tt_transformers.tt.common import precompute_freqs
from models.tt_transformers.tt.decoder import TransformerBlock as TtTransformerBlock
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.rope import RotarySetup
from models.utility_functions import comp_allclose, comp_pcc
from ttnn import ReplicateTensorToMesh

# pytest -svv models/tt_transformers/tests/test_mixtral_decoder.py::test_mixtral_decoder_inference[wormhole_b0-True-16]


def prepare_inputs_ttnn(x_bsh, hidden_size, mesh_device):
    """
    Prepare inputs for decode mode.
    x: (batch, seq, hidden_dim)
    B: batch (32)
    S: sequence len (1)
    H: dim (4096)
    """
    assert x_bsh.size(2) == hidden_size
    assert len(x_bsh.size()) == 3

    batch = x_bsh.size(0)
    seq_len = x_bsh.size(1)
    assert seq_len == 1, "Only supporting decode mode"

    x_1SBH = x_bsh.view(1, seq_len, batch, hidden_size)
    # Pad small batches to 32
    if batch < 32:
        zeros = torch.zeros(1, seq_len, 32, hidden_size)
        zeros[:, :, :batch, :] = x_1SBH
        x_1SBH = zeros

    # input goes to L1
    xs_1SBH = ttnn.from_torch(
        x_1SBH,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )

    return xs_1SBH


def get_single_rot_mat(dhead, mesh_device, start_pos=0, theta: float = 1000000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dhead, 2)[: (dhead // 2)].float() / dhead))
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    rot_matrix = torch.zeros(dhead, dhead)
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_matrix[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_matrix[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()
    rot_matrix = rot_matrix.transpose(-1, -2)

    # Support for start_pos different than 0
    freqs = start_pos * freqs
    sin_freqs, cos_freqs = torch.sin(freqs), torch.cos(freqs)
    current_rot_mat = torch.zeros(dhead, dhead)
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    current_rot_mat[torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    current_rot_mat[torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    return ttnn.from_torch(
        current_rot_mat.unsqueeze(0).unsqueeze(0).transpose(-1, -2),  # 1,1,head_dim,head_dim
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    ), ttnn.from_torch(
        rot_matrix.unsqueeze(0).unsqueeze(0),  # 1,1,head_dim,head_dim
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ReplicateTensorToMesh(mesh_device),
    )


@pytest.mark.parametrize(
    "batch",
    (
        32,
        16,
    ),
)
def test_mixtral_decoder_inference(t3k_mesh_device, reset_seeds, batch):
    """
    b: batch
    s: sequence length
    h: hidden size
    """

    pcc = 0.99
    dtype = ttnn.bfloat8_b
    mode = "decode"

    if batch == 32:
        generation_start_pos = 15000
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 30000
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch} not supported")

    model_args = ModelArgs(t3k_mesh_device)
    state_dict = model_args.load_state_dict()
    partial_state_dict = {k[9:]: v for k, v in state_dict.items() if (k.startswith("layers.0."))}
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    rope_setup = RotarySetup(
        t3k_mesh_device,
        batch,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )

    transformation_mats = rope_setup.get_both_trans_mats()

    model_args.is_mixture_of_experts = True

    tt_model = TtTransformerBlock(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layer_num=0,
        dtype=dtype,
        transformation_mats=transformation_mats,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        tt_model.mesh_device,
    )

    cos, sin = precompute_freqs(
        model_args.head_dim,
        model_args.max_seq_len * 2,
        model_args.rope_theta,
        model_args.rope_scaling_factor,
        model_args.orig_context_len,
    )
    freqs_cis = torch.complex(cos, sin)

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=t3k_mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            t3k_mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    generation_length = 10
    all_tests_pass = True

    seqlen = 1

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # pt_decode_input_bsh = (torch.rand(1,1,32,512) * 2) - 1
        # start_pos = generation_start_pos + i
        # start_pos_ids = [start_pos for _ in range(batch)]

        # if mode == "decode":
        #     shard_grid = ttnn.CoreRangeSet(
        #         {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 1))}
        #     )

        #     # Define shard shape
        #     shard_shape = [32, 32]

        #     # Create the shard specification
        #     shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)

        #     # Create the width-sharded memory config
        #     width_sharded_mem_config = ttnn.MemoryConfig(
        #         ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        #         ttnn.BufferType.L1,
        #         shard_spec
        #     )

        #     pt_decode_input = ttnn.as_tensor(
        #         pt_decode_input_bsh,
        #         dtype=ttnn.bfloat16,  # or your desired dtype
        #         layout=ttnn.TILE_LAYOUT,
        #         device=t3k_mesh_device,
        #         memory_config=width_sharded_mem_config,
        #     )

        # decode_input_b1sh = pt_decode_input

        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()

        decode_input = model_args.prepare_residual_tensor_decode(
            tt_decode_input,
            # ttnn.DRAM_MEMORY_CONFIG,
            model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
        )

        # Get cos/sin matrices for the current position of each user
        rot_mats = rope_setup.get_rot_mats(current_pos)

        tt_out = tt_model(
            decode_input,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode=mode,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(t3k_mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        tt_output_torch = tt_out[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(-1, 1, model_args.dim)
        # In this test all users have the same position
        freqs_cis_i = freqs_cis[current_pos[0], :].unsqueeze(0)

        # Reference model
        ref_output = reference_model(pt_decode_input, current_pos[0], freqs_cis_i, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Decoder Block Passed!")
        else:
            logger.warning("Decoder Block Failed!")
            all_tests_pass = False

        # Increment position
        current_pos = torch.tensor([generation_start_pos + i for _ in range(batch)])
        current_pos_tensor = ttnn.from_torch(
            current_pos,
            device=t3k_mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                t3k_mesh_device,
                dims=(None, 0) if (model_args.is_galaxy and batch > 1) else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
        )

    if all_tests_pass:
        logger.info(f"All {generation_length} decode iterations Passed!")
    else:
        logger.warning("One or more iterations of decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"

        # # TT Model
        # tt_out_b1sh = tt_model(
        #     decode_input_b1sh,
        #     start_pos_ids,
        #     current_rot_mat,
        #     mode=mode
        # )

        # tt_output_torch_b1h = (
        #     ttnn.to_torch(tt_out_b1sh, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
        #     .squeeze(1)
        #     .view(32, 1, seqlen, model_args.dim)
        # )[:batch, ...]

    #     positions = torch.LongTensor([start_pos])
    #     freqs_cis_i = precompute_freqs_cis(model_args.head_dim, 128_000)[positions]
    #     ref_output_bsh = reference_model(pt_decode_input_bsh, freqs_cis_i, positions, mask=None)

    #     passing, pcc_message = comp_pcc(ref_output_bsh, tt_output_torch_b1h, pcc)

    #     logger.info(comp_allclose(ref_output_bsh, tt_output_torch_b1h))
    #     logger.info(pcc_message)

    #     if passing:
    #         logger.info("Mistral Decoder Block Passed!")
    #     else:
    #         logger.warning("Mistral Decoder Block Failed!")
    #         all_tests_pass = False

    #     current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

    # if all_tests_pass:
    #     logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    # else:
    #     logger.warning("One or more iterations of Mistral decode Failed!")
    #     assert all_tests_pass, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
