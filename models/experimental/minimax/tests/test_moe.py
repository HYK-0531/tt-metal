import torch
import ttnn


from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from models.experimental.minimax.tt.minimax_moe import TTMiniMaxM1SparseMoeBlock
from models.experimental.minimax.tt.minimax_config import TTMiniMaxM1MoeConfig

from .test_utils import get_minimax_tokenizer_and_config_and_state_dict


@pytest.fixture
def setup():
    torch.manual_seed(1234)
    device = ttnn.open_device(device_id=0)
    tokenizer, config, state_dict = get_minimax_tokenizer_and_config_and_state_dict()
    return tokenizer, config, state_dict, device


def test_moe_load_weights(setup):
    _, config, state_dict, device = setup
    # Test block_sparse_moe
    test_base_address = f"model.layers.0.block_sparse_moe"
    block_sparse_moe = TTMiniMaxM1SparseMoeBlock(config, state_dict, test_base_address, device=device)


def test_moe_inference(setup):
    _, config, state_dict, device = setup
    # Test block_sparse_moe
    test_base_address = f"model.layers.0.block_sparse_moe"
    block_sparse_moe = TTMiniMaxM1SparseMoeBlock(config, state_dict, test_base_address, device=device)


def test_moe_sharded():
    torch.manual_seed(1234)
    device = ttnn.open_device(device_id=0)

    config_path = hf_hub_download(repo_id="MiniMaxAI/MiniMax-M1-80k", filename="config.json")

    # Note that the 00001 shard only contains layers > 0 ... 2 > block_sparse_moe > experts > 0 > w1 ... w3 > weight
    weights_path = hf_hub_download(repo_id="MiniMaxAI/MiniMax-M1-80k", filename="model-00001-of-00414.safetensors")

    with open(config_path, "r") as f:
        config = TTMiniMaxM1MoeConfig.from_pretrained("MiniMaxAI/MiniMax-M1-80k", trust_remote_code=True)

    state_dict_shard = load_file(weights_path, device="cpu")

    # Test block_sparse_moe
    test_base_address = f"model.layers.0.block_sparse_moe"
    block_sparse_moe = TTMiniMaxM1SparseMoeBlock(config, state_dict_shard, test_base_address, device=device)
