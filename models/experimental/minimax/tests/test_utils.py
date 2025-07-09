import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
from safetensors.torch import load_file
import os

from transformers import AutoTokenizer, AutoModelForCausalLM


# From: https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def get_minimax_tokenizer_and_config_and_state_dict():
    # Load tokenizer & model (456B parameters, MoE; loads across all available GPUs)
    tokenizer = AutoTokenizer.from_pretrained("MiniMaxAI/MiniMax-M1-80k")  # :contentReference[oaicite:0]{index=0}
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            "MiniMaxAI/MiniMax-M1-80k",
            torch_dtype=torch.bfloat16,
            device_map=None,  # automatically shards experts across GPUst
            offload_folder="offload",  # optional: offload to CPU/NVMe if you run out of GPU RAM
            offload_state_dict=True,
            trust_remote_code=True,
        )
    return model.config, model.state_dict()


def recursive_state_dict_print_from_path(weights_path):
    state_dict = load_file(weights_path)

    print("\n=== State Dict Summary ===")

    # Build a nested dictionary from the flat keys
    def build_nested_dict(keys):
        nested = {}
        for key in keys:
            parts = key.split(".")
            d = nested
            for part in parts:
                d = d.setdefault(part, {})
        return nested

    # Recursively print the nested names
    def print_nested(d, indent=0):
        for name, subtree in sorted(d.items()):
            print("  " * indent + name)
            if subtree:
                print_nested(subtree, indent + 1)

    nested = build_nested_dict(state_dict.keys())
    print_nested(nested)
