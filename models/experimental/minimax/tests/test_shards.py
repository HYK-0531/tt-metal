from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import os


# From: https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4
def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


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
