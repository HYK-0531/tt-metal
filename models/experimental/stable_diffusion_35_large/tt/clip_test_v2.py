import torch
import ttnn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

from clip_encoder import TtCLIPTextTransformer, TtCLIPTextTransformerParameters, TtCLIPConfig


class CLIPEmbeddingTester:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.device = ttnn.open_device(device_id=0)
        self.hf_device = torch.device("cpu")

        self.hf_model = CLIPTextModel.from_pretrained(model_name).to(self.hf_device)
        self.hf_model = self.hf_model.to(torch.bfloat16)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.hf_model.eval()

        self.config = self._create_ttnn_config()
        self.ttnn_params = self._load_ttnn_parameters()
        self.ttnn_model = TtCLIPTextTransformer(self.ttnn_params, self.config)

    def _create_ttnn_config(self) -> TtCLIPConfig:
        hf_config = self.hf_model.config
        return TtCLIPConfig(
            vocab_size=hf_config.vocab_size,
            d_model=hf_config.hidden_size,
            d_ff=hf_config.intermediate_size,
            num_heads=hf_config.num_attention_heads,
            num_layers=hf_config.num_hidden_layers,
            max_position_embeddings=77,
            layer_norm_eps=hf_config.layer_norm_eps,
            attention_dropout=hf_config.attention_dropout,
        )

    def _load_ttnn_parameters(self) -> TtCLIPTextTransformerParameters:
        state_dict = self.hf_model.state_dict()
        return TtCLIPTextTransformerParameters.from_torch(state_dict, dtype=ttnn.bfloat16, device=self.device)

    def prepare_inputs(self, texts: list[str]):
        hf_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(
            self.hf_device
        )

        ttnn_input_ids = ttnn.from_torch(
            torch.from_numpy(hf_inputs.input_ids.cpu().numpy()),
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        return hf_inputs, ttnn_input_ids

    def get_embeddings(self, hf_inputs, ttnn_inputs):
        # hf embeddings
        with torch.no_grad():
            hf_outputs = self.hf_model(**hf_inputs)
            hf_seq = hf_outputs.last_hidden_state.cpu()
            hf_pooled = hf_outputs.pooler_output.cpu()

        # ttnn embeddings
        ttnn_seq, ttnn_pooled = self.ttnn_model(ttnn_inputs, self.device)
        ttnn_seq = ttnn.to_torch(ttnn_seq).cpu()
        ttnn_pooled = ttnn.to_torch(ttnn_pooled).cpu()

        return hf_seq, hf_pooled, ttnn_seq, ttnn_pooled

    def check_shapes(self, hf_seq, hf_pooled, ttnn_seq, ttnn_pooled):
        print(
            f"Sequence shapes: HF {hf_seq.shape}, TTNN {ttnn_seq.shape} - {'PASS' if hf_seq.shape == ttnn_seq.shape else 'FAIL'}"
        )
        print(
            f"Pooled shapes: HF {hf_pooled.shape}, TTNN {ttnn_pooled.shape} - {'PASS' if hf_pooled.shape == ttnn_pooled.shape else 'FAIL'}"
        )

        return hf_seq.shape == ttnn_seq.shape and hf_pooled.shape == ttnn_pooled.shape

    def check_cosine_similarity(self, hf_seq, hf_pooled, ttnn_seq, ttnn_pooled):
        # sequence similarity
        hf_flat = hf_seq.view(-1, hf_seq.shape[-1]).float()  # reshape and convert to float32
        ttnn_flat = ttnn_seq.view(-1, ttnn_seq.shape[-1]).float()  # reshape and convert to float32
        seq_sim = F.cosine_similarity(hf_flat, ttnn_flat, dim=1)
        seq_mean = seq_sim.mean().item()

        # pooled similarity
        pooled_sim = F.cosine_similarity(hf_pooled.float(), ttnn_pooled.float(), dim=1)
        pooled_mean = pooled_sim.mean().item()

        print(f"Sequence cosine similarity: {seq_mean:.4f}")
        print(f"Pooled cosine similarity: {pooled_mean:.4f}")

        return seq_mean > 0.99 and pooled_mean > 0.99  # what should threshold be?


def test_clip():
    tester = CLIPEmbeddingTester()

    texts = ["A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"]
    hf_inputs, ttnn_inputs = tester.prepare_inputs(texts)

    hf_seq, hf_pooled, ttnn_seq, ttnn_pooled = tester.get_embeddings(hf_inputs, ttnn_inputs)

    shape_pass = tester.check_shapes(hf_seq, hf_pooled, ttnn_seq, ttnn_pooled)
    similarity_pass = tester.check_cosine_similarity(hf_seq, hf_pooled, ttnn_seq, ttnn_pooled)


if __name__ == "__main__":
    test_clip()
