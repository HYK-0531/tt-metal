import torch
import ttnn
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from t5_encoder import TtT5Encoder, TtT5EncoderParameters, TtT5Config


class T5EmbeddingTester:
    def __init__(self, model_name: str = "t5-small"):
        self.device = ttnn.open_device(device_id=0)
        self.hf_device = torch.device("cpu")

        self.hf_model = T5EncoderModel.from_pretrained(model_name).to(self.hf_device)
        self.hf_model = self.hf_model.to(torch.bfloat16)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.hf_model.eval()

        self.config = self._create_ttnn_config()
        self.ttnn_params = self._load_ttnn_parameters()
        self.ttnn_model = TtT5Encoder(
            self.ttnn_params,
            num_heads=self.config.num_heads,
            relative_attention_num_buckets=self.config.relative_attention_num_buckets,
            relative_attention_max_distance=self.config.relative_attention_max_distance,
            layer_norm_epsilon=self.config.layer_norm_epsilon,
        )

    def _create_ttnn_config(self) -> TtT5Config:
        hf_config = self.hf_model.config
        return TtT5Config(
            vocab_size=hf_config.vocab_size,
            d_model=hf_config.d_model,
            d_ff=hf_config.d_ff,
            d_kv=hf_config.d_kv,
            num_heads=hf_config.num_heads,
            num_layers=hf_config.num_layers,
            relative_attention_num_buckets=hf_config.relative_attention_num_buckets,
            relative_attention_max_distance=hf_config.relative_attention_max_distance,
            layer_norm_epsilon=hf_config.layer_norm_epsilon,
        )

    def _load_ttnn_parameters(self) -> TtT5EncoderParameters:
        state_dict = self.hf_model.state_dict()
        return TtT5EncoderParameters.from_torch(state_dict, dtype=ttnn.bfloat16, device=self.device)

    def prepare_inputs(self, texts: list[str]):
        hf_inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
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

        # ttnn embeddings
        ttnn_seq = self.ttnn_model(ttnn_inputs, self.device)
        ttnn_seq = ttnn.to_torch(ttnn_seq).cpu()

        return hf_seq, ttnn_seq

    def check_shapes(self, hf_seq, ttnn_seq):
        print(
            f"Sequence shapes: HF {hf_seq.shape}, TTNN {ttnn_seq.shape} - {'PASS' if hf_seq.shape == ttnn_seq.shape else 'FAIL'}"
        )

        return hf_seq.shape == ttnn_seq.shape

    def check_cosine_similarity(self, hf_seq, ttnn_seq):
        # sequence similarity
        hf_flat = hf_seq.view(-1, hf_seq.shape[-1]).float()  # reshape and convert to float32
        ttnn_flat = ttnn_seq.view(-1, ttnn_seq.shape[-1]).float()  # reshape and convert to float32
        seq_sim = F.cosine_similarity(hf_flat, ttnn_flat, dim=1)
        seq_mean = seq_sim.mean().item()

        print(f"Sequence cosine similarity: {seq_mean:.4f}")

        return seq_mean > 0.8


def test_t5():
    tester = T5EmbeddingTester()

    texts = ["A coffee shop on Main Street that serves excellent pastries and opens at 7 AM on weekdays"]
    hf_inputs, ttnn_inputs = tester.prepare_inputs(texts)

    hf_seq, ttnn_seq = tester.get_embeddings(hf_inputs, ttnn_inputs)

    shape_pass = tester.check_shapes(hf_seq, ttnn_seq)
    similarity_pass = tester.check_cosine_similarity(hf_seq, ttnn_seq)

    overall_pass = shape_pass and similarity_pass
    print(f"Test: {'PASS' if overall_pass else 'FAIL'}")

    return overall_pass


if __name__ == "__main__":
    test_t5()
