from typing import Any, Dict, Optional

import ttnn
import torch
import torch.nn.functional as F
from genmo.lib.progress import get_new_progress_bar
from genmo.lib.utils import Timer
from genmo.mochi_preview.vae.vae_stats import dit_latents_to_vae_latents
import numpy as np
import random

from genmo.mochi_preview.pipelines import (
    MochiSingleGPUPipeline,
    move_to_device,
    t5_tokenizer,
    get_conditioning,
    compute_packed_indices,
)
from genmo.mochi_preview.vae.models import normalize_decoded_frames


def sample_model(device, dit, conditioning, **args):
    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])

    generator = torch.Generator(device=device)
    generator.manual_seed(args["seed"])

    w, h, t = args["width"], args["height"], args["num_frames"]
    sample_steps = args["num_inference_steps"]
    cfg_schedule = args["cfg_schedule"]
    sigma_schedule = args["sigma_schedule"]

    assert len(cfg_schedule) == sample_steps, "cfg_schedule must have length sample_steps"
    assert (t - 1) % 6 == 0, "t - 1 must be divisible by 6"
    assert len(sigma_schedule) == sample_steps + 1, "sigma_schedule must have length sample_steps + 1"

    B = 1
    SPATIAL_DOWNSAMPLE = 8
    TEMPORAL_DOWNSAMPLE = 6
    IN_CHANNELS = 12
    PATCH_SIZE = 2
    latent_t = ((t - 1) // TEMPORAL_DOWNSAMPLE) + 1
    latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE
    num_visual_tokens = latent_t * latent_h * latent_w // (PATCH_SIZE**2)
    num_latents = latent_t * latent_h * latent_w

    z_BCTHW = torch.randn(
        (B, IN_CHANNELS, latent_t, latent_h, latent_w),
        device=device,
        dtype=torch.float32,
    )

    cond_text = cond_null = None
    if "cond" in conditioning:
        cond_text = conditioning["cond"]
        cond_null = conditioning["null"]
        cond_text["packed_indices"] = compute_packed_indices(device, cond_text["y_mask"][0], num_latents)
    else:
        assert False, "Batched mode not supported"

    def model_fn(*, z_1BNI, sigma_B, cfg_scale):
        cond_z_1BNI = dit.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma_B,
            y_feat_1BLY=cond_y_feat_1BLY,
            y_pool_11BX=cond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            N=N,
            uncond=False,
        )

        uncond_z_1BNI = dit.forward_inner(
            x_1BNI=z_1BNI,
            sigma=sigma_B,
            y_feat_1BLY=uncond_y_feat_1BLY,
            y_pool_11BX=uncond_y_pool_11BX,
            rope_cos_1HND=rope_cos_1HND,
            rope_sin_1HND=rope_sin_1HND,
            trans_mat=trans_mat,
            N=N,
            uncond=True,
        )

        assert cond_z_1BNI.shape == uncond_z_1BNI.shape
        torch_cond = dit.reverse_preprocess(cond_z_1BNI, latent_t, latent_h, latent_w, N)
        torch_uncond = dit.reverse_preprocess(uncond_z_1BNI, latent_t, latent_h, latent_w, N)
        ttnn.deallocate(cond_z_1BNI)
        ttnn.deallocate(uncond_z_1BNI)
        ttnn.deallocate(z_1BNI)
        torch_pred = torch_uncond + cfg_scale * (torch_cond - torch_uncond)
        return torch_pred

    # Preparation before first iteration
    rope_cos_1HND, rope_sin_1HND, trans_mat = dit.prepare_rope_features(T=latent_t, H=latent_h, W=latent_w)
    # Note that conditioning contains list of len 1 to index into
    num_text_tokens = cond_text["packed_indices"]["max_seqlen_in_batch_kv"] - num_visual_tokens
    cond_text["y_feat"][0] = cond_text["y_feat"][0][:, :num_text_tokens, :]
    cond_text["y_mask"][0] = cond_text["y_mask"][0][:, :num_text_tokens]
    cond_y_feat_1BLY, cond_y_pool_11BX = dit.prepare_text_features(
        t5_feat=cond_text["y_feat"][0], t5_mask=cond_text["y_mask"][0]
    )
    uncond_y_feat_1BLY, uncond_y_pool_11BX = dit.prepare_text_features(
        t5_feat=cond_null["y_feat"][0], t5_mask=cond_null["y_mask"][0]
    )

    for i in get_new_progress_bar(range(0, sample_steps), desc="Sampling"):
        sigma = sigma_schedule[i]
        dsigma = sigma - sigma_schedule[i + 1]

        z_1BNI, N = dit.preprocess_input(z_BCTHW)
        sigma_B = torch.full([B], sigma, device=device)
        pred_BCTHW = model_fn(z_1BNI=z_1BNI, sigma_B=sigma_B, cfg_scale=cfg_schedule[i])
        # assert pred_BCTHW.dtype == torch.float32
        z_BCTHW = z_BCTHW + dsigma * pred_BCTHW

    # Postprocess z
    # z_BCTHW = dit.reverse_preprocess(z_1BNI, latent_t, latent_h, latent_w, N).float()
    ttnn.deallocate(rope_cos_1HND)
    ttnn.deallocate(rope_sin_1HND)
    return dit_latents_to_vae_latents(z_BCTHW)


def decode_latents(decoder, z):
    assert z.ndim == 5
    z = decoder.prepare_input(z)
    samples = decoder(z)
    samples = decoder.postprocess_output(samples)
    return normalize_decoded_frames(samples)


class TTPipeline(MochiSingleGPUPipeline):
    """TensorTorch-specific version of MochiSingleGPUPipeline."""

    def __call__(self, batch_cfg, prompt, negative_prompt, **kwargs):
        assert self.decode_type not in ["tiled_spatial", "tiled_full"]
        with torch.inference_mode():
            print("get_conditioning")
            with move_to_device(self.text_encoder, self.device):
                conditioning = get_conditioning(
                    tokenizer=self.tokenizer,
                    encoder=self.text_encoder,
                    device=self.device,
                    batch_inputs=batch_cfg,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )

            print("get dit")
            dit = self.dit_factory.get_model(local_rank=0, device_id=0, world_size=1)
            print("sample_model")
            latents = sample_model(self.device, dit, conditioning, **kwargs)
            print("deallocate dit")
            dit.dealloc()

            print("get decoder")
            decoder = self.decoder_factory.get_model(local_rank=0, device_id=0, world_size=1)
            if self.decode_type == "tiled_full":
                # frames = decode_latents_tiled_full(self.decoder, latents, **self.decode_args)
                pass
            elif self.decode_type == "tiled_spatial":
                # frames = decode_latents_tiled_spatial(
                #     self.decoder, latents, **self.decode_args, num_tiles_w=4, num_tiles_h=2
                # )
                pass
            else:
                frames = decode_latents(decoder, latents)

            print("deallocate decoder")
            decoder.dealloc()

            return frames.cpu().numpy()
