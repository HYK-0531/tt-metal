#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Working TTNN speculative decoding demo with proper device setup.
This script demonstrates speculative decoding using real TTNN models.
"""

import argparse
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import statistics

import torch
from loguru import logger
import ttnn

from models.speculative_decoding import SpeculativeConfig, create_speculative_decoder
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tt.model import Transformer
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.tokenizer import Tokenizer


class TTNNPerformanceMetrics:
    """Class to track and calculate performance metrics for TTNN models"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_tokens = 0
        self.total_time = 0.0
        self.iterations = 0
        self.acceptance_rates = []
        self.tokens_per_iteration = []
        self.iteration_times = []

    def add_iteration(self, tokens_generated: int, time_taken: float, acceptance_rate: float = None):
        self.total_tokens += tokens_generated
        self.total_time += time_taken
        self.iterations += 1
        self.tokens_per_iteration.append(tokens_generated)
        self.iteration_times.append(time_taken)
        if acceptance_rate is not None:
            self.acceptance_rates.append(acceptance_rate)

    def get_summary(self) -> Dict:
        if self.iterations == 0:
            return {}

        avg_tokens_per_sec = self.total_tokens / self.total_time if self.total_time > 0 else 0
        avg_acceptance_rate = statistics.mean(self.acceptance_rates) if self.acceptance_rates else 0

        return {
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "iterations": self.iterations,
            "avg_tokens_per_sec": avg_tokens_per_sec,
            "avg_acceptance_rate": avg_acceptance_rate,
            "avg_tokens_per_iteration": statistics.mean(self.tokens_per_iteration),
            "avg_time_per_iteration": statistics.mean(self.iteration_times),
            "median_tokens_per_sec": statistics.median(
                [t / time for t, time in zip(self.tokens_per_iteration, self.iteration_times) if time > 0]
            ),
        }


def setup_ttnn_mesh_device(mesh_shape: Tuple[int, int] = (1, 1)) -> ttnn.MeshDevice:
    """
    Set up TTNN mesh device with proper configuration.

    Args:
        mesh_shape: Shape of the mesh (rows, cols)

    Returns:
        Configured TTNN mesh device
    """
    logger.info(f"Setting up TTNN mesh device with shape {mesh_shape}")

    # Create mesh shape
    mesh_shape_obj = ttnn.MeshShape(mesh_shape[0], mesh_shape[1])

    # Open mesh device with default configuration
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape_obj,
        l1_small_size=ttnn._ttnn.device.DEFAULT_L1_SMALL_SIZE,
        trace_region_size=ttnn._ttnn.device.DEFAULT_TRACE_REGION_SIZE,
        num_command_queues=1,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
        offset=None,
        physical_device_ids=[],
        worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
    )

    logger.info(f"TTNN mesh device created: {mesh_device}")
    return mesh_device


def setup_ttnn_models(mesh_device: ttnn.MeshDevice, use_3b_draft: bool = True, use_8b_target: bool = True):
    """
    Set up TTNN models for speculative decoding.

    Args:
        mesh_device: TTNN mesh device
        use_3b_draft: Whether to use 3B model as draft
        use_8b_target: Whether to use 8B model as target

    Returns:
        Tuple of (draft_model, target_model, tokenizer)
    """
    logger.info("Setting up TTNN models...")

    # Setup draft model (Llama 3.2 3B)
    if use_3b_draft:
        draft_model_args = ModelArgs(
            mesh_device,
            instruct=True,
            dummy_weights=False,
            max_batch_size=1,
            max_seq_len=4096,
        )
        # Override to use 3B model
        draft_model_args.model_name = "Llama-3.2-3B-Instruct"
        draft_model_args.CKPT_DIR = "models/tt_transformers/model_params/Llama-3.2-3B-Instruct"
        draft_model_args.n_layers = 28  # 3B model has 28 layers

        logger.info(f"Loading draft model: {draft_model_args.model_name}")
        try:
            draft_state_dict = draft_model_args.load_state_dict()

            draft_model = Transformer(
                args=draft_model_args,
                dtype=ttnn.bfloat16,
                mesh_device=mesh_device,
                state_dict=draft_state_dict,
                weight_cache_path=draft_model_args.weight_cache_path(ttnn.bfloat16),
            )
            logger.info("Draft model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load draft model: {e}")
            draft_model = None
    else:
        draft_model = None

    # Setup target model (Llama 3.1 8B)
    if use_8b_target:
        target_model_args = ModelArgs(
            mesh_device,
            instruct=True,
            dummy_weights=False,
            max_batch_size=1,
            max_seq_len=4096,
        )
        # Override to use 8B model
        target_model_args.model_name = "Llama-3.1-8B-Instruct"
        target_model_args.CKPT_DIR = "models/tt_transformers/model_params/Llama-3.1-8B-Instruct"
        target_model_args.n_layers = 32  # 8B model has 32 layers

        logger.info(f"Loading target model: {target_model_args.model_name}")
        try:
            target_state_dict = target_model_args.load_state_dict()

            target_model = Transformer(
                args=target_model_args,
                dtype=ttnn.bfloat16,
                mesh_device=mesh_device,
                state_dict=target_state_dict,
                weight_cache_path=target_model_args.weight_cache_path(ttnn.bfloat16),
            )
            logger.info("Target model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load target model: {e}")
            target_model = None
    else:
        target_model = None

    # Setup tokenizer
    tokenizer_path = target_model_args.tokenizer_path if use_8b_target else draft_model_args.tokenizer_path
    try:
        tokenizer = Tokenizer(tokenizer_path)
        logger.info("Tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        tokenizer = None

    logger.info("TTNN models setup complete")
    return draft_model, target_model, tokenizer


def run_ttnn_regular_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> Tuple[str, TTNNPerformanceMetrics]:
    """
    Run regular generation with TTNN target model only.

    Args:
        model: Target TTNN model
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Tuple of (generated_text, performance_metrics)
    """
    logger.info("Running TTNN regular generation...")

    metrics = TTNNPerformanceMetrics()

    # Tokenize input
    input_ids = tokenizer.encode(prompt, bos=True, eos=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Convert to TTNN tensor
    input_ttnn = ttnn.from_torch(input_tensor, device=model.mesh_device)

    generated_tokens = []
    current_input = input_ttnn

    start_time = time.time()

    for i in range(max_new_tokens):
        iter_start = time.time()

        # Forward pass through TTNN model
        with torch.no_grad():
            outputs = model(current_input)

        # Get logits for the last token
        if hasattr(outputs, "logits"):
            logits = ttnn.to_torch(outputs.logits)[:, -1, :]
        else:
            logits = ttnn.to_torch(outputs)[:, -1, :]

        # Apply temperature and top-p sampling
        if temperature > 0:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float("inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        generated_tokens.append(next_token.item())

        # Update input for next iteration
        next_token_ttnn = ttnn.from_torch(next_token, device=model.mesh_device)
        current_input = ttnn.concat([current_input, next_token_ttnn], dim=1)

        iter_time = time.time() - iter_start
        metrics.add_iteration(1, iter_time)

        # Stop if EOS token
        if next_token.item() == tokenizer.eos_id:
            break

    total_time = time.time() - start_time

    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)

    logger.info(f"TTNN regular generation completed in {total_time:.2f}s")
    logger.info(f"Generated {len(generated_tokens)} tokens")
    logger.info(f"Throughput: {len(generated_tokens) / total_time:.2f} tokens/sec")

    return generated_text, metrics


def run_ttnn_speculative_generation(
    draft_model,
    target_model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    max_draft_tokens: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> Tuple[str, TTNNPerformanceMetrics]:
    """
    Run speculative decoding generation with TTNN models.

    Args:
        draft_model: Draft TTNN model (smaller)
        target_model: Target TTNN model (larger)
        tokenizer: Tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        max_draft_tokens: Maximum draft tokens per iteration
        temperature: Sampling temperature
        top_p: Top-p sampling parameter

    Returns:
        Tuple of (generated_text, performance_metrics)
    """
    logger.info("Running TTNN speculative generation...")

    metrics = TTNNPerformanceMetrics()

    # Create speculative decoding configuration
    config = SpeculativeConfig(
        draft_model_name="llama3.2-3b",
        target_model_name="llama3.1-8b",
        max_draft_tokens=max_draft_tokens,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_id,
        pad_token_id=tokenizer.pad_id,
    )

    # Create speculative decoder with TTNN models
    decoder = create_speculative_decoder(
        draft_model, target_model, tokenizer, config, mesh_device=draft_model.mesh_device
    )

    start_time = time.time()

    # Generate text using speculative decoding
    generated_text, acceptance_stats = decoder.generate_text(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )

    total_time = time.time() - start_time

    # Calculate metrics from acceptance stats
    total_accepted = sum(acceptance_stats.get("accepted_counts", []))
    total_proposed = sum(acceptance_stats.get("proposed_counts", []))
    acceptance_rate = total_accepted / total_proposed if total_proposed > 0 else 0

    metrics.add_iteration(
        tokens_generated=len(generated_text.split()),
        time_taken=total_time,
        acceptance_rate=acceptance_rate,
    )

    logger.info(f"TTNN speculative generation completed in {total_time:.2f}s")
    logger.info(f"Acceptance rate: {acceptance_rate:.2%}")
    logger.info(f"Tokens accepted: {total_accepted}")
    logger.info(f"Tokens proposed: {total_proposed}")

    return generated_text, metrics


def run_ttnn_performance_comparison(
    draft_model,
    target_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    max_draft_tokens: int = 1,
    temperature: float = 0.6,
    top_p: float = 0.9,
    num_runs: int = 1,
) -> Dict:
    """
    Run comprehensive performance comparison with TTNN models.

    Args:
        draft_model: Draft TTNN model
        target_model: Target TTNN model
        tokenizer: Tokenizer
        prompts: List of prompts to test
        max_new_tokens: Maximum tokens to generate
        max_draft_tokens: Maximum draft tokens per iteration
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        num_runs: Number of runs per prompt for averaging

    Returns:
        Dictionary with comparison results
    """
    logger.info("Starting TTNN performance comparison...")

    results = {
        "config": {
            "max_new_tokens": max_new_tokens,
            "max_draft_tokens": max_draft_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_runs": num_runs,
            "num_prompts": len(prompts),
        },
        "regular_generation": {
            "runs": [],
            "avg_metrics": {},
        },
        "speculative_generation": {
            "runs": [],
            "avg_metrics": {},
        },
        "comparison": {},
    }

    regular_metrics_all = []
    speculative_metrics_all = []

    for prompt_idx, prompt in enumerate(prompts):
        logger.info(f"\n--- Testing prompt {prompt_idx + 1}/{len(prompts)} ---")
        logger.info(f"Prompt: {prompt[:100]}...")

        # Run regular generation multiple times
        for run in range(num_runs):
            logger.info(f"TTNN regular generation run {run + 1}/{num_runs}")
            generated_text, metrics = run_ttnn_regular_generation(
                target_model, tokenizer, prompt, max_new_tokens, temperature, top_p
            )

            run_result = {
                "prompt_idx": prompt_idx,
                "run": run,
                "generated_text": generated_text,
                "metrics": metrics.get_summary(),
            }
            results["regular_generation"]["runs"].append(run_result)
            regular_metrics_all.append(metrics)

        # Run speculative generation multiple times
        for run in range(num_runs):
            logger.info(f"TTNN speculative generation run {run + 1}/{num_runs}")
            generated_text, metrics = run_ttnn_speculative_generation(
                draft_model, target_model, tokenizer, prompt, max_new_tokens, max_draft_tokens, temperature, top_p
            )

            run_result = {
                "prompt_idx": prompt_idx,
                "run": run,
                "generated_text": generated_text,
                "metrics": metrics.get_summary(),
            }
            results["speculative_generation"]["runs"].append(run_result)
            speculative_metrics_all.append(metrics)

    # Calculate average metrics
    if regular_metrics_all:
        avg_regular_metrics = TTNNPerformanceMetrics()
        for metrics in regular_metrics_all:
            avg_regular_metrics.total_tokens += metrics.total_tokens
            avg_regular_metrics.total_time += metrics.total_time
            avg_regular_metrics.iterations += metrics.iterations
            avg_regular_metrics.acceptance_rates.extend(metrics.acceptance_rates)
            avg_regular_metrics.tokens_per_iteration.extend(metrics.tokens_per_iteration)
            avg_regular_metrics.iteration_times.extend(metrics.iteration_times)

        results["regular_generation"]["avg_metrics"] = avg_regular_metrics.get_summary()

    if speculative_metrics_all:
        avg_speculative_metrics = TTNNPerformanceMetrics()
        for metrics in speculative_metrics_all:
            avg_speculative_metrics.total_tokens += metrics.total_tokens
            avg_speculative_metrics.total_time += metrics.total_time
            avg_speculative_metrics.iterations += metrics.iterations
            avg_speculative_metrics.acceptance_rates.extend(metrics.acceptance_rates)
            avg_speculative_metrics.tokens_per_iteration.extend(metrics.tokens_per_iteration)
            avg_speculative_metrics.iteration_times.extend(metrics.iteration_times)

        results["speculative_generation"]["avg_metrics"] = avg_speculative_metrics.get_summary()

    # Calculate comparison metrics
    if regular_metrics_all and speculative_metrics_all:
        regular_tps = results["regular_generation"]["avg_metrics"]["avg_tokens_per_sec"]
        speculative_tps = results["speculative_generation"]["avg_metrics"]["avg_tokens_per_sec"]

        if regular_tps > 0:
            speedup = speculative_tps / regular_tps
        else:
            speedup = 0

        results["comparison"] = {
            "speedup": speedup,
            "regular_tokens_per_sec": regular_tps,
            "speculative_tokens_per_sec": speculative_tps,
            "avg_acceptance_rate": results["speculative_generation"]["avg_metrics"]["avg_acceptance_rate"],
        }

    return results


def print_ttnn_results(results: Dict):
    """Print TTNN performance comparison results."""
    print("\n" + "=" * 80)
    print("TTNN SPECULATIVE DECODING PERFORMANCE RESULTS")
    print("=" * 80)

    config = results["config"]
    print(f"Configuration:")
    print(f"  Max new tokens: {config['max_new_tokens']}")
    print(f"  Max draft tokens: {config['max_draft_tokens']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Top-p: {config['top_p']}")
    print(f"  Number of runs: {config['num_runs']}")
    print(f"  Number of prompts: {config['num_prompts']}")

    if "regular_generation" in results and "avg_metrics" in results["regular_generation"]:
        regular_metrics = results["regular_generation"]["avg_metrics"]
        print(f"\nRegular Generation (Target Model Only):")
        print(f"  Average tokens per second: {regular_metrics['avg_tokens_per_sec']:.2f}")
        print(f"  Total tokens generated: {regular_metrics['total_tokens']}")
        print(f"  Total time: {regular_metrics['total_time']:.2f}s")
        print(f"  Average time per iteration: {regular_metrics['avg_time_per_iteration']:.4f}s")

    if "speculative_generation" in results and "avg_metrics" in results["speculative_generation"]:
        speculative_metrics = results["speculative_generation"]["avg_metrics"]
        print(f"\nSpeculative Generation:")
        print(f"  Average tokens per second: {speculative_metrics['avg_tokens_per_sec']:.2f}")
        print(f"  Total tokens generated: {speculative_metrics['total_tokens']}")
        print(f"  Total time: {speculative_metrics['total_time']:.2f}s")
        print(f"  Average time per iteration: {speculative_metrics['avg_time_per_iteration']:.4f}s")
        print(f"  Average acceptance rate: {speculative_metrics['avg_acceptance_rate']:.2%}")

    if "comparison" in results:
        comparison = results["comparison"]
        print(f"\nPerformance Comparison:")
        print(f"  Speedup: {comparison['speedup']:.2f}x")
        print(f"  Regular throughput: {comparison['regular_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Speculative throughput: {comparison['speculative_tokens_per_sec']:.2f} tokens/sec")
        print(f"  Average acceptance rate: {comparison['avg_acceptance_rate']:.2%}")

        if comparison["speedup"] > 1:
            print(f"✅ Speculative decoding is {comparison['speedup']:.2f}x faster!")
        else:
            print(f"❌ Speculative decoding is {1/comparison['speedup']:.2f}x slower")

    print("=" * 80)


def main():
    """Main function to run the TTNN speculative decoding demo."""

    parser = argparse.ArgumentParser(description="TTNN Speculative Decoding Demo")
    parser.add_argument("--mesh_shape", type=int, nargs=2, default=[1, 1], help="Mesh shape (rows cols)")
    parser.add_argument("--max_new_tokens", type=int, default=30, help="Maximum tokens to generate")
    parser.add_argument("--max_draft_tokens", type=int, default=1, help="Maximum draft tokens per iteration")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of runs per prompt")
    parser.add_argument("--use_3b_draft", action="store_true", default=True, help="Use 3B model as draft")
    parser.add_argument("--use_8b_target", action="store_true", default=True, help="Use 8B model as target")

    args = parser.parse_args()

    # Test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time, there was a magical kingdom where",
        "The best way to learn programming is to",
        "In the year 2050, humanity will have achieved",
        "The most important invention of the 21st century is",
    ]

    print("Starting TTNN speculative decoding demo...")
    print("This demo uses real TTNN models with proper device setup.\n")

    try:
        # Setup TTNN mesh device
        mesh_device = setup_ttnn_mesh_device(tuple(args.mesh_shape))

        # Setup TTNN models
        draft_model, target_model, tokenizer = setup_ttnn_models(mesh_device, args.use_3b_draft, args.use_8b_target)

        if draft_model is None or target_model is None or tokenizer is None:
            logger.error("Failed to load required models or tokenizer")
            return

        # Run performance comparison
        results = run_ttnn_performance_comparison(
            draft_model,
            target_model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            args.max_draft_tokens,
            args.temperature,
            args.top_p,
            args.num_runs,
        )

        # Print results
        print_ttnn_results(results)

        # Clean up
        ttnn.close_mesh_device(mesh_device)

    except Exception as e:
        logger.error(f"Error in TTNN speculative decoding demo: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
