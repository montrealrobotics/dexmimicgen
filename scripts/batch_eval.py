#!/usr/bin/env python3
"""
Batch evaluation script for reproducible testing of multiple checkpoints.
Runs all evaluations in a single process to ensure identical environmental conditions.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add paths
sys.path.append("/home/artur/octo")
sys.path.append("/home/artur/dexmimicgen")

import numpy as np
import random
import jax

def load_config(config_path):
    """Load evaluation configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_batch_evaluation(config_path, verbose=True):
    """Run batch evaluation according to configuration."""

    # Load configuration
    config = load_config(config_path)
    global_seed = config.get('seed', 42)

    if verbose:
        print(f"=== Batch Evaluation with Seed {global_seed} ===")
        print(f"Configuration: {config_path}")
        print(f"Number of evaluations: {len(config['evaluations'])}")
        print()

    # Set comprehensive seeds for reproducibility
    random.seed(global_seed)
    np.random.seed(global_seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    jax_key = jax.random.PRNGKey(global_seed)

    if verbose:
        print(f"Global seeds set (Python, NumPy, JAX, CUDA deterministic)")

    # Results storage
    all_results = {}

    # Run each evaluation
    for eval_idx, eval_config in enumerate(config['evaluations']):
        eval_name = eval_config['name']
        checkpoint_path = eval_config['checkpoint']
        env_name = eval_config['env']
        num_episodes = eval_config['episodes']
        max_steps = eval_config.get('max_steps', 400)
        pca_enabled = eval_config.get('pca_enabled', False)

        # Create evaluation-specific seed for reproducibility
        eval_seed = global_seed + eval_idx

        if verbose:
            print(f"\n--- Evaluation: {eval_name} ---")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Environment: {env_name}")
            print(f"Episodes: {num_episodes}, Max Steps: {max_steps}")
            print(f"PCA Enabled: {pca_enabled}")
            print(f"Evaluation seed: {eval_seed}")

        try:
            # Import here to ensure proper initialization
            from eval_octo_checkpoint_cli import evaluate_octo_checkpoint

            # Run evaluation (evaluate_octo_checkpoint handles all seeding internally)
            results = evaluate_octo_checkpoint(
                checkpoint_path=checkpoint_path,
                env_name=env_name,
                num_episodes=num_episodes,
                max_steps=max_steps,
                render=False,
                save_video=False,
                verbose=verbose,
                seed=eval_seed,
                pca_enabled=pca_enabled
            )

            all_results[eval_name] = results

            if verbose:
                print(f"✓ Completed: Mean Return = {results['mean_return']:.3f} ± {results['std_return']:.3f}")
                print(f"  Success Rate: {results['success_rate']:.1%}")
                print(f"  Mean Length: {results['mean_length']:.1f}")

        except Exception as e:
            print(f"✗ Failed: {e}")
            all_results[eval_name] = {"error": str(e)}

    # Print summary
    print(f"\n{'='*60}")
    print("BATCH EVALUATION SUMMARY")
    print(f"{'='*60}")

    print(f"{'Evaluation':<20} {'Mean Return':<12} {'Std Return':<12} {'Success %':<10} {'Mean Steps':<10}")
    print("-" * 70)

    for eval_name, results in all_results.items():
        if "error" in results:
            print(f"{eval_name:<20} ERROR: {results['error']}")
        else:
            mean_return = results['mean_return']
            std_return = results['std_return']
            success_rate = results['success_rate'] * 100
            mean_length = results['mean_length']
            print(f"{eval_name:<20} {mean_return:<12.3f} {std_return:<12.3f} {success_rate:<10.1f} {mean_length:<10.1f}")

    print(f"{'='*60}")

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation for reproducible testing of multiple checkpoints"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="../eval_config.yaml",
        help="Path to evaluation configuration YAML file"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Convert relative path to absolute if needed
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    results = run_batch_evaluation(str(config_path), verbose=not args.quiet)

    # Return success/failure based on whether any evaluations failed
    failed_evals = [name for name, result in results.items() if "error" in result]
    if failed_evals:
        print(f"Failed evaluations: {failed_evals}")
        sys.exit(1)
    else:
        print("All evaluations completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
