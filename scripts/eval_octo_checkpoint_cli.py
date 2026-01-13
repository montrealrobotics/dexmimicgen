"""
Command-line interface for evaluating Octo model checkpoints on dexmimicgen environments.

This script provides a simple CLI for evaluating finetuned Octo models on dexmimicgen
environments. It adapts the dexmimicgen environments to match Octo's expected gym interface.

Usage:
    python eval_octo_checkpoint_cli.py --checkpoint /path/to/octo_checkpoint --env SingleArmDrawerCleanup

Available environments:
    - SingleArmDrawerCleanup
    - TwoArmDrawerCleanup
    - TwoArmBoxCleanup
    - TwoArmThreading
    - TwoArmThreePieceAssembly
    - TwoArmTransport
    - TwoArmLiftTray
    - TwoArmCoffee
    - TwoArmPouring
    - TwoArmCanSortRandom
"""

import os
import sys
import argparse
import random

import gym
import gym.spaces
import jax
import numpy as np
from absl import logging

import dexmimicgen
import robosuite
from robosuite import load_composite_controller_config
import imageio

# Import PCA action wrapper
from dexmimicgen.utils.pca_action_wrapper import PCAActionWrapper, create_pca_from_dataset

try:
    from octo.model.octo_model import OctoModel
    from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
    OCTO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Octo not available: {e}. Please install Octo and update the path.")
    OCTO_AVAILABLE = False


class DexMimicGenGymWrapper(gym.Env):
    """
    Gym wrapper for dexmimicgen environments to match Octo's expected interface.
    """

    def __init__(self, env_name, **env_kwargs):
        super().__init__()

        # Map environment names to robots
        ENV_ROBOTS = {
            "SingleArmDrawerCleanup": ["PandaDexRH"],
            "TwoArmDrawerCleanup": ["PandaDexRH", "PandaDexLH"],
            "TwoArmBoxCleanup": ["PandaDexRH", "PandaDexLH"],
            "TwoArmThreading": ["Panda", "Panda"],
            "TwoArmThreePieceAssembly": ["Panda", "Panda"],
            "TwoArmTransport": ["Panda", "Panda"],
            "TwoArmLiftTray": ["PandaDexRH", "PandaDexLH"],
            "TwoArmCoffee": ["GR1FixedLowerBody"],
            "TwoArmPouring": ["GR1FixedLowerBody"],
            "TwoArmCanSortRandom": ["GR1ArmsOnly"],
        }

        assert env_name in ENV_ROBOTS, f"Environment {env_name} not found in ENV_ROBOTS"

        # Set up environment configuration
        self.env_name = env_name
        self.save_video = env_kwargs.pop("save_video", False)
        self.video_path = env_kwargs.pop("video_path", "./videos")
        default_env_kwargs = {
            "env_name": env_name,
            "robots": ENV_ROBOTS[env_name],
            "controller_configs": load_composite_controller_config(
                robot=ENV_ROBOTS[env_name][0]
                # controller="panda_dex_double.json"
            ),
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": False,
            "use_camera_obs": True,
            "control_freq": 20,
            "camera_names": ["agentview", "robot0_eye_in_hand"],
            "camera_heights": [256, 128],
            "camera_widths": [256, 128],
            "seed": env_kwargs.get("seed", None),
        }

        # Override defaults with provided kwargs
        default_env_kwargs.update(env_kwargs)

        # Ensure numpy seed is set before environment creation for reproducible physics
        if "seed" in default_env_kwargs:
            np.random.seed(default_env_kwargs["seed"])

        # Create the robosuite environment
        self.env = robosuite.make(**default_env_kwargs)

        # Set up video recording if enabled
        self.frames = []
        self.is_recording = False
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)

        # Set up action and observation spaces
        action_spec = self.env.action_spec
        self.action_space = gym.spaces.Box(low=action_spec[0], high=action_spec[1], dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "image_primary": gym.spaces.Box(
                low=0, high=255, shape=(256, 256, 3), dtype=np.uint8
            ),
            "image_wrist": gym.spaces.Box(
                low=0, high=255, shape=(128, 128, 3), dtype=np.uint8
            ),
            "proprio": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32),
        })

    def reset(self, rng=None, **kwargs):
        # Update rng if provided and the environment supports it
        if rng is not None and hasattr(self.env, 'set_rng'):
            self.env.set_rng(rng)
        obs = self.env.reset(**kwargs)
        octo_obs = self._convert_obs_to_octo_format(obs)
        return octo_obs, {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        octo_obs = self._convert_obs_to_octo_format(obs)
        return octo_obs, reward, done, False, info

    def _convert_obs_to_octo_format(self, obs):
        """Convert robosuite observations to Octo's expected format."""
        octo_obs = {}

        # Primary image observation (agentview)
        if "agentview_image" in obs:
            image = obs["agentview_image"][::-1]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                octo_obs["image_primary"] = image
            elif len(image.shape) == 3 and image.shape[0] == 3:
                octo_obs["image_primary"] = np.transpose(image, (1, 2, 0))

        # Wrist image observation (agentview)
        if "robot0_eye_in_hand_image" in obs:
            image = obs["robot0_eye_in_hand_image"][::-1]
            if len(image.shape) == 3 and image.shape[-1] == 3:
                octo_obs["image_wrist"] = image
            elif len(image.shape) == 3 and image.shape[0] == 3:
                octo_obs["image_wrist"] = np.transpose(image, (1, 2, 0))

        # Proprioceptive observations
        # legacy dataset
        proprio = np.asarray(np.concatenate((obs["robot0_joint_pos"][::-1], obs["robot0_gripper_qpos"][[0,2,4,6,8,11]][::-1]), axis=-1), dtype=np.float32)
        # proprio = np.asarray(np.concatenate((obs["robot0_joint_pos"], obs["robot0_gripper_qpos"][[0,2,4,6,8,11]]), axis=-1), dtype=np.float32)
        # proprio = np.asarray(np.concatenate((obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"][[0,2,4,6,8,11]]), axis=-1), dtype=np.float32)
        octo_obs["proprio"] = proprio

        return octo_obs
    
    def drawer_opened(self):
        """Check if drawer was opened so far."""
        if hasattr(self.env, 'drawer_opened'):
            return self.env.drawer_opened
        else:
            raise ValueError("env doesnt have a drawer")
            

    def get_task(self):
        """Return task specification with language instruction."""
        if hasattr(self.env, 'get_task'):
            return self.env.get_task()
        else:
            return {
                "language_instruction": f"Complete the {self.env_name} task",
                "goal": {}
            }

    def render(self):
        return self.env.render()

    def start_recording(self, filename):
        """Start collecting frames for video recording."""
        if self.save_video:
            self.frames = []
            self.is_recording = True
            self.video_filename = filename

    def capture_frame(self, obs):
        """Capture the current frame from observations if recording is active."""
        if self.save_video and self.is_recording:
            # Get the frame from the observation
            if "image_primary" in obs:
                frame = obs["image_primary"]
                if frame is not None:
                    # Remove batch dimension if present
                    if len(frame.shape) == 4 and frame.shape[0] == 1:
                        frame = frame[0]  # Remove batch dimension

                    # Ensure frame is in the correct format (H, W, C) with 3 channels
                    if len(frame.shape) == 3 and frame.shape[-1] == 3:
                        self.frames.append(frame)
                    elif len(frame.shape) == 3 and frame.shape[0] == 3:
                        # Convert from CHW to HWC
                        converted_frame = np.transpose(frame, (1, 2, 0))
                        self.frames.append(converted_frame)

    def stop_recording(self):
        """Stop recording and save the video."""
        if self.save_video and self.is_recording:
            self.is_recording = False
            if self.frames:
                video_path = os.path.join(self.video_path, self.video_filename)
                writer = imageio.get_writer(video_path, fps=20)
                for frame in self.frames:
                    writer.append_data(frame)
                writer.close()
                print(f"Video saved to {video_path}")
            self.frames = []

    def close(self):
        return self.env.close()


def evaluate_octo_checkpoint(
    checkpoint_path: str,
    env_name: str,
    num_episodes: int = 3,
    max_steps: int = 400,
    render: bool = False,
    save_video: bool = False,
    verbose: bool = True,
    seed: int = 0,
    pca_enabled: bool = False,
):
    """
    Evaluate an Octo checkpoint on a dexmimicgen environment.

    Args:
        checkpoint_path: Path to the Octo checkpoint directory
        env_name: Name of the dexmimicgen environment
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        render: Whether to render the environment
        save_video: Whether to save videos of the rollouts
        verbose: Whether to print verbose output
        seed: Random seed for reproducible evaluation
        pca_enabled: Whether to enable PCA action space reduction

    Returns:
        dict: Evaluation results
    """
    # Set comprehensive seeds for reproducible evaluation
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Set CUDA deterministic mode for reproducibility (if CUDA is available)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Forces deterministic CUDA kernel launches
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic cuBLAS operations

    # Additional determinism settings
    os.environ['MUJOCO_GL'] = 'disable'  # Disable OpenGL rendering determinism issues

    # Create a deterministic JAX key for reproducible model sampling
    jax_key = jax.random.PRNGKey(seed)


    if not OCTO_AVAILABLE:
        raise RuntimeError("Octo is not available. Please install Octo and update the path.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if verbose:
        logging.info(f"Loading Octo model from {checkpoint_path}")
    model = OctoModel.load_pretrained(checkpoint_path, step=30000)

    if verbose:
        logging.info(f"Model loaded, dataset stats keys: {list(model.dataset_statistics.keys())}")

    if verbose:
        logging.info(f"Creating {env_name} environment")
    env_kwargs = {"has_renderer": render, "seed": seed} if render else {"seed": seed}
    if save_video:
        video_dir = "./videos"
        os.makedirs(video_dir, exist_ok=True)
        env_kwargs["save_video"] = True
        env_kwargs["video_path"] = video_dir
    env = DexMimicGenGymWrapper(env_name, **env_kwargs)

    if verbose:
        logging.info(f"Environment created: {type(env)}")

    # Apply Octo wrappers
    env = NormalizeProprio(env, model.dataset_statistics)
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=4)

    if verbose:
        logging.info(f"Octo wrappers applied: NormalizeProprio, HistoryWrapper, RHCWrapper")

    # Apply PCA action wrapper only when requested
    if pca_enabled:
        if verbose:
            logging.info("Applying PCA action space reduction")

        # Create PCA model for action space reduction
        pca_model, mean_action, std_action = create_pca_from_dataset(
            "/home/artur/dexmimicgen/datasets/generated/two_arm_drawer_cleanup.hdf5",
            hand_indices=(6, 12),
            n_components=2
        )
        env = PCAActionWrapper(env, pca_model, mean_action, std_action)

    # Create deterministic policy function factory
    def create_deterministic_policy(episode_key):
        """Create a policy function with deterministic RNG from episode_key."""
        def policy_fn(observations, tasks):
            deterministic_key = jax.random.PRNGKey(episode_key.sum())  # Deterministic from episode key
            return model.sample_actions(
                observations,
                tasks,
                rng=deterministic_key,
                unnormalization_statistics=model.dataset_statistics["action"],
            )
        return policy_fn

    # Run evaluation
    episode_returns = []
    episode_lengths = []
    successes = []
    opened = []

    for episode_idx in range(num_episodes):
        if verbose:
            print(f"\nEpisode {episode_idx + 1}/{num_episodes}")

        episode_seed = seed + episode_idx  # Different seed per episode for variety, but deterministic
        np.random.seed(episode_seed)

        random.seed(episode_seed)

        # Create episode-specific rng for reproducible object placement
        episode_rng = np.random.default_rng(episode_seed)

        episode_jax_key = jax.random.fold_in(jax_key, episode_idx)

        policy_fn = create_deterministic_policy(episode_jax_key)

        if verbose and episode_idx == 0:
            logging.info(f"Episode {episode_idx} using seed {episode_seed}")

        # Start video recording for this episode if enabled
        if save_video:
            video_filename = f"{env_name}_episode_{episode_idx + 1}.mp4"
            env.start_recording(video_filename)

        np.random.seed(episode_seed)
        obs, _ = env.reset(rng=episode_rng)
        if render:
            env.render()

        # Capture initial frame for video recording
        env.capture_frame(obs)

        # Get task specification
        task_spec = env.get_task()
        language_instruction = task_spec.get("language_instruction", f"Complete the {env_name} task")

        if verbose:
            print(f"Task: {language_instruction}")

        task = model.create_tasks(texts=[language_instruction])

        # Run episode
        episode_return = 0.0
        step_count = 0
        success = False

        while step_count < max_steps:
            # Get actions from model
            obs['timestep'] = np.array([step_count])
            actions = policy_fn(jax.tree_map(lambda x: x[None], obs), task)
            actions = actions[0]

            obs, reward, done, trunc, _ = env.step(actions)
            if render:
                env.render()

            # Capture frame for video recording
            env.capture_frame(obs)

            episode_return += reward
            step_count += len(obs)

            # Check success
            if hasattr(env.env, '_check_success'):
                success = env.env._check_success()

            if done or trunc:
                break

        # Stop video recording for this episode if enabled
        if save_video:
            env.stop_recording()

        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        successes.append(success)
        opened.append(env.drawer_opened())

    
    env.close()

    # Calculate statistics
    results = {
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "successes": successes,
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "success_rate": np.mean(successes),
        "num_episodes": num_episodes,
        "environment": env_name,
        "checkpoint_path": checkpoint_path,
        'opened': opened,
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Octo model checkpoints on dexmimicgen environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available environments:
  SingleArmDrawerCleanup, TwoArmDrawerCleanup, TwoArmBoxCleanup,
  TwoArmThreading, TwoArmThreePieceAssembly, TwoArmTransport,
  TwoArmLiftTray, TwoArmCoffee, TwoArmPouring, TwoArmCanSortRandom

Example usage:
  python eval_octo_checkpoint_cli.py --checkpoint /path/to/checkpoint --env SingleArmDrawerCleanup
  python eval_octo_checkpoint_cli.py --checkpoint /path/to/checkpoint --env TwoArmDrawerCleanup --episodes 5 --render
  python eval_octo_checkpoint_cli.py --checkpoint /path/to/checkpoint --env TwoArmDrawerCleanup --seed 42
        """
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        required=True,
        help="Path to the Octo checkpoint directory"
    )
    parser.add_argument(
        "--env",
        "-e",
        default="SingleArmDrawerCleanup",
        help="Name of the dexmimicgen environment to evaluate on"
    )
    parser.add_argument(
        "--episodes",
        "-n",
        type=int,
        default=3,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos of the rollouts to the current directory"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=0,
        help="Random seed for reproducible evaluation (default: 0)"
    )

    args = parser.parse_args()

    results = evaluate_octo_checkpoint(
        checkpoint_path=args.checkpoint,
        env_name=args.env,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_video=args.save_video,
        verbose=not args.quiet,
        seed=args.seed,
        pca_enabled=False,  # PCA disabled by default for CLI
    )

    # Print summary
    print("\nEvaluation Results:")
    print("=" * 50)
    print(f"Average Return: {results['mean_return']:.3f} ± {results['std_return']:.3f}")
    print(f"Average Episode Length: {results['mean_length']:.1f}")
    print(f"Success Rate: {results['success_rate']:.1%}")

    if not args.quiet:
        print("\nIndividual Episodes:")
        for i, (ret, length, success, opened) in enumerate(zip(
            results["episode_returns"],
            results["episode_lengths"],
            results["successes"],
            results["opened"],
        )):
            print(f"  Episode {i+1}: Return={ret:.3f}, Length={length}, Success={success}, Opened={opened}")


if __name__ == "__main__":
    main()