"""
PCA Action Post-Processing Wrapper for dexmimicgen environments.

This wrapper replaces the last 6 dimensions (hand part) of actions with
a 2-component PCA reconstruction, allowing models to operate in a
reduced action space.
"""

import gym
import numpy as np
from sklearn.decomposition import PCA
import h5py


class PCAActionWrapper(gym.Wrapper):
    """
    Gym wrapper that post-processes actions by replacing the hand part
    (last 6 dimensions) with a 2-component PCA reconstruction.

    Assumes the input action has shape [..., action_dim] where the last
    2 dimensions are PCA components for the hand, and the first dimensions
    are the original action space components (e.g., joint positions).
    """

    def __init__(self, env, pca_model, mean_action, std_action, hand_action_indices=(6, 12)):
        """
        Initialize the PCA action wrapper.

        Args:
            env: The gym environment to wrap
            pca_model: Fitted sklearn PCA model with 2 components
            mean_action: Mean of the original hand actions used for PCA fitting
            std_action: Standard deviation of the original hand actions used for PCA fitting
            hand_action_indices: Tuple (start, end) indicating which indices of the
                               full action correspond to hand actions. Default (6, 12)
                               assumes first 6 are joints, last 6 are hand.
        """
        super().__init__(env)
        self.pca_model = pca_model
        self.mean_action = mean_action
        self.std_action = std_action
        self.hand_start, self.hand_end = hand_action_indices
        self.hand_dim = self.hand_end - self.hand_start

        # Modify action space to reflect reduced dimensionality
        # Original action space should have shape (action_dim,) where action_dim includes 6 hand dims
        # New action space has 4 fewer dimensions (6 hand - 2 PCA)
        if hasattr(env.action_space, 'shape') and len(env.action_space.shape) == 1:
            original_dim = env.action_space.shape[0]
            new_dim = original_dim - (self.hand_dim - pca_model.n_components_)
            self.action_space = gym.spaces.Box(
                low=env.action_space.low[:new_dim] if hasattr(env.action_space, 'low') else -np.inf,
                high=env.action_space.high[:new_dim] if hasattr(env.action_space, 'high') else np.inf,
                shape=(new_dim,),
                dtype=env.action_space.dtype
            )
        else:
            # If action space is not well-defined, keep original
            self.action_space = env.action_space

    def step(self, action):
        """
        Post-process the action by reconstructing hand actions from PCA components.

        Args:
            action: Action array where last 2 dimensions are PCA components for hand

        Returns:
            Same as env.step() but with reconstructed full action
        """
        # Reconstruct hand actions from PCA components
        hand_latents = action[..., -self.pca_model.n_components_:]

        # Inverse transform to get normalized hand actions
        reconstructed_norm = self.pca_model.inverse_transform(hand_latents)

        # Denormalize to get original scale hand actions
        hand_actions = (reconstructed_norm * self.std_action) + self.mean_action

        # Combine with non-hand actions (first dimensions)
        non_hand_actions = action[..., :-self.pca_model.n_components_]
        full_action = np.concatenate([non_hand_actions, hand_actions], axis=-1)

        # Step the environment with reconstructed action
        return self.env.step(full_action)


def create_pca_from_dataset(dataset_path, hand_indices=(6, 12), n_components=2):
    """
    Create PCA model from dataset hand actions.

    Args:
        dataset_path: Path to HDF5 dataset file
        hand_indices: Tuple (start, end) for hand action indices in action array
        n_components: Number of PCA components (default 2)

    Returns:
        tuple: (pca_model, mean_action, std_action)
    """
    data = h5py.File(dataset_path, "r")
    hand_actions_raw = []

    # Collect all hand actions from dataset
    for ep in data["data"].keys():
        hand_actions_raw.append(data[f"data/{ep}/actions/"][:, hand_indices[0]:hand_indices[1]])

    hand_actions_raw = np.concatenate(hand_actions_raw, axis=0)

    # Normalize data
    mean_action = np.mean(hand_actions_raw, axis=0)
    std_action = np.std(hand_actions_raw, axis=0)
    hand_actions_norm = (hand_actions_raw - mean_action) / (std_action + 1e-6)

    # Fit PCA
    pca_model = PCA(n_components=n_components)
    pca_model.fit(hand_actions_norm)

    return pca_model, mean_action, std_action
