"""
Dataset Regeneration Script for dexmimicgen

This script regenerates datasets by replaying actions from bi-manual manipulation tasks
in a single-arm environment and re-rendering image observations at high resolution.

The regenerated datasets contain only single-arm (robot0) observations and actions,
filtering out all robot1 data from the original bi-manual datasets.

Image resolutions:
- agentview_image: 256x256
- robot0_eye_in_hand_image: 128x128

Args:
    --input_dataset (str): Path to the input HDF5 dataset file.
    --output_dataset (str): Path to save the regenerated HDF5 dataset file.
    --n (int, optional): Number of trajectories to regenerate (default: all).
    --filter_key (str, optional): Key to filter specific trajectories in the dataset.
    --verbose (bool): Enable additional logging.
    --video_path (str, optional): Path to save rollout video for visualization/debugging.
    --video_skip (int): Frame skip rate for video recording (default: 5).
    --render_image_names (list of str, optional): Camera names to include in video (default: agentview).
"""

import argparse
from bz2 import compress
import datetime
import json
import os
import random
import time

import h5py
import imageio
import numpy as np
import robosuite
from robosuite import load_composite_controller_config
from termcolor import colored

# IMPORTANT: you need to import the package to register the environments
import dexmimicgen
from sklearn.decomposition import PCA


def regenerate_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    verbose=False,
):
    """
    Helper function to regenerate a single trajectory using the simulator environment.
    Collects observations during playback and returns them for saving to new dataset.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        verbose (bool): Enable verbose logging

    Returns:
        dict: Collected observations from the trajectory
    """
    write_video = video_writer is not None
    video_count = 0

    if verbose:
        if initial_state.get("ep_meta") is not None:
            ep_meta = json.loads(initial_state["ep_meta"])
            lang = ep_meta.get("lang", None)
            if lang is not None:
                print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))

    try:
        reset_to(env, initial_state)
    except Exception as e:
        if verbose:
            print(colored(f"Warning: Failed to set initial state ({e}), using reset position", "yellow"))
        # Fallback: just load the model without setting state
        reset_to(env, {"model": initial_state["model"], "ep_meta": initial_state.get("ep_meta")})

    traj_len = states.shape[0]
    assert states.shape[0] == actions.shape[0]

    print(colored("Running episode...", "yellow"))

    # Initialize observation collection - only collect robot0 observations
    regenerated_obs = {
        'agentview_image': [],
        'robot0_eye_in_hand_image': [],
        'robot0_eef_pos': [],
        'robot0_eef_quat': [],
        'robot0_eef_quat_site': [],
        'robot0_gripper_qpos': [],
        'robot0_gripper_qvel': [],
        'robot0_joint_pos': [],
        'robot0_joint_pos_cos': [],
        'robot0_joint_pos_sin': [],
        'robot0_joint_vel': [],
    }
    regenerated_states = []

    success = False
    for i in range(traj_len):
        start = time.time()

        env.step(actions[i][:12])
        obs = env._get_observations()

        if env._check_success():
            success = True

        regenerated_states.append(env.sim.get_state().flatten())

        # Collect single-arm observations only
        for key in regenerated_obs.keys():
            if key in obs:
                if "image" in key:
                    regenerated_obs[key].append(obs[key][::-1])
                else:
                    regenerated_obs[key].append(obs[key])

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(height=512, width=512, camera_name=cam_name)[
                        ::-1
                    ]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

    if not success:
        print(colored("warning: playback did not succeed", "red"))

    # Convert lists to numpy arrays
    for key in regenerated_obs.keys():
        regenerated_obs[key] = np.array(regenerated_obs[key])
    
    regenerated_states = np.stack(regenerated_states)

    return regenerated_obs, regenerated_states


def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break


def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta


class ObservationKeyToModalityDict(dict):
    """
    Custom dictionary class with the sole additional purpose of automatically registering new "keys" at runtime
    without breaking. This is mainly for backwards compatibility, where certain keys such as "latent", "actions", etc.
    are used automatically by certain models (e.g.: VAEs) but were never specified by the user externally in their
    config. Thus, this dictionary will automatically handle those keys by implicitly associating them with the low_dim
    modality.
    """

    def __getitem__(self, item):
        # If a key doesn't already exist, warn the user and add default mapping
        if item not in self.keys():
            print(
                f"ObservationKeyToModalityDict: {item} not found,"
                f" adding {item} to mapping with assumed low_dim modality!"
            )
            self.__setitem__(item, "low_dim")
        return super(ObservationKeyToModalityDict, self).__getitem__(item)


def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return env._get_observations() if should_ret else None


def regenerate_dataset(args):
    # some arg checking
    write_video = args.video_path is not None
    if write_video and args.render_image_names is None:
        args.render_image_names = ["agentview"]

    env = None

    # create environment
    env_meta = get_env_metadata_from_dataset(dataset_path=args.input_dataset)
    env_meta["env_name"] = "SingleArmDrawerCleanup"

    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["env_name"] = "SingleArmDrawerCleanup"
    env_kwargs["robots"] = ["PandaDexRH"]
    env_kwargs["has_renderer"] = False
    env_kwargs["has_offscreen_renderer"] = True  # Always need offscreen renderer for camera observations
    env_kwargs["use_camera_obs"] = True
    env_kwargs["camera_names"] = ["agentview", "robot0_eye_in_hand"]
    env_kwargs["camera_heights"] = [256, 128]  # agentview: 256x256, robot0_eye_in_hand: 128x128
    env_kwargs["camera_widths"] = [256, 128]

    if args.verbose:
        print(
            colored(
                "Initializing environment for {}...".format(env_kwargs["env_name"]),
                "yellow",
            )
        )
    if "env_lang" in env_kwargs:
        env_kwargs.pop("env_lang")

    env = robosuite.make(**env_kwargs)

    # Open input and output files
    f_in = h5py.File(args.input_dataset, "r")
    f_out = h5py.File(args.output_dataset, "w")

    # Create data group
    data_group = f_out.create_group("data")

    # list of all demonstration episodes (sorted in increasing number order)
    if args.filter_key is not None:
        print("using filter key: {}".format(args.filter_key))
        demos = [
            elem.decode("utf-8")
            for elem in np.array(f_in["mask/{}".format(args.filter_key)])
        ]
    elif "data" in f_in.keys():
        demos = list(f_in["data"].keys())

    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to regenerate
    if args.n is not None:
        random.shuffle(demos)
        demos = demos[: args.n]

    # maybe dump video
    video_writer = None
    if write_video:
        video_writer = imageio.get_writer(args.video_path, fps=20)
    
    # PCA over hand joint position actions
    hand_actions_raw = []
    for demo in demos:
        hand_actions_raw.append(f_in[f"data/{demo}/actions"][:, 6:12])

    hand_actions_raw = np.concatenate(hand_actions_raw, axis=0)

    # Normalize data: PCA is sensitive to scale. 
    mean_action = np.mean(hand_actions_raw, axis=0)
    std_action = np.std(hand_actions_raw, axis=0)
    hand_actions_norm = (hand_actions_raw - mean_action) / (std_action + 1e-6)

    k = 2
    pca_reduced = PCA(n_components=k)
    pca_reduced.fit(hand_actions_norm)

    total_frames = 0
    for ind in range(len(demos)):
        ep = demos[ind]
        print(colored("\nRegenerating episode: {}".format(ep), "yellow"))

        states = f_in["data/{}/states".format(ep)][()]
        # hacky way to get single-arm states
        indices = [0] + list(range(1, 19)) + list(range(38, 47)) + list(range(47, 65)) + list(range(84, len(states[0])))
        single_arm_states = states[:, indices]
        initial_state = {}
        initial_state["model"] = env.sim.model.get_xml()
        initial_state["ep_meta"] = f_in["data/{}".format(ep)].attrs.get("ep_meta", None)
        initial_state["states"] = single_arm_states[0]

        # supply actions if using open-loop action playback
        actions = f_in["data/{}/actions".format(ep)][()]
        total_frames += actions.shape[0]

        # Regenerate observations
        new_obs, new_states = regenerate_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=single_arm_states,
            actions=actions,
            video_writer=video_writer,
            video_skip=args.video_skip,
            camera_names=args.render_image_names,
            verbose=args.verbose,
        )

        # PCA hand actions
        ep_hand_actions_norm = (actions[:, 6:12] - mean_action) / (std_action + 1e-6)
        pca_hand_actions = pca_reduced.transform(ep_hand_actions_norm)
        latent_actions = np.concatenate((actions[:, :6], pca_hand_actions), axis=-1)

        # Create demo group in output file
        demo_group = data_group.create_group(ep)

        # Copy attributes
        for attr_key in f_in[f"data/{ep}"].attrs.keys():
            demo_group.attrs[attr_key] = f_in[f"data/{ep}"].attrs[attr_key]

        # Create obs subgroup and save regenerated observations
        obs_group = demo_group.create_group("obs")
        for obs_key, obs_data in new_obs.items():
            obs_group.create_dataset(obs_key, data=obs_data, compression="gzip")

        demo_group.create_dataset("actions", data=actions[:,:12], compression="gzip")
        demo_group.create_dataset("pca_actions", data=latent_actions)
        demo_group.create_dataset("states", data=new_states, compression="gzip")
        f_out[f"data/{ep}"].attrs["model_file"] = initial_state["model"] 

        # Filter action_dict to only include right arm components for single-arm dataset
        if "action_dict" in f_in[f"data/{ep}"]:
            action_dict_group = f_in[f"data/{ep}/action_dict"]
            filtered_action_dict_group = demo_group.create_group("action_dict")

            # Only copy right arm action components (filter out left arm components)
            right_arm_keys = ["right_gripper", "right_rel_pos", "right_rel_rot_6d", "right_rel_rot_axis_angle"]
            for action_key in action_dict_group.keys():
                if action_key in right_arm_keys:
                    filtered_action_dict_group.create_dataset(
                        action_key,
                        data=action_dict_group[action_key][()],
                        compression="gzip"
                    )
            # Add joint positions for arm
            # robot = env.robots[0]
            # raw_arm_target = np.roll(new_states[:, np.array(robot._ref_joint_pos_indexes) + 1], -1, axis=0)
            # qpos_delta = raw_arm_target - new_states[:, np.array(robot._ref_joint_pos_indexes) + 1]
            # qpos_delta[-1] = np.zeros_like(qpos_delta[-1])
            # filtered_action_dict_group.create_dataset(
            #     "right_rel_qpos",
            #     data=qpos_delta,
            #     compression="gzip"
            # )

    # Copy mask group if it exists
    if "mask" in f_in.keys():
        f_in.copy("mask", f_out)
    
    # Set metadata
    f_out["data"].attrs["total"] = total_frames
    f_out["data"].attrs["env_args"] = json.dumps(env_meta)


    f_in.close()
    f_out.close()

    if write_video:
        print(colored(f"Saved video to {args.video_path}", "green"))
        video_writer.close()

    if env is not None:
        env.close()

    print(colored(f"Successfully regenerated dataset saved to {args.output_dataset}", "green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        required=True,
        help="path to save regenerated hdf5 dataset",
    )
    parser.add_argument(
        "--filter_key",
        type=str,
        default=None,
        help="(optional) filter key, to select a subset of trajectories in the file",
    )

    # number of trajectories to regenerate. If omitted, regenerate all of them.
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are regenerated",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log additional information",
    )

    # Dump a video of the regeneration rollout to the specified path
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="(optional) render regeneration rollout to this video file path",
    )

    # How often to write video frames during the regeneration
    parser.add_argument(
        "--video_skip",
        type=int,
        default=5,
        help="render frames to video every n steps",
    )

    # camera names to render for video
    parser.add_argument(
        "--render_image_names",
        type=str,
        nargs="+",
        default=[
            "agentview",
        ],
        help="(optional) camera name(s) to use for rendering to video",
    )

    args = parser.parse_args()
    regenerate_dataset(args)
