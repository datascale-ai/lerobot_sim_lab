#!/usr/bin/env python3
"""
SmolVLA Model Evaluation Script
Evaluate a trained SmolVLA policy on SO100 pick-and-place task
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# Set offline mode before imports
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import gymnasium as gym

try:
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False

import gym_hil

# Import scenario library for random cube positions
from lerobot_sim_lab.config.scenarios.pen_grab import PEN_SCENARIOS
from lerobot_sim_lab.config.scenarios.pick_place import SCENARIOS


def create_env(task="SO100PickCubeBase-v0", render=False, control_dt=1 / 30):
    """Create SO100 environment

    Args:
        task: Environment task name
        render: Whether to show visualization
        control_dt: Control timestep in seconds (default 1/30 ≈ 0.0333 = 30Hz)
                   Must match the training data frequency!
    """
    # Register environment
    sys.modules["gym_gym_manipulator"] = gym_hil
    if "gym_gym_manipulator/SO100PickCubeBase-v0" not in gym.registry:
        gym.register(
            id="gym_gym_manipulator/SO100PickCubeBase-v0",
            entry_point="gym_hil.envs:SO100PickCubeGymEnv",
            max_episode_steps=400,
            kwargs={"control_dt": control_dt},
        )
    if "gym_gym_manipulator/SO100GrabPenBase-v0" not in gym.registry:
        gym.register(
            id="gym_gym_manipulator/SO100GrabPenBase-v0",
            entry_point="gym_hil.envs:SO100GrabPenGymEnv",
            max_episode_steps=4000,
            kwargs={"control_dt": control_dt},
        )

    env = gym.make(f"gym_gym_manipulator/{task}")

    if render:
        from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

        # Keep sync_every_n_steps=1 for smooth visualization
        env = PassiveViewerWrapper(env, default_camera="camera_front_new", sync_every_n_steps=1)

    return env


def adapt_observation(
    obs, task_description="Pick up the red cube from the yellow region and place it in the purple target region"
):
    """
    Adapt gym environment observation to SmolVLA policy format

    Args:
        obs: Environment observation dict with 'state' and 'pixels' keys
        task_description: Language instruction for the task

    Returns:
        Dict compatible with SmolVLA policy input
    """
    # Extract state (first 6 values: qpos)
    state = torch.from_numpy(obs["state"][:6].astype(np.float32))

    # Extract images from pixels dict
    pixels = obs.get("pixels", {})

    # Convert images to (C, H, W) float32 format
    policy_obs = {
        "observation.state": state.unsqueeze(0),  # (1, 6)
        "task": task_description,  # Language instruction
    }

    # Add camera observations if available
    if "front" in pixels:
        img_front = torch.from_numpy(pixels["front"]).float() / 255.0
        if img_front.shape[-1] == 3:  # (H, W, C) -> (C, H, W)
            img_front = img_front.permute(2, 0, 1)
        policy_obs["observation.images.camera1"] = img_front.unsqueeze(0)  # (1, C, H, W)

    if "wrist" in pixels:
        img_wrist = torch.from_numpy(pixels["wrist"]).float() / 255.0
        if img_wrist.shape[-1] == 3:
            img_wrist = img_wrist.permute(2, 0, 1)
        policy_obs["observation.images.camera2"] = img_wrist.unsqueeze(0)

    return policy_obs


def evaluate_policy(
    policy,
    preprocessor,
    postprocessor,
    env,
    num_episodes=5,
    task_description=None,
    task_name=None,
    device="cuda",
    render=False,
    save_video=False,
    video_dir=None,
    randomize_cube_pos=True,
):
    """
    Evaluate policy on multiple episodes

    Args:
        policy: Trained SmolVLA policy
        preprocessor: Policy preprocessor pipeline
        postprocessor: Policy postprocessor pipeline
        env: Gymnasium environment
        num_episodes: Number of episodes to evaluate
        task_description: Language instruction (if None, uses default)
        device: Device to run on
        render: Whether to render visualization
        save_video: Whether to save episode videos
        video_dir: Directory to save videos (required if save_video=True)
        randomize_cube_pos: Whether to randomize cube position per episode

    Returns:
        Dict with evaluation statistics
    """
    is_grab_pen = task_name == "SO100GrabPenBase-v0"
    if task_description is None:
        if is_grab_pen:
            task_description = "Pick up the pen from the table and place it into the box."
        else:
            task_description = "Pick up the red cube from the yellow region and place it in the purple target region"

    results = {
        "successes": [],
        "episode_lengths": [],
        "episode_rewards": [],
    }

    for ep_idx in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {ep_idx + 1}/{num_episodes}")
        print(f"Task: {task_description}")

        if is_grab_pen:
            scenario_ids = [s["id"] for s in PEN_SCENARIOS if s["id"] != 0]
            scenario_id = scenario_ids[ep_idx % len(scenario_ids)]
            scenario = next(s for s in PEN_SCENARIOS if s["id"] == scenario_id)
            print(f"Scenario: {scenario['name']} (id={scenario_id})")
        else:
            # Randomly select cube position from scenario library
            if randomize_cube_pos:
                scenario = np.random.choice(SCENARIOS)
                cube_pos = scenario["cube_pos"]
                print(f"Scenario: {scenario['name']}")
                print(f"Cube position: ({cube_pos[0]:.4f}, {cube_pos[1]:.4f}, {cube_pos[2]:.4f})")
            else:
                cube_pos = None
                print("Cube position: Default (0.06, 0.135, 0.017)")

        print(f"{'='*60}")

        # Reset environment with custom scenario
        if is_grab_pen:
            reset_options = {"scenario_idx": scenario_id}
        else:
            reset_options = {"cube_pos": cube_pos} if cube_pos is not None else None
        obs, info = env.reset(options=reset_options)
        done = False
        truncated = False
        step_count = 0
        episode_reward = 0.0

        # Video recording
        frames = [] if save_video else None

        max_steps = getattr(env, "_max_episode_steps", None)
        if max_steps is None:
            max_steps = getattr(env, "spec", None) and getattr(env.spec, "max_episode_steps", None)
        if max_steps is None:
            max_steps = 450
        while not (done or truncated) and step_count < max_steps:
            # Capture frame for video (front camera view)
            if save_video and "pixels" in obs and "front" in obs["pixels"]:
                # Store frame as RGB (H, W, C)
                frame = obs["pixels"]["front"].copy()
                frames.append(frame)
            # Adapt observation for policy
            policy_obs = adapt_observation(obs, task_description)

            # Move to device
            for key in policy_obs:
                if isinstance(policy_obs[key], torch.Tensor):
                    policy_obs[key] = policy_obs[key].to(device)

            # Apply preprocessor (includes tokenization)
            policy_obs = preprocessor(policy_obs)

            # Get action from policy
            with torch.no_grad():
                action_tensor = policy.select_action(policy_obs)

            # Apply postprocessor
            action_tensor = postprocessor(action_tensor)

            # Convert to numpy
            action = action_tensor.cpu().numpy().flatten()

            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Step {step_count}: reward={episode_reward:.3f}")

        # Episode summary
        success = info.get("is_success", False) or info.get("succeed", False)
        results["successes"].append(success)
        results["episode_lengths"].append(step_count)
        results["episode_rewards"].append(episode_reward)

        status = "✅ SUCCESS" if success else "❌ FAILURE"
        print(f"\n{status}")
        print(f"  Steps: {step_count}")
        print(f"  Reward: {episode_reward:.3f}")

        # Save video if requested
        if save_video and frames and video_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_filename = f"episode_{ep_idx+1:03d}_{timestamp}_{'success' if success else 'failure'}.mp4"
            video_path = Path(video_dir) / video_filename

            # Create video writer
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = 30  # Match training data fps
            video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

            # Write frames
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)

            video_writer.release()
            print(f"  💾 Video saved: {video_path}")

    # Calculate statistics
    success_rate = np.mean(results["successes"]) * 100
    avg_length = np.mean(results["episode_lengths"])
    avg_reward = np.mean(results["episode_rewards"])

    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Length: {avg_length:.1f} steps")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA policy")
    parser.add_argument("--policy", type=str, required=True, help="Path to trained policy checkpoint")
    parser.add_argument("--task", type=str, default="SO100PickCubeBase-v0", help="Task name")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--task-description", type=str, default=None, help="Language instruction for the task")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu)")
    parser.add_argument("--render", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--control-dt",
        type=float,
        default=1 / 30,
        help="Control timestep in seconds (default 1/30≈0.033=30Hz). Must match training data!",
    )
    parser.add_argument("--save-video", action="store_true", help="Save episode videos (front camera)")
    parser.add_argument("--video-dir", type=str, default="./eval_videos", help="Directory to save videos")
    parser.add_argument("--randomize-cube", action="store_true", help="Randomize cube position per episode")

    args = parser.parse_args()

    # Check policy path
    policy_path = Path(args.policy)
    if not policy_path.exists():
        print(f"❌ Policy path does not exist: {policy_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("SmolVLA Policy Evaluation")
    print(f"{'='*60}")
    print(f"Policy: {args.policy}")
    print(f"Task: {args.task}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print(f"Render: {args.render}")
    print(f"Control DT: {args.control_dt}s ({1/args.control_dt:.0f}Hz)")
    print(f"Randomize Cube: {args.randomize_cube}")
    print(f"Save Video: {args.save_video}")
    if args.save_video:
        print(f"Video Dir: {args.video_dir}")
    print(f"{'='*60}\n")

    # Create video directory if needed
    if args.save_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Video directory created: {video_dir}\n")

    # Load policy
    print("Loading policy...")
    policy = SmolVLAPolicy.from_pretrained(
        pretrained_name_or_path=args.policy,
    )
    policy = policy.to(args.device)
    policy.eval()
    print("✅ Policy loaded successfully\n")

    # Load preprocessor and postprocessor
    print("Loading preprocessor and postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=args.policy,
    )
    print("✅ Preprocessor and postprocessor loaded\n")

    # Create environment
    print("Creating environment...")
    env = create_env(args.task, render=args.render, control_dt=args.control_dt)
    print("✅ Environment created\n")

    # Run evaluation
    results = evaluate_policy(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        env=env,
        num_episodes=args.num_episodes,
        task_description=args.task_description,
        task_name=args.task,
        device=args.device,
        render=args.render,
        save_video=args.save_video,
        video_dir=args.video_dir if args.save_video else None,
        randomize_cube_pos=args.randomize_cube,
    )

    env.close()
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
