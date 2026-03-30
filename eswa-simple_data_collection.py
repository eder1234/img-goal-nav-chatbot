import os
import json
from datetime import datetime

import cv2
import habitat
import numpy as np
import quaternion
import yaml

from habitat.config import read_write
from habitat.config.default import get_agent_config

from modules.pose_config import PoseConfig
from modules.scenes import Scene


def quaternion_to_list(q):
    """
    Convert a quaternion object to a JSON-serializable [w, x, y, z] list.
    """
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_manual_command_and_action(key):
    """
    Map keyboard input to:
    - user-level command label
    - Habitat action name
    """
    keymap = {
        ord("w"): ("forward", "move_forward"),
        ord("a"): ("left", "turn_left"),
        ord("d"): ("right", "turn_right"),
        ord("f"): ("finish", None),
        27: ("finish", None),  # ESC
    }
    return keymap.get(key, (None, None))


def build_env(config_path, scene_path):
    """
    Create Habitat environment with manual-control-friendly settings.
    """
    overrides = [
        "habitat.environment.max_episode_steps=10000",
        "habitat.simulator.forward_step_size=0.1",
        "habitat.simulator.turn_angle=2",
    ]

    config = habitat.get_config(config_path, overrides=overrides)

    with read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.depth_sensor.normalize_depth = False

    env = habitat.Env(config=config)
    env.episodes[0].scene_id = scene_path
    env.reset()

    return env, config


def display_observation(rgb, depth):
    """
    Show RGB and depth observations for manual teleoperation.
    """
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    depth_vis = depth.squeeze() if depth.ndim == 3 else depth
    depth_vis = depth_vis.astype(np.float32)

    if depth_vis.max() > depth_vis.min():
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    combined = np.hstack([rgb_bgr, depth_vis])
    cv2.imshow("Manual Data Collection - RGB | Depth", combined)


def collect_manual_path():
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    exp_config_path = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/exp_config/config.yaml"
    habitat_config_path = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test-0.yaml"

    output_root = "manual_operation/simple_logs"
    ensure_dir(output_root)

    # ------------------------------------------------------------------
    # Load experiment config
    # ------------------------------------------------------------------
    with open(exp_config_path, "r") as f:
        config_exp = yaml.safe_load(f)

    scene = Scene(config_exp["config"]["scene"])
    path_name = config_exp["config"]["path_name"]
    id_run = config_exp["config"]["id_run"]

    environment_name = scene.get_name() if hasattr(scene, "get_name") else config_exp["config"]["scene"]
    scene_path = scene.get_path()

    # ------------------------------------------------------------------
    # Build environment
    # ------------------------------------------------------------------
    env, habitat_config = build_env(habitat_config_path, scene_path)

    # ------------------------------------------------------------------
    # Set reliable initial pose from PoseConfig
    # ------------------------------------------------------------------
    start_position = PoseConfig.poses[path_name]["position"]
    start_rotation = PoseConfig.poses[path_name]["quaternion"]

    env.sim.set_agent_state(start_position, start_rotation)
    observations = env.step(env.action_space.sample())

    forward_step_size = float(habitat_config.habitat.simulator.forward_step_size)
    turn_angle = float(habitat_config.habitat.simulator.turn_angle)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    rgb_frames = []
    depth_frames = []
    steps_metadata = []

    # Save starting pose in metadata
    start_state = env.sim.get_agent_state()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{environment_name}_{path_name}_{id_run}_{run_timestamp}"

    print("Environment ready.")
    print(f"Environment name: {environment_name}")
    print(f"Scene path: {scene_path}")
    print(f"Path name: {path_name}")
    print(f"Forward step size: {forward_step_size}")
    print(f"Turn angle: {turn_angle}")
    print("Controls: w=forward, a=left, d=right, f=finish, ESC=finish")

    step_index = 0

    try:
        while not env.episode_over:
            rgb = observations["rgb"]
            depth = observations["depth"]

            display_observation(rgb, depth)

            key = cv2.waitKey(0)
            user_command, habitat_action = get_manual_command_and_action(key)

            if user_command is None:
                print("Invalid key. Use w, a, d, f, or ESC.")
                continue

            # Log the observation BEFORE executing the action.
            agent_state = env.sim.get_agent_state()

            rgb_frames.append(rgb.copy())
            depth_frames.append(depth.copy())

            step_record = {
                "frame_id": int(step_index),
                "user_command": user_command,
                "habitat_action": habitat_action,
                "position": [float(x) for x in agent_state.position],
                "rotation_quaternion_wxyz": quaternion_to_list(agent_state.rotation),
            }
            steps_metadata.append(step_record)

            print(
                f"Step {step_index:04d} | "
                f"command={user_command} | action={habitat_action}"
            )

            if user_command == "finish":
                break

            observations = env.step(habitat_action)
            step_index += 1

    finally:
        cv2.destroyAllWindows()
        env.close()

    # ------------------------------------------------------------------
    # Convert frame lists to arrays
    # ------------------------------------------------------------------
    rgb_array = np.stack(rgb_frames, axis=0) if rgb_frames else np.empty((0,))
    depth_array = np.stack(depth_frames, axis=0) if depth_frames else np.empty((0,))

    # ------------------------------------------------------------------
    # Build JSON metadata
    # ------------------------------------------------------------------
    metadata = {
        "run_name": run_name,
        "environment_name": environment_name,
        "scene_path": scene_path,
        "path_name": path_name,
        "id_run": id_run,
        "forward_step_size": forward_step_size,
        "turn_angle": turn_angle,
        "start_pose": {
            "position": [float(x) for x in start_state.position],
            "rotation_quaternion_wxyz": quaternion_to_list(start_state.rotation),
        },
        "num_logged_frames": int(len(steps_metadata)),
        "steps": steps_metadata,
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    rgb_out = os.path.join(output_root, f"{run_name}_rgb.npy")
    depth_out = os.path.join(output_root, f"{run_name}_depth.npy")
    json_out = os.path.join(output_root, f"{run_name}_metadata.json")

    np.save(rgb_out, rgb_array)
    np.save(depth_out, depth_array)

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(rgb_out)
    print(depth_out)
    print(json_out)
    print(f"Logged frames: {len(steps_metadata)}")


if __name__ == "__main__":
    collect_manual_path()