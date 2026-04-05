import os
import json
import time
import yaml
import joblib
from datetime import datetime

import cv2
import habitat
import numpy as np
import quaternion
import torch

from habitat.config import read_write
from habitat.config.default import get_agent_config

from modules.pose_config import PoseConfig
from modules.scenes import Scene
from modules.rgbd_similarity import RGBDSimilarity
from modules.feature_based_point_cloud_registration import FeatureBasedPointCloudRegistration


# ---------------------------------------------------------------------
# USER CONFIG
# ---------------------------------------------------------------------
EXP_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/exp_config/config.yaml"
HABITAT_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test-0.yaml"
EXAMPLES_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/examples/config.yaml"

VISUAL_MEMORY_DIR = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/visual_memories/rep_rep-3_224_20260330_125808"
MODEL_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_navigator_outputs_cv_tolerant/navigator.joblib"
SCALER_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_navigator_outputs_cv_tolerant/scaler.joblib"

OUTPUT_DIR = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/autonav_logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAV_CONF = "LightGlue"
FEATURE_MODE = "mnn"
ID_RUN_FOR_FEATURES = 3000
MAX_STEPS = 1500


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
FEATURE_COLS = ["rmse", "tx", "ty", "tz", "qw", "qx", "qy", "qz", "sim_score"]

LABEL_TO_ACTION = {
    "forward": "move_forward",
    "left": "turn_left",
    "right": "turn_right",
}

KEY_TO_LABEL = {
    ord("w"): "forward",
    ord("a"): "left",
    ord("s"): "accept_prediction",
    ord("d"): "right",
    ord("u"): "update",
    ord("f"): "finish",
    27: "finish",  # ESC
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def quaternion_to_list(q) -> list:
    return [float(q.w), float(q.x), float(q.y), float(q.z)]


def build_env(scene_path: str):
    overrides = [
        "habitat.environment.max_episode_steps=10000",
        "habitat.simulator.forward_step_size=0.1",
        "habitat.simulator.turn_angle=2",
    ]

    config = habitat.get_config(HABITAT_CONFIG_PATH, overrides=overrides)

    with read_write(config):
        agent_config = get_agent_config(sim_config=config.habitat.simulator)
        agent_config.sim_sensors.depth_sensor.normalize_depth = False

    env = habitat.Env(config=config)
    env.episodes[0].scene_id = scene_path
    env.reset()
    return env, config


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model_and_scaler(model_path: str, scaler_path: str):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def load_visual_memory(vm_dir: str):
    rgbs = np.load(os.path.join(vm_dir, "selected_rgbs.npy"))
    depths = np.load(os.path.join(vm_dir, "selected_depths.npy"))
    return rgbs, depths


def build_feature_modules(device: str):
    feature_config = load_yaml(EXAMPLES_CONFIG_PATH)

    rgbd_similarity = RGBDSimilarity(device=device)

    feature_registration = FeatureBasedPointCloudRegistration(
        config=feature_config,
        device=device,
        id_run=ID_RUN_FOR_FEATURES,
        feature_nav_conf=FEATURE_NAV_CONF,
        feature_mode=FEATURE_MODE,
        topological_map=None,
        manual_operation=False,
    )

    return rgbd_similarity, feature_registration


def predict_navigation_label(
    observed_rgb,
    observed_depth,
    key_rgb,
    key_depth,
    rgbd_similarity,
    feature_registration,
    model,
    scaler,
):
    timings = {}

    t0 = time.time()
    sim_score = float(
        rgbd_similarity.compute_image_similarity(
            observed_rgb, observed_depth, key_rgb, key_depth
        )
    )
    timings["similarity_time"] = time.time() - t0

    t0 = time.time()
    bot_lost, est_quaternion, rmse, est_t_source_target, est_T = feature_registration.compute_relative_pose(
        observed_rgb, observed_depth, key_rgb, key_depth
    )
    timings["registration_time"] = time.time() - t0

    if bot_lost or est_quaternion is None or rmse is None or est_t_source_target is None:
        raise RuntimeError("Relative pose estimation failed or not enough matches were found.")

    qw = float(est_quaternion.w)
    qx = float(est_quaternion.x)
    qy = float(est_quaternion.y)
    qz = float(est_quaternion.z)

    tx, ty, tz = [float(v) for v in np.asarray(est_t_source_target).reshape(-1)[:3]]
    rmse = float(rmse)

    feature_vector = np.array([[rmse, tx, ty, tz, qw, qx, qy, qz, sim_score]], dtype=np.float32)

    t0 = time.time()
    feature_vector_scaled = scaler.transform(feature_vector)
    timings["scaler_time"] = time.time() - t0

    t0 = time.time()
    pred_label = model.predict(feature_vector_scaled)[0]
    pred_proba = model.predict_proba(feature_vector_scaled)[0]
    timings["prediction_time"] = time.time() - t0

    class_to_proba = {cls: float(prob) for cls, prob in zip(model.classes_, pred_proba)}
    pred_conf = class_to_proba[pred_label]

    out = {
        "pred_label": str(pred_label),
        "pred_confidence": float(pred_conf),
        "probabilities": class_to_proba,
        "rmse": rmse,
        "tx": tx,
        "ty": ty,
        "tz": tz,
        "qw": qw,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "sim_score": sim_score,
        "bot_lost": bool(bot_lost),
        "timings": timings,
    }
    return out


def depth_to_color(depth_img: np.ndarray) -> np.ndarray:
    depth_vis = depth_img.squeeze() if depth_img.ndim == 3 else depth_img
    depth_vis = depth_vis.astype(np.float32)
    if depth_vis.max() > depth_vis.min():
        depth_vis = (depth_vis - depth_vis.min()) / (depth_vis.max() - depth_vis.min())
    else:
        depth_vis = depth_vis * 0.0
    depth_vis = (depth_vis * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def draw_text_block(img: np.ndarray, lines: list, x: int = 8, y0: int = 22, dy: int = 22):
    out = img.copy()
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(
            out,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(line),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def show_debug_view(observed_rgb, key_rgb, debug_lines, window_name="Autonomous Navigation POC"):
    obs_bgr = cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2BGR)
    key_bgr = cv2.cvtColor(key_rgb, cv2.COLOR_RGB2BGR)

    obs_bgr = draw_text_block(obs_bgr, debug_lines[:6])
    key_bgr = draw_text_block(key_bgr, debug_lines[6:])

    h = max(obs_bgr.shape[0], key_bgr.shape[0])
    w = obs_bgr.shape[1] + key_bgr.shape[1]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[: obs_bgr.shape[0], : obs_bgr.shape[1]] = obs_bgr
    canvas[: key_bgr.shape[0], obs_bgr.shape[1] : obs_bgr.shape[1] + key_bgr.shape[1]] = key_bgr

    cv2.imshow(window_name, canvas)


def ask_yes_no_terminal(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    ans = input(prompt + suffix).strip().lower()
    if ans == "":
        return default
    return ans in ("y", "yes", "1", "true")


def get_user_decision():
    """
    Keys:
      w -> manual forward
      a -> manual left
      s -> accept autonomous prediction
      d -> manual right
      f / ESC -> finish
    """
    while True:
        key = cv2.waitKey(0)
        if key in KEY_TO_LABEL:
            return KEY_TO_LABEL[key]
        print("Invalid key. Use: w, a, s, d, u, f (or ESC).")


def resolve_oscillation(pred_label: str, action_buffer: list, probabilities: dict) -> tuple:
    """
    Detect immediate left/right oscillation using the last two predicted actions.

    If the last two actions are one 'left' and one 'right', replace the current
    action with the more probable of 'forward' or 'update'.

    Returns:
        processed_label, oscillation_detected, updated_action_buffer
    """
    updated_buffer = action_buffer.copy()
    updated_buffer.append(pred_label)

    # Keep only the last two actions
    if len(updated_buffer) > 2:
        updated_buffer.pop(0)

    oscillation_detected = False
    processed_label = pred_label

    if (
        len(updated_buffer) == 2
        and updated_buffer[0] in ["left", "right"]
        and updated_buffer[1] in ["left", "right"]
        and updated_buffer[0] != updated_buffer[1]
    ):
        oscillation_detected = True

        p_forward = float(probabilities.get("forward", 0.0))
        p_update = float(probabilities.get("update", 0.0))

        if p_forward > p_update:
            processed_label = "forward"
        else:
            processed_label = "update"

        # Replace the last action in the buffer with the corrected action
        updated_buffer[-1] = processed_label

    return processed_label, oscillation_detected, updated_buffer


def execute_label(env, label: str):
    if label == "update":
        return None
    if label == "finish":
        return None
    action = LABEL_TO_ACTION.get(label, None)
    if action is None:
        raise ValueError(f"Unsupported label for execution: {label}")
    observations = env.step(action)
    return observations

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    display_frames = ask_yes_no_terminal("Display observed/key RGB frames?", default=True)

    exp_cfg = load_yaml(EXP_CONFIG_PATH)
    scene = Scene(exp_cfg["config"]["scene"])
    path_name = exp_cfg["config"]["path_name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env, habitat_cfg = build_env(scene.get_path())
    forward_step_size = float(habitat_cfg.habitat.simulator.forward_step_size)
    turn_angle = float(habitat_cfg.habitat.simulator.turn_angle)

    start_position = PoseConfig.poses[path_name]["position"]
    start_rotation = PoseConfig.poses[path_name]["quaternion"]
    env.sim.set_agent_state(start_position, start_rotation)
    observations = env.step(env.action_space.sample())

    vm_rgbs, vm_depths = load_visual_memory(VISUAL_MEMORY_DIR)
    num_keyframes = len(vm_rgbs)

    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    rgbd_similarity, feature_registration = build_feature_modules(device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"autonav_{path_name}_{timestamp}"

    observed_rgbs = []
    observed_depths = []
    logs = []

    vm_image_index = 0
    step_idx = 0

    # Oscillation handling memory
    action_buffer = []

    print("Environment ready.")
    print(f"Scene path       : {scene.get_path()}")
    print(f"path_name        : {path_name}")
    print(f"Visual memory len: {num_keyframes}")
    print(f"Forward step     : {forward_step_size}")
    print(f"Turn angle       : {turn_angle}")
    print("Controls         : w=manual forward, a=manual left, s=accept prediction, d=manual right, u=manual update, f=finish")
    try:
        while not env.episode_over and step_idx < MAX_STEPS:
            observed_rgb = observations["rgb"]
            observed_depth = observations["depth"]

            key_rgb = vm_rgbs[vm_image_index]
            key_depth = vm_depths[vm_image_index]

            total_t0 = time.time()
            prediction_ok = True
            prediction_error = None

            try:
                pred_out = predict_navigation_label(
                    observed_rgb=observed_rgb,
                    observed_depth=observed_depth,
                    key_rgb=key_rgb,
                    key_depth=key_depth,
                    rgbd_similarity=rgbd_similarity,
                    feature_registration=feature_registration,
                    model=model,
                    scaler=scaler,
                )
            except Exception as e:
                prediction_ok = False
                prediction_error = str(e)
                pred_out = {
                    "pred_label": "finish",
                    "pred_confidence": 0.0,
                    "probabilities": {},
                    "rmse": None,
                    "tx": None,
                    "ty": None,
                    "tz": None,
                    "qw": None,
                    "qx": None,
                    "qy": None,
                    "qz": None,
                    "sim_score": None,
                    "bot_lost": True,
                    "timings": {
                        "similarity_time": None,
                        "registration_time": None,
                        "scaler_time": None,
                        "prediction_time": None,
                    },
                }
            raw_pred_label = pred_out["pred_label"]
            processed_pred_label, oscillation_detected, action_buffer = resolve_oscillation(
                raw_pred_label,
                action_buffer,
                pred_out["probabilities"]
            )
            debug_lines = [
                f"OBS | step={step_idx}",
                f"OBS | kf_idx={vm_image_index}/{num_keyframes-1}",
                f"OBS | raw_pred={raw_pred_label}",
                f"OBS | proc_pred={processed_pred_label}",
                f"OBS | conf={pred_out['pred_confidence']:.3f}",
                f"OBS | sim={pred_out['sim_score'] if pred_out['sim_score'] is not None else 'None'}",
                f"KEY | accept=s",
                f"KEY | manual: w/a/d/u",
                f"KEY | finish=f",
                f"KEY | oscillation={oscillation_detected}",
                f"KEY | bot_lost={pred_out['bot_lost']}",
                f"KEY | ok={prediction_ok}",
            ]

            if display_frames:
                show_debug_view(observed_rgb, key_rgb, debug_lines)

            user_decision = get_user_decision()

            if user_decision == "finish":
                executed_label = "finish"
                manual_override = True
            elif user_decision == "accept_prediction":
                executed_label = processed_pred_label
                manual_override = False
            else:
                executed_label = user_decision
                manual_override = True

            agent_state = env.sim.get_agent_state()

            observed_rgbs.append(observed_rgb.copy())
            observed_depths.append(observed_depth.copy())

            if executed_label == "update":
                old_vm_image_index = vm_image_index
                vm_image_index += 1
                if vm_image_index >= num_keyframes:
                    vm_image_index = num_keyframes - 1
                    executed_label = "finish"
                    finished_by_vm = True
                else:
                    finished_by_vm = False
            else:
                old_vm_image_index = vm_image_index
                finished_by_vm = False

            action_exec_t0 = time.time()
            if executed_label in ("forward", "left", "right"):
                observations = execute_label(env, executed_label)
            action_exec_time = time.time() - action_exec_t0

            total_loop_time = time.time() - total_t0

            logs.append(
                {
                    "step_idx": int(step_idx),
                    "path_name": path_name,
                    "scene_path": scene.get_path(),
                    "vm_image_index_before": int(old_vm_image_index),
                    "vm_image_index_after": int(vm_image_index),
                    "num_keyframes": int(num_keyframes),
                    "predicted_label_raw": raw_pred_label,
                    "predicted_label_processed": processed_pred_label,
                    "oscillation_detected": bool(oscillation_detected),
                    "predicted_confidence": float(pred_out["pred_confidence"]),
                    "predicted_probabilities": pred_out["probabilities"],
                    "executed_label": executed_label,
                    "manual_override": bool(manual_override),
                    "prediction_ok": bool(prediction_ok),
                    "prediction_error": prediction_error,
                    "bot_lost": bool(pred_out["bot_lost"]),
                    "rmse": pred_out["rmse"],
                    "tx": pred_out["tx"],
                    "ty": pred_out["ty"],
                    "tz": pred_out["tz"],
                    "qw": pred_out["qw"],
                    "qx": pred_out["qx"],
                    "qy": pred_out["qy"],
                    "qz": pred_out["qz"],
                    "sim_score": pred_out["sim_score"],
                    "similarity_time": pred_out["timings"]["similarity_time"],
                    "registration_time": pred_out["timings"]["registration_time"],
                    "scaler_time": pred_out["timings"]["scaler_time"],
                    "prediction_time": pred_out["timings"]["prediction_time"],
                    "action_execution_time": float(action_exec_time),
                    "total_loop_time": float(total_loop_time),
                    "agent_position": [float(x) for x in agent_state.position],
                    "agent_rotation_quaternion_wxyz": quaternion_to_list(agent_state.rotation),
                }
            )

            print(
                f"Step {step_idx:04d} | "
                f"kf={old_vm_image_index:03d}->{vm_image_index:03d} | "
                f"raw={raw_pred_label} | proc={processed_pred_label} | exec={executed_label} | "
                f"osc={oscillation_detected} | manual={manual_override} | "
                f"sim={pred_out['sim_score'] if pred_out['sim_score'] is not None else 'None'} | "
                f"rmse={pred_out['rmse'] if pred_out['rmse'] is not None else 'None'}"
            )

            step_idx += 1

            if executed_label == "finish" or finished_by_vm:
                break

        print("\nRun finished.")
        print(f"Logged steps: {len(logs)}")

    finally:
        if display_frames:
            cv2.destroyAllWindows()
        env.close()

    ensure_dir(OUTPUT_DIR)

    rgb_path = os.path.join(OUTPUT_DIR, f"{run_name}_rgb.npy")
    depth_path = os.path.join(OUTPUT_DIR, f"{run_name}_depth.npy")
    metadata_path = os.path.join(OUTPUT_DIR, f"{run_name}_metadata.json")

    np.save(rgb_path, np.stack(observed_rgbs, axis=0) if observed_rgbs else np.empty((0,)))
    np.save(depth_path, np.stack(observed_depths, axis=0) if observed_depths else np.empty((0,)))

    metadata = {
        "run_name": run_name,
        "path_name": path_name,
        "scene_path": scene.get_path(),
        "visual_memory_dir": VISUAL_MEMORY_DIR,
        "visual_memory_length": int(num_keyframes),
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH,
        "forward_step_size": forward_step_size,
        "turn_angle": turn_angle,
        "device": device,
        "display_frames": bool(display_frames),
        "feature_nav_conf": FEATURE_NAV_CONF,
        "feature_mode": FEATURE_MODE,
        "feature_cols": FEATURE_COLS,
        "steps": logs,
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nSaved files:")
    print(rgb_path)
    print(depth_path)
    print(metadata_path)


if __name__ == "__main__":
    main()