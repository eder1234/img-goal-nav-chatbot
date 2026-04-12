import os
import json
import time
import yaml
import joblib
import argparse
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
from modules.obstacle_avoidance import ObstacleAvoidance
from modules.localizer import Localizer


# ---------------------------------------------------------------------
# DEFAULT USER CONFIG
# ---------------------------------------------------------------------
EXP_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/exp_config/config.yaml"
HABITAT_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/habitat-lab/habitat/config/benchmark/nav/pointnav/pointnav_habitat_test-0.yaml"
EXAMPLES_CONFIG_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/examples/config.yaml"

DEFAULT_VISUAL_MEMORY_DIR = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/visual_memories/rep_rep-3_224_20260330_125808"

DEFAULT_MODEL_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_navigator_outputs_cv_tolerant/navigator_relu_adam__hls-(128,64,32)__alpha-0.001__lr-0.0001_cv.joblib"
DEFAULT_SCALER_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_navigator_outputs_cv_tolerant/scaler_relu_adam__hls-(128,64,32)__alpha-0.001__lr-0.0001_cv.joblib"

DEFAULT_LOCALIZER_MODEL_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_localizer_outputs_cv/localizer_rf_balanced_cv.joblib"
DEFAULT_LOCALIZER_SCALER_PATH = "/home/rodriguez/Documents/GitHub/habitat/habitat-lab/manual_operation/trained_localizer_outputs_cv/localizer_scaler_rf_balanced_cv.joblib"

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
    ord("r"): "relocalize",
    ord("f"): "finish",
    27: "finish",
}

TERMINAL_TO_LABEL = {
    "w": "forward",
    "a": "left",
    "s": "accept_prediction",
    "d": "right",
    "u": "update",
    "r": "relocalize",
    "f": "finish",
    "esc": "finish",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Autonomous retracing with learned localizer.")
    parser.add_argument("--scene", type=str, default=None, help="Scene key override, e.g. rep")
    parser.add_argument("--init_loc", type=str, default=None, help="Initial location / path_name override, e.g. rep-3")
    parser.add_argument("--vm", type=str, default=DEFAULT_VISUAL_MEMORY_DIR, help="Visual memory directory")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Navigator model path")
    parser.add_argument("--scaler", type=str, default=DEFAULT_SCALER_PATH, help="Navigator scaler path")
    parser.add_argument("--localizer_model", type=str, default=DEFAULT_LOCALIZER_MODEL_PATH, help="Learned localizer model path")
    parser.add_argument("--localizer_scaler", type=str, default=DEFAULT_LOCALIZER_SCALER_PATH, help="Learned localizer scaler path")
    parser.add_argument("--no_loc", action="store_true", help="Disable learned localizer")
    parser.add_argument("--no_obs", action="store_true", help="Disable obstacle avoidance module")
    parser.add_argument("--no_save", action="store_true", help="Do not save logs and observations")
    parser.add_argument("--display_pose", action="store_true", help="Print current pose in terminal")
    return parser.parse_args()


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
    pose_out = feature_registration.compute_relative_pose(
        observed_rgb, observed_depth, key_rgb, key_depth
    )
    timings["registration_time"] = time.time() - t0

    if len(pose_out) == 5:
        bot_lost, est_quaternion, rmse, est_t_source_target, est_T = pose_out
    elif len(pose_out) == 4:
        bot_lost, est_quaternion, rmse, est_t_source_target = pose_out
        est_T = None
    else:
        raise RuntimeError(f"Unexpected compute_relative_pose output length: {len(pose_out)}")

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


def draw_text_block(img: np.ndarray, lines: list, x: int = 8, y0: int = 22, dy: int = 22):
    out = img.copy()
    for i, line in enumerate(lines):
        y = y0 + i * dy
        cv2.putText(out, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def show_debug_view(observed_rgb, key_rgb, debug_lines, window_name="Autonomous Retracing"):
    obs_bgr = cv2.cvtColor(observed_rgb, cv2.COLOR_RGB2BGR)
    key_bgr = cv2.cvtColor(key_rgb, cv2.COLOR_RGB2BGR)

    n_half = (len(debug_lines) + 1) // 2
    obs_bgr = draw_text_block(obs_bgr, debug_lines[:n_half])
    key_bgr = draw_text_block(key_bgr, debug_lines[n_half:])

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


def get_user_decision(display_frames: bool):
    if display_frames:
        while True:
            key = cv2.waitKey(0)
            if key in KEY_TO_LABEL:
                return KEY_TO_LABEL[key]
            print("Invalid key. Use: w, a, s, d, u, r, f (or ESC).")
    else:
        while True:
            raw = input("Command [w/a/s/d/u/r/f]: ").strip().lower()
            if raw in TERMINAL_TO_LABEL:
                return TERMINAL_TO_LABEL[raw]
            print("Invalid command. Use: w, a, s, d, u, r, f.")


def resolve_oscillation(pred_label: str, action_buffer: list, probabilities: dict) -> tuple:
    updated_buffer = action_buffer.copy()
    updated_buffer.append(pred_label)

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
        processed_label = "forward" if p_forward > p_update else "update"
        updated_buffer[-1] = processed_label

    return processed_label, oscillation_detected, updated_buffer


def execute_label(env, label: str):
    if label in ("update", "finish", "relocalize"):
        return None
    action = LABEL_TO_ACTION.get(label, None)
    if action is None:
        raise ValueError(f"Unsupported label for execution: {label}")
    observations = env.step(action)
    return observations


def summarize_localization_for_log(loc_result: dict):
    selected = loc_result.get("selected_candidate", None)
    return {
        "trigger": loc_result.get("trigger"),
        "step_idx": int(loc_result.get("step_idx", -1)),
        "current_vm_index_before": int(loc_result.get("current_vm_index_before", -1)),
        "selected_index": loc_result.get("selected_index"),
        "current_vm_index_after": int(loc_result.get("current_vm_index_after", -1)),
        "localization_success": bool(loc_result.get("localization_success", False)),
        "num_candidates_total": int(loc_result.get("num_candidates_total", 0)),
        "num_candidates_admissible": int(loc_result.get("num_candidates_admissible", 0)),
        "message": loc_result.get("message"),
        "selected_candidate_summary": None if selected is None else {
            "candidate_index": int(selected["candidate_index"]),
            "localization_score": float(selected["localization_score"]),
            "raw_model_score": None if selected.get("raw_model_score") is None else float(selected["raw_model_score"]),
            "rmse": None if selected.get("rmse") is None or not np.isfinite(selected["rmse"]) else float(selected["rmse"]),
            "tx": None if selected.get("tx") is None or not np.isfinite(selected["tx"]) else float(selected["tx"]),
            "ty": None if selected.get("ty") is None or not np.isfinite(selected["ty"]) else float(selected["ty"]),
            "tz": None if selected.get("tz") is None or not np.isfinite(selected["tz"]) else float(selected["tz"]),
            "sim_score": None if selected.get("sim_score") is None or not np.isfinite(selected["sim_score"]) else float(selected["sim_score"]),
        },
    }


def maybe_print_pose(agent_state, step_idx: int, enabled: bool):
    if not enabled:
        return
    pos = [float(x) for x in agent_state.position]
    rot = quaternion_to_list(agent_state.rotation)
    print(f"POSE | step={step_idx} | position={pos} | rotation_wxyz={rot}")


def maybe_store_observation(observed_rgbs, observed_depths, rgb, depth, enabled: bool):
    if not enabled:
        return
    observed_rgbs.append(rgb.copy())
    observed_depths.append(depth.copy())


def maybe_store_log(logs, record: dict, enabled: bool):
    if not enabled:
        return
    logs.append(record)


def main():
    args = parse_args()
    display_frames = ask_yes_no_terminal("Display observed/key RGB frames?", default=True)

    exp_cfg = load_yaml(EXP_CONFIG_PATH)
    base_scene_name = exp_cfg["config"]["scene"]
    base_path_name = exp_cfg["config"]["path_name"]

    scene_name = args.scene if args.scene is not None else base_scene_name
    path_name = args.init_loc if args.init_loc is not None else base_path_name
    visual_memory_dir = args.vm

    scene = Scene(scene_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env, habitat_cfg = build_env(scene.get_path())
    forward_step_size = float(habitat_cfg.habitat.simulator.forward_step_size)
    turn_angle = float(habitat_cfg.habitat.simulator.turn_angle)

    start_position = PoseConfig.poses[path_name]["position"]
    start_rotation = PoseConfig.poses[path_name]["quaternion"]
    env.sim.set_agent_state(start_position, start_rotation)
    observations = env.step(env.action_space.sample())

    vm_rgbs, vm_depths = load_visual_memory(visual_memory_dir)
    num_keyframes = len(vm_rgbs)

    model, scaler = load_model_and_scaler(args.model, args.scaler)
    rgbd_similarity, feature_registration = build_feature_modules(device)

    obstacle_avoidance = None if args.no_obs else ObstacleAvoidance()
    localizer = None if args.no_loc else Localizer(
        model_path=args.localizer_model,
        scaler_path=args.localizer_scaler,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"autoret_localizer_{path_name}_{timestamp}"

    collect_outputs = not args.no_save

    observed_rgbs = []
    observed_depths = []
    logs = []
    localization_history = []

    vm_image_index = 0
    step_idx = 0
    action_buffer = []

    print("Environment ready.")
    print(f"Scene name             : {scene_name}")
    print(f"Scene path             : {scene.get_path()}")
    print(f"Init location          : {path_name}")
    print(f"Visual memory dir      : {visual_memory_dir}")
    print(f"Visual memory len      : {num_keyframes}")
    print(f"Forward step           : {forward_step_size}")
    print(f"Turn angle             : {turn_angle}")
    print(f"Learned localizer      : {not args.no_loc}")
    print(f"Obstacle avoidance     : {not args.no_obs}")
    print(f"Display pose           : {args.display_pose}")
    print(f"Display frames         : {display_frames}")
    print(f"No save mode           : {args.no_save}")
    print("Controls               : w=manual forward, a=manual left, s=accept prediction, d=manual right, u=manual update, r=re-localize, f=finish")

    try:
        initial_localization_result = None
        if localizer is not None:
            try:
                initial_localization_result = localizer.localize(
                    observed_rgb=observations["rgb"],
                    observed_depth=observations["depth"],
                    vm_rgbs=vm_rgbs,
                    vm_depths=vm_depths,
                    rgbd_similarity=rgbd_similarity,
                    feature_registration=feature_registration,
                    current_vm_index=vm_image_index,
                    trigger="startup",
                    step_idx=step_idx,
                )
                localization_history.append(summarize_localization_for_log(initial_localization_result))
                if initial_localization_result["localization_success"]:
                    vm_image_index = int(initial_localization_result["selected_index"])
                    print(f"[Localizer/startup] Selected keyframe index: {vm_image_index}")
                else:
                    print("[Localizer/startup] No scored candidate found. Keeping vm_image_index=0.")
            except Exception as e:
                initial_localization_result = {
                    "trigger": "startup",
                    "step_idx": int(step_idx),
                    "current_vm_index_before": 0,
                    "selected_index": None,
                    "current_vm_index_after": 0,
                    "localization_success": False,
                    "num_candidates_total": 0,
                    "num_candidates_admissible": 0,
                    "message": f"localizer_exception: {str(e)}",
                    "selected_candidate": None,
                    "candidates": [],
                }
                localization_history.append(summarize_localization_for_log(initial_localization_result))
                print(f"[Localizer/startup] Failed: {e}")

        while not env.episode_over and step_idx < MAX_STEPS:
            observed_rgb = observations["rgb"]
            observed_depth = observations["depth"]

            key_rgb = vm_rgbs[vm_image_index]
            key_depth = vm_depths[vm_image_index]

            total_t0 = time.time()
            prediction_ok = True
            prediction_error = None
            latest_localization_event = None

            agent_state_before = env.sim.get_agent_state()
            maybe_print_pose(agent_state_before, step_idx, args.display_pose)

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
                pred_out["probabilities"],
            )

            obstacle_info = {
                "label_before_obstacle_avoidance": processed_pred_label,
                "label_after_obstacle_avoidance": processed_pred_label,
                "obstacle_detected": False,
                "obstacle_reason": "obstacle_avoidance_disabled",
                "central_valid_ratio": None,
                "central_min_depth": None,
                "central_median_depth": None,
                "central_close_percentile_depth": None,
                "chosen_avoidance_label": None,
            }

            label_after_obstacle_avoidance = processed_pred_label
            if obstacle_avoidance is not None:
                obstacle_info = obstacle_avoidance.adjust_label(
                    observed_depth=observed_depth,
                    probabilities=pred_out["probabilities"],
                    proposed_label=processed_pred_label,
                )
                label_after_obstacle_avoidance = obstacle_info["label_after_obstacle_avoidance"]

            debug_lines = [
                f"OBS | step={step_idx}",
                f"OBS | kf_idx={vm_image_index}/{num_keyframes-1}",
                f"OBS | raw_pred={raw_pred_label}",
                f"OBS | osc_pred={processed_pred_label}",
                f"OBS | obs_pred={label_after_obstacle_avoidance}",
                f"OBS | conf={pred_out['pred_confidence']:.3f}",
                f"OBS | sim={pred_out['sim_score'] if pred_out['sim_score'] is not None else 'None'}",
                f"KEY | accept=s",
                f"KEY | manual=w/a/d/u",
                f"KEY | relocalize=r",
                f"KEY | finish=f",
                f"KEY | osc={oscillation_detected}",
                f"KEY | obstacle={obstacle_info['obstacle_detected']}",
                f"KEY | bot_lost={pred_out['bot_lost']}",
                f"KEY | ok={prediction_ok}",
            ]

            if display_frames:
                show_debug_view(observed_rgb, key_rgb, debug_lines)

            user_decision = get_user_decision(display_frames=display_frames)

            if user_decision == "relocalize":
                maybe_store_observation(
                    observed_rgbs,
                    observed_depths,
                    observed_rgb,
                    observed_depth,
                    enabled=collect_outputs,
                )

                if localizer is None:
                    relocalization_result = {
                        "trigger": "manual_relocalize",
                        "step_idx": int(step_idx),
                        "current_vm_index_before": int(vm_image_index),
                        "selected_index": None,
                        "current_vm_index_after": int(vm_image_index),
                        "localization_success": False,
                        "num_candidates_total": 0,
                        "num_candidates_admissible": 0,
                        "message": "localizer_disabled_by_flag",
                        "selected_candidate": None,
                        "candidates": [],
                    }
                    print("[Localizer/manual] Disabled by --no_loc.")
                else:
                    try:
                        relocalization_result = localizer.localize(
                            observed_rgb=observed_rgb,
                            observed_depth=observed_depth,
                            vm_rgbs=vm_rgbs,
                            vm_depths=vm_depths,
                            rgbd_similarity=rgbd_similarity,
                            feature_registration=feature_registration,
                            current_vm_index=vm_image_index,
                            trigger="manual_relocalize",
                            step_idx=step_idx,
                        )
                        if relocalization_result["localization_success"]:
                            vm_image_index = int(relocalization_result["selected_index"])
                            print(f"[Localizer/manual] Selected keyframe index: {vm_image_index}")
                        else:
                            print("[Localizer/manual] No scored candidate found. Keeping current keyframe.")
                    except Exception as e:
                        relocalization_result = {
                            "trigger": "manual_relocalize",
                            "step_idx": int(step_idx),
                            "current_vm_index_before": int(vm_image_index),
                            "selected_index": None,
                            "current_vm_index_after": int(vm_image_index),
                            "localization_success": False,
                            "num_candidates_total": 0,
                            "num_candidates_admissible": 0,
                            "message": f"localizer_exception: {str(e)}",
                            "selected_candidate": None,
                            "candidates": [],
                        }
                        print(f"[Localizer/manual] Failed: {e}")

                latest_localization_event = summarize_localization_for_log(relocalization_result)
                localization_history.append(latest_localization_event)

                total_loop_time = time.time() - total_t0
                maybe_store_log(
                    logs,
                    {
                        "step_idx": int(step_idx),
                        "path_name": path_name,
                        "scene_name": scene_name,
                        "scene_path": scene.get_path(),
                        "vm_image_index_before": int(latest_localization_event["current_vm_index_before"]),
                        "vm_image_index_after": int(vm_image_index),
                        "num_keyframes": int(num_keyframes),
                        "predicted_label_raw": raw_pred_label,
                        "predicted_label_processed": processed_pred_label,
                        "predicted_label_after_obstacle_avoidance": label_after_obstacle_avoidance,
                        "oscillation_detected": bool(oscillation_detected),
                        "predicted_confidence": float(pred_out["pred_confidence"]),
                        "predicted_probabilities": pred_out["probabilities"],
                        "executed_label": "relocalize",
                        "manual_override": True,
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
                        "action_execution_time": 0.0,
                        "total_loop_time": float(total_loop_time),
                        "agent_position": [float(x) for x in agent_state_before.position],
                        "agent_rotation_quaternion_wxyz": quaternion_to_list(agent_state_before.rotation),
                        "obstacle_avoidance_enabled": obstacle_avoidance is not None,
                        "obstacle_detected": obstacle_info["obstacle_detected"],
                        "obstacle_reason": obstacle_info["obstacle_reason"],
                        "central_valid_ratio": obstacle_info["central_valid_ratio"],
                        "central_min_depth": obstacle_info["central_min_depth"],
                        "central_median_depth": obstacle_info["central_median_depth"],
                        "central_close_percentile_depth": obstacle_info["central_close_percentile_depth"],
                        "label_before_obstacle_avoidance": obstacle_info["label_before_obstacle_avoidance"],
                        "label_after_obstacle_avoidance": obstacle_info["label_after_obstacle_avoidance"],
                        "chosen_avoidance_label": obstacle_info["chosen_avoidance_label"],
                        "localization": latest_localization_event,
                    },
                    enabled=collect_outputs,
                )

                print(
                    f"Step {step_idx:04d} | relocalize | "
                    f"kf={latest_localization_event['current_vm_index_before']:03d}->{vm_image_index:03d} | "
                    f"success={latest_localization_event['localization_success']}"
                )
                step_idx += 1
                continue

            if user_decision == "finish":
                executed_label = "finish"
                manual_override = True
            elif user_decision == "accept_prediction":
                executed_label = label_after_obstacle_avoidance
                manual_override = False
            else:
                executed_label = user_decision
                manual_override = True

            maybe_store_observation(
                observed_rgbs,
                observed_depths,
                observed_rgb,
                observed_depth,
                enabled=collect_outputs,
            )

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

            maybe_store_log(
                logs,
                {
                    "step_idx": int(step_idx),
                    "path_name": path_name,
                    "scene_name": scene_name,
                    "scene_path": scene.get_path(),
                    "vm_image_index_before": int(old_vm_image_index),
                    "vm_image_index_after": int(vm_image_index),
                    "num_keyframes": int(num_keyframes),
                    "predicted_label_raw": raw_pred_label,
                    "predicted_label_processed": processed_pred_label,
                    "predicted_label_after_obstacle_avoidance": label_after_obstacle_avoidance,
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
                    "agent_position": [float(x) for x in agent_state_before.position],
                    "agent_rotation_quaternion_wxyz": quaternion_to_list(agent_state_before.rotation),
                    "obstacle_avoidance_enabled": obstacle_avoidance is not None,
                    "obstacle_detected": obstacle_info["obstacle_detected"],
                    "obstacle_reason": obstacle_info["obstacle_reason"],
                    "central_valid_ratio": obstacle_info["central_valid_ratio"],
                    "central_min_depth": obstacle_info["central_min_depth"],
                    "central_median_depth": obstacle_info["central_median_depth"],
                    "central_close_percentile_depth": obstacle_info["central_close_percentile_depth"],
                    "label_before_obstacle_avoidance": obstacle_info["label_before_obstacle_avoidance"],
                    "label_after_obstacle_avoidance": obstacle_info["label_after_obstacle_avoidance"],
                    "chosen_avoidance_label": obstacle_info["chosen_avoidance_label"],
                    "localization": latest_localization_event,
                },
                enabled=collect_outputs,
            )

            print(
                f"Step {step_idx:04d} | "
                f"kf={old_vm_image_index:03d}->{vm_image_index:03d} | "
                f"raw={raw_pred_label} | proc={processed_pred_label} | "
                f"obs={label_after_obstacle_avoidance} | exec={executed_label} | "
                f"osc={oscillation_detected} | obs_det={obstacle_info['obstacle_detected']} | "
                f"manual={manual_override} | "
                f"sim={pred_out['sim_score'] if pred_out['sim_score'] is not None else 'None'} | "
                f"rmse={pred_out['rmse'] if pred_out['rmse'] is not None else 'None'}"
            )

            step_idx += 1

            if executed_label == "finish" or finished_by_vm:
                break

        print("\nRun finished.")
        print(f"Logged steps: {len(logs) if collect_outputs else 0}")

    finally:
        if display_frames:
            cv2.destroyAllWindows()
        env.close()

    if args.no_save:
        print("Skipping save operations.")
        return

    ensure_dir(OUTPUT_DIR)

    rgb_path = os.path.join(OUTPUT_DIR, f"{run_name}_rgb.npy")
    depth_path = os.path.join(OUTPUT_DIR, f"{run_name}_depth.npy")
    metadata_path = os.path.join(OUTPUT_DIR, f"{run_name}_metadata.json")

    np.save(rgb_path, np.stack(observed_rgbs, axis=0) if observed_rgbs else np.empty((0,)))
    np.save(depth_path, np.stack(observed_depths, axis=0) if observed_depths else np.empty((0,)))

    metadata = {
        "run_name": run_name,
        "path_name": path_name,
        "scene_name": scene_name,
        "scene_path": scene.get_path(),
        "visual_memory_dir": visual_memory_dir,
        "visual_memory_length": int(num_keyframes),
        "model_path": args.model,
        "scaler_path": args.scaler,
        "localizer_model_path": None if args.no_loc else args.localizer_model,
        "localizer_scaler_path": None if args.no_loc else args.localizer_scaler,
        "forward_step_size": forward_step_size,
        "turn_angle": turn_angle,
        "device": device,
        "display_frames": bool(display_frames),
        "display_pose": bool(args.display_pose),
        "feature_nav_conf": FEATURE_NAV_CONF,
        "feature_mode": FEATURE_MODE,
        "feature_cols": FEATURE_COLS,
        "localization_enabled": localizer is not None,
        "obstacle_avoidance_enabled": obstacle_avoidance is not None,
        "localization_history": localization_history,
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