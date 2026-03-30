import os
import json
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from modules.rgbd_similarity import RGBDSimilarity


class VisualMemorySelector:
    """
    Adaptive visual-memory selector for RGB-D frame sequences stored as .npy arrays.

    Logic:
    1. Compute an RGB-D similarity matrix over sampled frames.
    2. Compute threshold from the first off-diagonal:
           threshold = mean(D1) - sigma_k * std(D1)
    3. Always select first frame.
    4. Compare each incoming frame against the last selected keyframe.
       If similarity < threshold, select the frame.
    5. Always force the last frame.

    Notes:
    - rgb_frames shape expected: (N, H, W, 3)
    - depth_frames shape expected: (N, H, W) or (N, H, W, 1)
    """

    def __init__(self, sigma_k: float = 2.0, device: Optional[str] = None):
        self.sigma_k = float(sigma_k)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_model = RGBDSimilarity(device=self.device)

    @staticmethod
    def _validate_inputs(rgb_frames: np.ndarray, depth_frames: np.ndarray) -> None:
        if not isinstance(rgb_frames, np.ndarray) or not isinstance(depth_frames, np.ndarray):
            raise TypeError("rgb_frames and depth_frames must be numpy arrays.")

        if len(rgb_frames) != len(depth_frames):
            raise ValueError("rgb_frames and depth_frames must have the same number of frames.")

        if rgb_frames.ndim != 4 or rgb_frames.shape[-1] != 3:
            raise ValueError("rgb_frames must have shape (N, H, W, 3).")

        if depth_frames.ndim not in (3, 4):
            raise ValueError("depth_frames must have shape (N, H, W) or (N, H, W, 1).")

    @staticmethod
    def _normalize_depth_shape(depth_frame: np.ndarray) -> np.ndarray:
        if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
            return depth_frame[..., 0]
        return depth_frame

    @staticmethod
    def load_npy_arrays(rgb_npy_path: str, depth_npy_path: str) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(rgb_npy_path):
            raise FileNotFoundError(f"RGB npy file not found: {rgb_npy_path}")
        if not os.path.exists(depth_npy_path):
            raise FileNotFoundError(f"Depth npy file not found: {depth_npy_path}")

        rgb_frames = np.load(rgb_npy_path)
        depth_frames = np.load(depth_npy_path)
        return rgb_frames, depth_frames

    @staticmethod
    def sample_arrays(
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        sample_rate: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if sample_rate < 1:
            raise ValueError("sample_rate must be >= 1")

        indices = np.arange(len(rgb_frames))[::sample_rate]
        return rgb_frames[indices], depth_frames[indices], indices

    def extract_feature_bank(
        self,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray
    ) -> List[torch.Tensor]:
        self._validate_inputs(rgb_frames, depth_frames)

        feature_bank: List[torch.Tensor] = []

        with torch.no_grad():
            for rgb, depth in tqdm(
                zip(rgb_frames, depth_frames),
                total=len(rgb_frames),
                desc="Extracting RGB-D features"
            ):
                depth = self._normalize_depth_shape(depth)
                rgbd_tensor = self.similarity_model.preprocess_image(rgb, depth)
                feat = self.similarity_model.extract_features(rgbd_tensor)
                feature_bank.append(feat.cpu())

        return feature_bank

    def compute_similarity_matrix_from_features(
        self,
        feature_bank: List[torch.Tensor]
    ) -> np.ndarray:
        num_frames = len(feature_bank)
        sim_matrix = np.eye(num_frames, dtype=np.float32)

        for i in tqdm(range(num_frames), desc="Computing similarity matrix"):
            f1 = feature_bank[i].to(self.similarity_model.device)
            for j in range(i + 1, num_frames):
                f2 = feature_bank[j].to(self.similarity_model.device)
                score = self.similarity_model.compute_similarity(f1, f2)
                sim_matrix[i, j] = float(score)
                sim_matrix[j, i] = float(score)

        return sim_matrix

    def compute_similarity_matrix(
        self,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray
    ) -> np.ndarray:
        feature_bank = self.extract_feature_bank(rgb_frames, depth_frames)
        return self.compute_similarity_matrix_from_features(feature_bank)

    def compute_threshold(
        self,
        similarity_matrix: np.ndarray
    ) -> Dict[str, np.ndarray]:
        if similarity_matrix.ndim != 2 or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
            raise ValueError("similarity_matrix must be square.")

        off_diag = np.diag(similarity_matrix, k=1)

        if len(off_diag) == 0:
            raise ValueError("Similarity matrix is too small to compute first off-diagonal statistics.")

        mu = float(np.mean(off_diag))
        sigma = float(np.std(off_diag))
        threshold = float(mu - self.sigma_k * sigma)

        return {
            "threshold": threshold,
            "mu": mu,
            "sigma": sigma,
            "off_diag": off_diag,
        }

    def select_keyframes_from_matrix(
        self,
        similarity_matrix: np.ndarray,
        real_indices: Optional[np.ndarray] = None
    ) -> Dict:
        num_frames = similarity_matrix.shape[0]
        if num_frames == 0:
            raise ValueError("Empty similarity matrix.")

        stats = self.compute_threshold(similarity_matrix)
        threshold = stats["threshold"]

        selected_sampled_indices = [0]
        similarity_trace = [1.0]
        last_selected_idx = 0

        for i in range(1, num_frames):
            current_sim = float(similarity_matrix[last_selected_idx, i])
            similarity_trace.append(current_sim)

            if current_sim < threshold:
                selected_sampled_indices.append(i)
                last_selected_idx = i

        if selected_sampled_indices[-1] != num_frames - 1:
            selected_sampled_indices.append(num_frames - 1)

        if real_indices is None:
            selected_real_indices = selected_sampled_indices.copy()
        else:
            selected_real_indices = [int(real_indices[i]) for i in selected_sampled_indices]

        return {
            "selected_sampled_indices": [int(i) for i in selected_sampled_indices],
            "selected_real_indices": [int(i) for i in selected_real_indices],
            "similarity_trace": [float(x) for x in similarity_trace],
            "threshold": float(stats["threshold"]),
            "mu": float(stats["mu"]),
            "sigma": float(stats["sigma"]),
            "off_diag": stats["off_diag"],
        }

    def select_from_arrays(
        self,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        sample_rate: int = 1
    ) -> Dict:
        self._validate_inputs(rgb_frames, depth_frames)

        rgb_sampled, depth_sampled, real_indices = self.sample_arrays(
            rgb_frames, depth_frames, sample_rate=sample_rate
        )

        sim_matrix = self.compute_similarity_matrix(rgb_sampled, depth_sampled)
        result = self.select_keyframes_from_matrix(sim_matrix, real_indices=real_indices)

        result["similarity_matrix"] = sim_matrix
        result["sample_rate"] = int(sample_rate)
        result["num_total_frames"] = int(len(rgb_frames))
        result["num_sampled_frames"] = int(len(rgb_sampled))

        return result

    def save_outputs(
        self,
        output_dir: str,
        rgb_frames: np.ndarray,
        depth_frames: np.ndarray,
        result: Dict,
        save_matrix_csv: bool = True
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)

        selected_real_indices = result["selected_real_indices"]
        selected_rgbs = rgb_frames[selected_real_indices]
        selected_depths = depth_frames[selected_real_indices]

        np.save(os.path.join(output_dir, "selected_rgbs.npy"), selected_rgbs)
        np.save(os.path.join(output_dir, "selected_depths.npy"), selected_depths)

        pd.DataFrame({"frame_idx": selected_real_indices}).to_csv(
            os.path.join(output_dir, "selected_indices.csv"),
            index=False
        )

        metadata = {
            "selected_real_indices": selected_real_indices,
            "selected_sampled_indices": result["selected_sampled_indices"],
            "threshold": result["threshold"],
            "mu": result["mu"],
            "sigma": result["sigma"],
            "sample_rate": result["sample_rate"],
            "num_total_frames": result["num_total_frames"],
            "num_sampled_frames": result["num_sampled_frames"],
        }

        with open(os.path.join(output_dir, "selection_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        if save_matrix_csv:
            sim_matrix = result["similarity_matrix"]
            pd.DataFrame(sim_matrix).to_csv(
                os.path.join(output_dir, "similarity_matrix.csv"),
                index=False
            )

    def run_from_npy(
        self,
        rgb_npy_path: str,
        depth_npy_path: str,
        sample_rate: int = 1,
        output_dir: Optional[str] = None,
        save_matrix_csv: bool = True
    ) -> Dict:
        rgb_frames, depth_frames = self.load_npy_arrays(rgb_npy_path, depth_npy_path)
        result = self.select_from_arrays(rgb_frames, depth_frames, sample_rate=sample_rate)

        if output_dir is not None:
            self.save_outputs(
                output_dir=output_dir,
                rgb_frames=rgb_frames,
                depth_frames=depth_frames,
                result=result,
                save_matrix_csv=save_matrix_csv,
            )

        return result


def parse_args():
    parser = argparse.ArgumentParser(description="Adaptive visual memory selector for RGB-D npy logs.")
    parser.add_argument("--rgb_npy", type=str, required=True, help="Path to RGB frames .npy")
    parser.add_argument("--depth_npy", type=str, required=True, help="Path to depth frames .npy")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save selected visual memory")
    parser.add_argument("--sigma", type=float, default=2.0, help="Sigma multiplier in threshold = mean - sigma*std")
    parser.add_argument("--rate", type=int, default=1, help="Sampling rate (1 every N frames)")
    parser.add_argument("--no_matrix_csv", action="store_true", help="Do not save similarity matrix as CSV")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selector = VisualMemorySelector(sigma_k=args.sigma)
    result = selector.run_from_npy(
        rgb_npy_path=args.rgb_npy,
        depth_npy_path=args.depth_npy,
        sample_rate=args.rate,
        output_dir=args.output_dir,
        save_matrix_csv=not args.no_matrix_csv,
    )

    print("\nSelection finished.")
    print(f"Threshold: {result['threshold']:.4f}")
    print(f"Mean(D1): {result['mu']:.4f}")
    print(f"Std(D1): {result['sigma']:.4f}")
    print(f"Selected frames ({len(result['selected_real_indices'])}):")
    print(result["selected_real_indices"])