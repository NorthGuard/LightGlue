"""LightGlue feature matching function."""
from pathlib import Path
import torch
import json
import numpy as np
import cv2

from lightglue import LightGlue, rbd
from .versioning import get_versioning_info

__version__ = "1.0.0"

RANSAC_THRESHOLD = 7.0


def _match_features_pair(matcher, feats0: dict, feats1: dict):
    """
    Match two feature sets using LightGlue and compute match quality metrics.
    
    Args:
        matcher: LightGlue matcher instance
        feats0: Features from image 0 (dict with keypoints, descriptors, etc.)
        feats1: Features from image 1 (dict with keypoints, descriptors, etc.)
    
    Returns:
        dict: Match results with num_matches, num_inliers, homography_matrix, matched_keypoints
    """
    # Ensure features are in the right format (with batch dimension)
    if feats0["keypoints"].dim() == 2:
        feats0 = {k: v[None] if isinstance(v, torch.Tensor) else v for k, v in feats0.items()}
    if feats1["keypoints"].dim() == 2:
        feats1 = {k: v[None] if isinstance(v, torch.Tensor) else v for k, v in feats1.items()}
    
    # Match features
    matches = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches = [rbd(x) for x in [feats0, feats1, matches]]
    
    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    match_indices = matches["matches"]
    
    # Get matched keypoints
    mkpts0 = kpts0[match_indices[..., 0]].cpu().numpy()
    mkpts1 = kpts1[match_indices[..., 1]].cpu().numpy()
    
    num_matches = len(mkpts0)
    
    result = {
        "num_matches": num_matches,
        "matched_keypoints_0": mkpts0.tolist(),
        "matched_keypoints_1": mkpts1.tolist(),
        "num_inliers": 0,
        "homography_matrix": None,
    }
    
    if num_matches < 4:
        return result
    
    # Compute Homography: image1 -> image0
    M, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, RANSAC_THRESHOLD)
    
    # Count inliers (mask contains 1 for inliers, 0 for outliers)
    num_inliers = int(np.sum(mask)) if mask is not None else 0
    
    result["num_inliers"] = num_inliers
    result["homography_matrix"] = M.tolist() if M is not None else None
    
    return result


def match_features(feats0: dict, feats1: dict, feats1_flipped: dict = None, device: torch.device = None, features: str = "superpoint"):
    """
    Match SuperPoint features using LightGlue to determine match quality.
    
    Args:
        feats0: Features from image 0 (dict with keypoints, descriptors, keypoint_scores, etc.)
        feats1: Features from image 1 (dict with keypoints, descriptors, keypoint_scores, etc.)
        feats1_flipped: Optional features from horizontally flipped image 1
        device: Torch device (cuda or cpu). If None, uses feats0 device.
        features: Feature type for LightGlue matcher (default: "superpoint")
    
    Returns:
        dict: Match results with comparison between feats1 and feats1_flipped (if provided)
    """
    # Determine device
    if device is None:
        if isinstance(feats0["keypoints"], torch.Tensor):
            device = feats0["keypoints"].device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure features are tensors on correct device
    def ensure_tensor(x, device):
        if isinstance(x, list):
            return torch.tensor(x, device=device)
        elif isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        return x
    
    feats0 = {k: ensure_tensor(v, device) if isinstance(v, (list, np.ndarray, torch.Tensor)) else v 
              for k, v in feats0.items()}
    feats1 = {k: ensure_tensor(v, device) if isinstance(v, (list, np.ndarray, torch.Tensor)) else v 
              for k, v in feats1.items()}
    
    # Load matcher
    matcher = LightGlue(features=features).eval().to(device)
    
    # Match image0 with image1
    match_original = _match_features_pair(matcher, feats0, feats1)
    
    result = {
        "match_original": match_original,
    }
    
    # Match image0 with flipped image1 if provided
    if feats1_flipped is not None:
        feats1_flipped = {k: ensure_tensor(v, device) if isinstance(v, (list, np.ndarray, torch.Tensor)) else v 
                         for k, v in feats1_flipped.items()}
        match_flipped = _match_features_pair(matcher, feats0, feats1_flipped)
        result["match_flipped"] = match_flipped
        
        # Determine which match is better
        inliers_orig = match_original["num_inliers"]
        inliers_flipped = match_flipped["num_inliers"]
        
        result["best_match"] = "flipped" if inliers_flipped > inliers_orig else "original"
        result["inliers_comparison"] = {
            "original": inliers_orig,
            "flipped": inliers_flipped
        }
        
        # Add best match's homography to top level for easy access
        if result["best_match"] == "flipped":
            result["homography_matrix"] = match_flipped["homography_matrix"]
        else:
            result["homography_matrix"] = match_original["homography_matrix"]
    else:
        result["best_match"] = "original"
        result["homography_matrix"] = match_original["homography_matrix"]
    
    return result


def __main__():
    """Test function for running feature matching from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Match SuperPoint features using LightGlue')
    parser.add_argument('feats0', type=str, help='Path to JSON file with features from image 0')
    parser.add_argument('feats1', type=str, help='Path to JSON file with features from image 1')
    parser.add_argument('--feats1-flipped', type=str, default=None, help='Path to JSON file with features from flipped image 1')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file')
    parser.add_argument('--features', type=str, default='superpoint', help='Feature type for LightGlue (default: superpoint)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load feature files
    feats0_path = Path(args.feats0)
    feats1_path = Path(args.feats1)
    
    if not feats0_path.exists():
        print(f"Error: File {args.feats0} not found.")
        return
    if not feats1_path.exists():
        print(f"Error: File {args.feats1} not found.")
        return
    
    with open(feats0_path, 'r') as f:
        feats0_data = json.load(f)
    with open(feats1_path, 'r') as f:
        feats1_data = json.load(f)
    
    # Extract features from JSON (assuming they're in a "features" key)
    feats0_raw = feats0_data.get("features", feats0_data)
    feats1_raw = feats1_data.get("features", feats1_data)
    
    # Check if feats1 has a "flipped" key (from compute_superpoint_for_image.py)
    feats1_flipped_raw = None
    if "flipped" in feats1_raw:
        feats1_flipped_raw = feats1_raw["flipped"]
        print("Note: Found 'flipped' key in feats1, will use it for comparison")
        # Remove "flipped" key from main feats1 dict
        feats1_raw = {k: v for k, v in feats1_raw.items() if k != "flipped"}
    
    # Convert lists to tensors
    feats0 = {
        "keypoints": torch.tensor(feats0_raw["keypoints"], device=device),
        "descriptors": torch.tensor(feats0_raw["descriptors"], device=device),
        "keypoint_scores": torch.tensor(feats0_raw.get("keypoint_scores", []), device=device) if feats0_raw.get("keypoint_scores") else None,
        "image_size": torch.tensor(feats0_raw.get("image_size", [0, 0]), device=device) if feats0_raw.get("image_size") else None,
    }
    feats1 = {
        "keypoints": torch.tensor(feats1_raw["keypoints"], device=device),
        "descriptors": torch.tensor(feats1_raw["descriptors"], device=device),
        "keypoint_scores": torch.tensor(feats1_raw.get("keypoint_scores", []), device=device) if feats1_raw.get("keypoint_scores") else None,
        "image_size": torch.tensor(feats1_raw.get("image_size", [0, 0]), device=device) if feats1_raw.get("image_size") else None,
    }
    
    feats1_flipped = None
    feats1_flipped_path = None
    # Use explicit --feats1-flipped file if provided (overrides "flipped" key)
    if args.feats1_flipped:
        feats1_flipped_path = Path(args.feats1_flipped)
        if not feats1_flipped_path.exists():
            print(f"Error: File {args.feats1_flipped} not found.")
            return
        
        with open(feats1_flipped_path, 'r') as f:
            feats1_flipped_data = json.load(f)
        
        feats1_flipped_raw = feats1_flipped_data.get("features", feats1_flipped_data)
        print("Note: Using explicit --feats1-flipped file (overrides 'flipped' key in feats1)")
    # Otherwise, use the "flipped" key from feats1 if it exists
    elif feats1_flipped_raw is not None:
        feats1_flipped_raw = feats1_flipped_raw
    
    # Convert flipped features to tensors if we have them
    if feats1_flipped_raw is not None:
        feats1_flipped = {
            "keypoints": torch.tensor(feats1_flipped_raw["keypoints"], device=device),
            "descriptors": torch.tensor(feats1_flipped_raw["descriptors"], device=device),
            "keypoint_scores": torch.tensor(feats1_flipped_raw.get("keypoint_scores", []), device=device) if feats1_flipped_raw.get("keypoint_scores") else None,
            "image_size": torch.tensor(feats1_flipped_raw.get("image_size", [0, 0]), device=device) if feats1_flipped_raw.get("image_size") else None,
        }

    # Match features
    print(f"\n=== Matching features ===")
    print(f"Image 0 features: {len(feats0['keypoints'])} keypoints")
    print(f"Image 1 features: {len(feats1['keypoints'])} keypoints")
    if feats1_flipped:
        print(f"Image 1 flipped features: {len(feats1_flipped['keypoints'])} keypoints")
    
    try:
        match_result = match_features(feats0, feats1, feats1_flipped, device, features=args.features)
        
        print(f"\n=== Match Results ===")
        print(f"Original match: {match_result['match_original']['num_matches']} matches, {match_result['match_original']['num_inliers']} inliers")
        if feats1_flipped:
            print(f"Flipped match: {match_result['match_flipped']['num_matches']} matches, {match_result['match_flipped']['num_inliers']} inliers")
            print(f"\nBest match: {match_result['best_match'].upper()}")
            print(f"  - Original inliers: {match_result['inliers_comparison']['original']}")
            print(f"  - Flipped inliers: {match_result['inliers_comparison']['flipped']}")
            print(f"  - Difference: {abs(match_result['inliers_comparison']['original'] - match_result['inliers_comparison']['flipped'])} inliers")
        
        # Print homography matrix
        if match_result.get("homography_matrix"):
            print(f"\n=== Homography Matrix (image1 -> image0) ===")
            homography = np.array(match_result["homography_matrix"])
            print(homography)
        else:
            print(f"\nNote: No homography matrix computed (insufficient matches)")
        
    except Exception as e:
        print(f"Error matching features: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get versioning information
    versioning_info = get_versioning_info(__version__)
    
    # Prepare output data
    output_data = {
        "versioning": versioning_info,
        "feats0_path": str(feats0_path),
        "feats1_path": str(feats1_path),
        "match_result": match_result
    }
    
    if feats1_flipped:
        if feats1_flipped_path:
            output_data["feats1_flipped_path"] = str(feats1_flipped_path)
        else:
            output_data["feats1_flipped_source"] = "from feats1['flipped'] key"
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = feats0_path.parent / f"{feats0_path.stem}_vs_{feats1_path.stem}_match.json"
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Match results saved to {output_path}")
    print(f"  - Best match: {match_result['best_match']}")
    print(f"  - Inliers: {match_result['match_original']['num_inliers']}")
    if feats1_flipped:
        print(f"  - Flipped inliers: {match_result['match_flipped']['num_inliers']}")


if __name__ == "__main__":
    __main__()


__all__ = ["match_features", "__version__", "__main__"]

