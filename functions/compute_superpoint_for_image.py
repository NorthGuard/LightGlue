"""SuperPoint feature extraction function."""
from pathlib import Path
import torch
import json

from lightglue import SuperPoint, load_image, rbd
from .versioning import get_versioning_info

__version__ = "1.0.0"


def _extract_single_image_features(extractor, image: torch.Tensor):
    """
    Extract features from a single image tensor and format for output.
    
    Args:
        extractor: SuperPoint extractor instance
        image: Image tensor (already on correct device)
    
    Returns:
        dict: Formatted features dictionary
    """
    feats = extractor.extract(image)
    feats = rbd(feats)  # Remove batch dimension
    
    result = {
        "keypoints": feats["keypoints"].cpu().numpy().tolist(),
        "descriptors": feats["descriptors"].cpu().numpy().tolist(),
        "keypoint_scores": feats.get("keypoint_scores", None)
    }
    
    if result["keypoint_scores"] is not None:
        result["keypoint_scores"] = result["keypoint_scores"].cpu().numpy().tolist()
    
    result["num_keypoints"] = len(result["keypoints"])
    result["image_size"] = feats.get("image_size", None)
    if result["image_size"] is not None:
        result["image_size"] = result["image_size"].cpu().numpy().tolist()
    
    return result


def extract_features(image: torch.Tensor, device: torch.device, include_flipped: bool = False, max_keypoints: int = 2048):
    """
    Extract SuperPoint features from an image tensor.
    
    Args:
        image: Image tensor (shape: [3, H, W] or [1, 3, H, W], already on correct device)
        device: Torch device (cuda or cpu)
        include_flipped: If True, also extract features from horizontally flipped image
        max_keypoints: Maximum number of keypoints to extract
    
    Returns:
        dict: Dictionary containing features and metadata
    """
    # Load extractor
    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
    
    # Ensure image is on correct device and has batch dimension
    if image.dim() == 3:
        image = image[None]  # Add batch dimension
    image = image.to(device)
    
    # Extract features from original image
    result = _extract_single_image_features(extractor, image)
    
    # Extract features from flipped image if requested
    if include_flipped:
        image_flipped = torch.flip(image, dims=[-1])
        result["flipped"] = _extract_single_image_features(extractor, image_flipped)
    
    return result


def __main__():
    """Test function for running the feature extraction from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract SuperPoint features from an image')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, default=None, help='Path to output JSON file (default: image_name_features.json)')
    parser.add_argument('--include-flipped', action='store_true', help='Also extract features from horizontally flipped image')
    parser.add_argument('--max-keypoints', type=int, default=2048, help='Maximum number of keypoints to extract')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    image_path = Path(args.image)
    
    if not image_path.exists():
        print(f"Error: File {args.image} not found.")
        return

    # Load image
    image = load_image(image_path).to(device)

    # Extract features
    print(f"\n=== Extracting SuperPoint features from {image_path.name} ===")
    try:
        features = extract_features(image, device, include_flipped=args.include_flipped, max_keypoints=args.max_keypoints)
        
        print(f"Extracted {features['num_keypoints']} keypoints")
        if args.include_flipped:
            print(f"Extracted {features['flipped']['num_keypoints']} keypoints from flipped image")
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return

    # Get versioning information
    versioning_info = get_versioning_info(__version__)
    
    # Prepare output data
    output_data = {
        "versioning": versioning_info,
        "image_path": str(image_path),
        "features": features
    }
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = image_path.parent / f"{image_path.stem}_features.json"
    
    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Features saved to {output_path}")
    print(f"  - Keypoints: {features['num_keypoints']}")
    print(f"  - Descriptors shape: {len(features['descriptors'])} x {len(features['descriptors'][0])}")
    if args.include_flipped:
        print(f"  - Flipped keypoints: {features['flipped']['num_keypoints']}")


if __name__ == "__main__":
    __main__()


__all__ = ["extract_features", "__version__", "__main__"]
