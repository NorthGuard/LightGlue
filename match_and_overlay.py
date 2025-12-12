import argparse
from pathlib import Path
import cv2
import torch
import numpy as np

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

def process_image_pair(extractor, matcher, image0, image1, device):
    """
    Process a pair of images and return match quality metrics.
    
    Returns:
        tuple: (num_inliers, homography_matrix, mkpts0, mkpts1)
    """
    # Match images
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]

    kpts0 = feats0["keypoints"]
    kpts1 = feats1["keypoints"]
    matches = matches01["matches"]
    
    # Get matched keypoints
    mkpts0 = kpts0[matches[..., 0]].cpu().numpy()
    mkpts1 = kpts1[matches[..., 1]].cpu().numpy()

    num_matches = len(mkpts0)
    
    if num_matches < 4:
        return 0, None, mkpts0, mkpts1

    # Compute Homography: prod (image1) -> orig (image0)
    M, mask = cv2.findHomography(mkpts1, mkpts0, cv2.RANSAC, 5.0)
    
    # Count inliers (mask contains 1 for inliers, 0 for outliers)
    num_inliers = int(np.sum(mask)) if mask is not None else 0

    return num_inliers, M, mkpts0, mkpts1

def main():
    parser = argparse.ArgumentParser(description='LightGlue Match and Overlay')
    parser.add_argument('orig', type=str, help='Path to original (background) image')
    parser.add_argument('prod', type=str, help='Path to production (foreground) image')
    parser.add_argument('--alpha_orig', type=float, default=0.5, help='Transparency for original image')
    parser.add_argument('--alpha_prod', type=float, default=0.5, help='Transparency for production image')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # Load extractor and matcher
    extractor = SuperPoint(max_num_keypoints=2048).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # Load images
    image0_path = Path(args.orig)
    image1_path = Path(args.prod)

    if not image0_path.exists():
        print(f"Error: File {args.orig} not found.")
        return
    if not image1_path.exists():
        print(f"Error: File {args.prod} not found.")
        return

    # Load original image as tensor
    image0 = load_image(image0_path).to(device)
    
    # Load production image as tensor
    image1 = load_image(image1_path).to(device)
    
    # Create flipped version (horizontal flip)
    # For tensor: flip along width dimension (dim=-1)
    image1_flipped = torch.flip(image1, dims=[-1])

    print("\n=== Testing Original Production Image ===")
    num_inliers_orig, M_orig, mkpts0_orig, mkpts1_orig = process_image_pair(
        extractor, matcher, image0, image1, device
    )
    print(f"Matches found: {len(mkpts0_orig)}")
    print(f"RANSAC inliers: {num_inliers_orig}")
    
    print("\n=== Testing Flipped Production Image ===")
    num_inliers_flipped, M_flipped, mkpts0_flipped, mkpts1_flipped = process_image_pair(
        extractor, matcher, image0, image1_flipped, device
    )
    print(f"Matches found: {len(mkpts0_flipped)}")
    print(f"RANSAC inliers: {num_inliers_flipped}")
    
    # Select best match
    print("\n=== Match Quality Comparison ===")
    if num_inliers_orig > num_inliers_flipped:
        print(f"✓ Best match: ORIGINAL (inliers: {num_inliers_orig} vs {num_inliers_flipped})")
        use_flipped = False
        M = M_orig
        num_inliers = num_inliers_orig
    else:
        print(f"✓ Best match: FLIPPED (inliers: {num_inliers_flipped} vs {num_inliers_orig})")
        use_flipped = True
        M = M_flipped
        num_inliers = num_inliers_flipped
    
    if num_inliers == 0:
        print("Error: Not enough matches to compute homography.")
        return

    print(f"Certainty: {num_inliers} inliers")
    
    print("\n=== Homographic Transformation Matrix ===")
    print(M)

    # Load images as numpy arrays for overlay
    orig_img = cv2.imread(str(image0_path))
    prod_img = cv2.imread(str(image1_path))
    
    # Flip if necessary
    if use_flipped:
        prod_img = cv2.flip(prod_img, 1)  # Horizontal flip

    h, w = orig_img.shape[:2]

    # Warp production image to match original image
    warped_prod = cv2.warpPerspective(prod_img, M, (w, h))

    # Create overlay
    overlay = cv2.addWeighted(orig_img, args.alpha_orig, warped_prod, args.alpha_prod, 0)

    output_path = "overlay.jpg"
    cv2.imwrite(output_path, overlay)
    print(f"\n✓ Overlay saved to {output_path}")

if __name__ == "__main__":
    main()
