"""
SIFT Feature Matching - Scale-Invariant Feature Transform

This module implements the SIFT (Scale-Invariant Feature Transform) algorithm
for detecting and matching keypoints across images with different scales,
rotations, and illuminations.

SIFT Pipeline:
1. Scale-space construction - Build Gaussian pyramid
2. DoG keypoint detection - Find extrema in Difference of Gaussians
3. Keypoint localization - Threshold and refine detections
4. Orientation assignment - Compute dominant orientations via HOG
5. Descriptor generation - Create 128-dimensional feature vectors
6. Feature matching - Match descriptors using Euclidean distance

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SIFTFeatureMatcher:
    """SIFT feature detector and matcher."""

    # Sobel kernels for gradient computation
    SOBEL_X = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    SOBEL_Y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    def __init__(self, output_dir: str = "output"):
        """
        Initialize SIFT Feature Matcher.

        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def build_gaussian_pyramid(self, image: np.ndarray,
                               sigmas: List[float]) -> List[np.ndarray]:
        """
        Build Gaussian pyramid (octave) with multiple scales.

        Args:
            image: Input grayscale image
            sigmas: List of sigma values for Gaussian blurring

        Returns:
            List of Gaussian-blurred images at different scales
        """
        pyramid = []
        for sigma in sigmas:
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
            pyramid.append(blurred)

        return pyramid

    def compute_dog(self, gaussian_pyramid: List[np.ndarray]) -> np.ndarray:
        """
        Compute Difference of Gaussians (DoG) from Gaussian pyramid.

        Args:
            gaussian_pyramid: List of Gaussian-blurred images

        Returns:
            3D array of DoG responses [scale, height, width]
        """
        dog_responses = []
        for i in range(len(gaussian_pyramid) - 1):
            dog = gaussian_pyramid[i + 1] - gaussian_pyramid[i]
            dog_responses.append(dog)

        return np.array(dog_responses)

    def detect_extrema(self, dog_pyramid: np.ndarray, dim: int = 3) -> np.ndarray:
        """
        Detect local extrema (maxima and minima) in DoG pyramid.

        Args:
            dog_pyramid: 3D DoG pyramid [scale, height, width]
            dim: Size of search window

        Returns:
            Binary map indicating extrema locations
        """
        rows, cols = dog_pyramid[0].shape
        extrema_map = np.zeros((rows, cols))

        half_dim = dim // 2

        # Only check middle scale (index 1) against its neighbors
        for i in range(half_dim, rows - half_dim):
            for j in range(half_dim, cols - half_dim):
                # Extract 3D neighborhood (across scales and spatial)
                neighborhood = dog_pyramid[:,
                                          i - half_dim:i + half_dim + 1,
                                          j - half_dim:j + half_dim + 1]

                center_val = dog_pyramid[1, i, j]
                max_val = np.max(neighborhood)
                min_val = np.min(neighborhood)

                # Mark if local maximum or minimum
                if (center_val == max_val) or (center_val == min_val):
                    extrema_map[i, j] = 1

        return extrema_map

    def threshold_keypoints(self, keypoint_map: np.ndarray,
                           dog_response: np.ndarray,
                           threshold: float) -> np.ndarray:
        """
        Threshold keypoints based on DoG response strength.

        Args:
            keypoint_map: Binary extrema map
            dog_response: DoG response values
            threshold: Threshold as fraction of max response

        Returns:
            Thresholded keypoint map
        """
        # Take absolute value and normalize
        abs_response = np.abs(dog_response)
        max_resp = np.max(abs_response)

        if max_resp > 0:
            normalized = abs_response / max_resp
        else:
            normalized = abs_response

        # Apply threshold
        thresholded = np.zeros_like(keypoint_map)
        strong_enough = normalized >= threshold
        thresholded[strong_enough & (keypoint_map == 1)] = 1

        return thresholded

    def compute_orientation_histogram(self, patch: np.ndarray, flag: int = 1) -> np.ndarray:
        """
        Compute histogram of orientations (8 bins).

        Args:
            patch: Orientation patch (values 0, 45, 90, ..., 315)
            flag: 1 to normalize, 0 otherwise

        Returns:
            8-bin histogram
        """
        hist = np.zeros(8)
        rows, cols = patch.shape

        for i in range(rows):
            for j in range(cols):
                bin_idx = int(patch[i, j] / 45)
                if 0 <= bin_idx < 8:
                    hist[bin_idx] += 1

        if flag == 1 and np.max(hist) > 0:
            hist = hist / np.max(hist)

        return hist

    def quantize_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Quantize angles to 8 directions (0, 45, 90, ..., 315).

        Args:
            angles: Angle map in degrees

        Returns:
            Quantized angle map
        """
        quantized = np.zeros_like(angles)
        rows, cols = angles.shape

        for i in range(rows):
            for j in range(cols):
                angle = angles[i, j]

                # Shift negative to positive
                if angle < 0 and angle >= -180:
                    angle += 180

                # Quantize
                if 0 <= angle < 22.5 or 337.5 <= angle <= 360:
                    quantized[i, j] = 0
                elif 22.5 <= angle < 67.5:
                    quantized[i, j] = 45
                elif 67.5 <= angle < 112.5:
                    quantized[i, j] = 90
                elif 112.5 <= angle < 157.5:
                    quantized[i, j] = 135
                elif 157.5 <= angle < 202.5:
                    quantized[i, j] = 180
                elif 202.5 <= angle < 247.5:
                    quantized[i, j] = 225
                elif 247.5 <= angle < 292.5:
                    quantized[i, j] = 270
                elif 292.5 <= angle < 337.5:
                    quantized[i, j] = 315

        return quantized

    def rotate_image(self, img: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate a 17x17 image patch.

        Args:
            img: Input patch
            angle: Rotation angle in degrees

        Returns:
            Rotated 17x17 patch
        """
        result = np.zeros((17, 17))
        center = 8

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                x_old = i - center
                y_old = j - center

                rad = np.radians(angle)
                x_new = round(np.cos(rad) * x_old - np.sin(rad) * y_old)
                y_new = round(np.sin(rad) * x_old + np.cos(rad) * y_old)

                if -8 <= x_new <= 8 and -8 <= y_new <= 8:
                    result[x_new + center][y_new + center] = img[i, j]

        return result

    def extract_descriptor(self, keypoint: Tuple[int, int],
                          orientations: np.ndarray,
                          image: np.ndarray) -> Tuple[int, int, np.ndarray]:
        """
        Extract 128-dimensional SIFT descriptor for keypoint.

        Args:
            keypoint: (row, col) of keypoint
            orientations: Quantized orientation map
            image: Original image

        Returns:
            (row, col, 128-dim descriptor vector)
        """
        i, j = keypoint

        # Extract 17x17 window
        window = orientations[i-8:i+9, j-8:j+9]

        # Find dominant orientation
        hist = self.compute_orientation_histogram(window, flag=1)
        dominant_idx = np.argmax(hist)
        dominant_angle = dominant_idx * 45

        # Extract and rotate image patch
        img_patch = image[i-8:i+9, j-8:j+9]
        rotated = self.rotate_image(img_patch, -dominant_angle)

        # Compute gradients on rotated patch
        Ix = cv2.filter2D(rotated, -1, self.SOBEL_X)
        Iy = cv2.filter2D(rotated, -1, self.SOBEL_Y)
        directions = np.arctan2(Iy, Ix) * 180 / np.pi
        quantized_dirs = self.quantize_angles(directions)

        # Build 128-dim descriptor (4x4 grid, 8 bins each)
        descriptor = np.zeros(128)
        for grid_i in range(4):
            for grid_j in range(4):
                cell = quantized_dirs[grid_i*4:grid_i*4+4, grid_j*4:grid_j*4+4]
                cell_hist = self.compute_orientation_histogram(cell, flag=0)
                start_idx = grid_i * 32 + grid_j * 8
                descriptor[start_idx:start_idx+8] = cell_hist

        return (i, j, descriptor)

    def process_image_octaves(self, image_path: str) -> List[Tuple]:
        """
        Process image through multi-scale SIFT pipeline.

        Args:
            image_path: Path to image

        Returns:
            List of descriptors [(row, col, 128-dim vector), ...]
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Multi-scale processing
        sigmas = [2, np.sqrt(2)*2, 4, np.sqrt(2)*4]
        scales = [(400, 400), (200, 200), (100, 100), (50, 50)]
        thresholds = [0.5, 0.5, 0.5, 0.5]

        all_descriptors = []

        for scale_idx, size in enumerate(scales):
            img_resized = cv2.resize(image, size)
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

            # Build pyramid and compute DoG
            pyramid = self.build_gaussian_pyramid(img_gray, sigmas)
            dog = self.compute_dog(pyramid)

            # Detect keypoints
            extrema = self.detect_extrema(dog, dim=3)
            keypoints = self.threshold_keypoints(extrema, dog[1], thresholds[scale_idx])

            # Extract descriptors
            keypoint_locs = np.where(keypoints == 1)

            # Compute orientations for descriptor extraction
            Ix = cv2.filter2D(img_gray, -1, self.SOBEL_X)
            Iy = cv2.filter2D(img_gray, -1, self.SOBEL_Y)
            directions = np.arctan2(Iy, Ix) * 180 / np.pi
            quantized_dirs = self.quantize_angles(directions)

            for row, col in zip(keypoint_locs[0], keypoint_locs[1]):
                # Skip if too close to border (need 17x17 patch)
                if row < 8 or row >= img_gray.shape[0] - 8:
                    continue
                if col < 8 or col >= img_gray.shape[1] - 8:
                    continue

                descriptor = self.extract_descriptor((row, col), quantized_dirs, img_gray)

                # Scale coordinates back to 400x400 reference
                scale_factor = 400 / size[0]
                scaled_row = int(descriptor[0] * scale_factor)
                scaled_col = int(descriptor[1] * scale_factor)

                all_descriptors.append((scaled_row, scaled_col, descriptor[2]))

        return all_descriptors

    def match_features(self, desc1: List[Tuple], desc2: List[Tuple],
                      threshold: float = 21) -> List[List]:
        """
        Match features using Euclidean distance.

        Args:
            desc1: Descriptors from image 1
            desc2: Descriptors from image 2
            threshold: Distance threshold for matching

        Returns:
            List of matches [[(r1, c1), (r2, c2)], ...]
        """
        matches = []

        for d1 in desc1:
            for d2 in desc2:
                # Euclidean distance between descriptors
                dist = 0
                for k in range(128):
                    dist += (d1[2][k] - d2[2][k]) ** 2
                dist = np.sqrt(dist)

                if dist <= threshold:
                    matches.append([(d1[0], d1[1]), (d2[0], d2[1])])

        return matches

    def visualize_matches(self, img1_path: str, img2_path: str,
                         matches: List) -> np.ndarray:
        """
        Create visualization of matched features.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            matches: List of matched keypoint pairs

        Returns:
            Concatenated image with matches visualized
        """
        img1 = cv2.resize(cv2.imread(img1_path), (400, 400))
        img2 = cv2.resize(cv2.imread(img2_path), (400, 400))

        # Concatenate side by side
        result = np.concatenate((img1, img2), axis=1)

        # Draw matches
        radius = 5
        color = (0, 0, 255)
        thickness = 2

        for match in matches:
            pt1 = (match[0][1], match[0][0])  # (col, row)
            pt2 = (match[1][1] + 400, match[1][0])  # Offset by image width

            cv2.circle(result, pt1, radius, color, thickness)
            cv2.circle(result, pt2, radius, color, thickness)
            cv2.line(result, pt1, pt2, color, thickness)

        return result

    def run_matching(self, img1_path: str, img2_path: str,
                    threshold: float = 21) -> None:
        """
        Run complete SIFT matching pipeline.

        Args:
            img1_path: Path to first image
            img2_path: Path to second image
            threshold: Matching distance threshold
        """
        print("\n" + "="*70)
        print("SIFT Feature Detection and Matching")
        print("="*70 + "\n")

        # Process both images
        print(f"Processing {Path(img1_path).name}...")
        desc1 = self.process_image_octaves(img1_path)
        print(f"  Detected {len(desc1)} keypoints")

        print(f"\nProcessing {Path(img2_path).name}...")
        desc2 = self.process_image_octaves(img2_path)
        print(f"  Detected {len(desc2)} keypoints")

        # Match features
        print(f"\nMatching features (threshold={threshold})...")
        matches = self.match_features(desc1, desc2, threshold)
        print(f"  Found {len(matches)} matches")

        # Visualize
        print("\nCreating visualizations...")
        result_img = self.visualize_matches(img1_path, img2_path, matches)

        # Save outputs
        img1_name = Path(img1_path).stem
        img2_name = Path(img2_path).stem

        output_name = f"{img1_name}_{img2_name}_matches.png"
        cv2.imwrite(str(self.output_dir / output_name), result_img)

        # Create matplotlib figure
        plt.figure(figsize=(16, 8))
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f'SIFT Feature Matches: {len(matches)} correspondences')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{img1_name}_{img2_name}_visualization.png",
                   dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[OK] Outputs saved to {self.output_dir}/")


def main():
    """Main entry point for SIFT matching."""
    parser = argparse.ArgumentParser(
        description='SIFT Feature Detection and Matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python sift_detector.py --image1 I8.jpg --image2 I9.jpg
  python sift_detector.py --image1 I8.jpg --image2 I9.jpg --threshold 25
        """
    )

    parser.add_argument('--image1', type=str, required=True,
                       help='Path to first image')
    parser.add_argument('--image2', type=str, required=True,
                       help='Path to second image')
    parser.add_argument('--threshold', type=float, default=21,
                       help='Matching distance threshold (default: 21)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')

    args = parser.parse_args()

    try:
        matcher = SIFTFeatureMatcher(output_dir=args.output_dir)
        matcher.run_matching(args.image1, args.image2, args.threshold)

        print("\n" + "="*70)
        print("[SUCCESS] SIFT matching completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
