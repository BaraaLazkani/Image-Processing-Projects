"""
Local Binary Patterns (LBP) - Texture-based Image Matching

This module implements Local Binary Patterns for texture feature extraction
and image matching using histogram comparison with Hamming distance metric.

LBP encodes local texture information by comparing each pixel with its 8
neighbors, creating a binary pattern that is rotation and illumination invariant.

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


class LBPMatcher:
    """Local Binary Patterns feature extractor and image matcher."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize LBP Matcher.

        Args:
            output_dir: Directory path for saving output visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    @staticmethod
    def _binary_to_decimal(binary_pattern: np.ndarray) -> int:
        """
        Convert binary array to decimal value.

        Args:
            binary_pattern: Binary array (0s and 1s)

        Returns:
            Decimal representation of binary pattern
        """
        decimal_value = 0
        pattern_length = len(binary_pattern)

        for i in range(pattern_length - 1, -1, -1):
            decimal_value += binary_pattern[pattern_length - 1 - i] * (2 ** i)

        return decimal_value

    def compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Compute Local Binary Pattern features for an image.

        For each pixel, compare it with its 8 neighbors in a 3x3 window.
        If neighbor >= center, set bit to 1; otherwise 0.
        Convert the 8-bit pattern to decimal value.

        Args:
            image: Input grayscale image

        Returns:
            LBP feature map (same size as input, minus 1-pixel border)
        """
        rows, cols = image.shape
        lbp_result = np.zeros((rows, cols), dtype=int)

        # Process each pixel (excluding border)
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                # Extract 8-neighbor pattern (clockwise from top-left)
                neighbor_pattern = np.zeros(8)
                neighbor_pattern[0] = image[i - 1, j - 1] > image[i, j]  # Top-left
                neighbor_pattern[1] = image[i - 1, j] > image[i, j]      # Top
                neighbor_pattern[2] = image[i - 1, j + 1] > image[i, j]  # Top-right
                neighbor_pattern[3] = image[i, j + 1] > image[i, j]      # Right
                neighbor_pattern[4] = image[i + 1, j + 1] > image[i, j]  # Bottom-right
                neighbor_pattern[5] = image[i + 1, j] > image[i, j]      # Bottom
                neighbor_pattern[6] = image[i + 1, j - 1] > image[i, j]  # Bottom-left
                neighbor_pattern[7] = image[i, j - 1] > image[i, j]      # Left

                # Convert binary pattern to decimal
                lbp_result[i - 1, j - 1] = self._binary_to_decimal(neighbor_pattern)

        return lbp_result

    @staticmethod
    def compute_histogram(lbp_image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Compute histogram of LBP feature values.

        Args:
            lbp_image: LBP feature map
            normalize: If True, normalize histogram by max value

        Returns:
            256-bin histogram of LBP values
        """
        rows, cols = lbp_image.shape
        histogram = np.zeros(256)

        # Count occurrences of each LBP value
        for i in range(rows):
            for j in range(cols):
                bin_index = int(lbp_image[i, j])
                histogram[bin_index] += 1

        # Normalize if requested
        if normalize:
            max_value = np.max(histogram)
            if max_value > 0:
                histogram = histogram / max_value

        return histogram

    @staticmethod
    def hamming_distance(hist1: np.ndarray, hist2: np.ndarray,
                        threshold: float = 0.01) -> float:
        """
        Compute Hamming-based similarity between two histograms.

        Calculates the percentage of histogram bins that differ by less
        than the threshold.

        Args:
            hist1: First histogram
            hist2: Second histogram
            threshold: Difference threshold for considering bins as matching

        Returns:
            Similarity percentage (0-100)
        """
        difference = np.abs(hist1 - hist2)
        matching_bins = np.sum(difference < threshold)
        similarity_percentage = (matching_bins * 100) / 256

        return similarity_percentage

    def process_images(self, image_paths: List[str], threshold: float = 0.01) -> None:
        """
        Process multiple images and compute LBP-based similarity matrix.

        Args:
            image_paths: List of paths to input images
            threshold: Hamming distance threshold for matching
        """
        num_images = len(image_paths)
        print(f"\n{'='*70}")
        print(f"Processing {num_images} images with LBP feature extraction")
        print(f"{'='*70}\n")

        # Load and process all images
        images = []
        lbp_features = []
        histograms = []

        for idx, img_path in enumerate(image_paths, 1):
            print(f"[{idx}/{num_images}] Processing {Path(img_path).name}...")

            # Load and convert to grayscale
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")

            image_resized = cv2.resize(image, (100, 100))
            image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

            # Compute LBP features
            lbp_result = self.compute_lbp(image_gray)
            histogram = self.compute_histogram(lbp_result, normalize=True)

            images.append(image_gray)
            lbp_features.append(lbp_result)
            histograms.append(histogram)

            print(f"    LBP shape: {lbp_result.shape}, Histogram size: {len(histogram)}")

        # Compute similarity matrix
        print(f"\n{'='*70}")
        print("Computing pairwise similarity matrix...")
        print(f"{'='*70}\n")

        print(f"Hamming distance threshold: {threshold}")
        print(f"Reference value: {25/256:.6f} ({25/256*100:.2f}%)\n")

        similarity_matrix = np.zeros((num_images, num_images))

        for i in range(num_images):
            for j in range(num_images):
                similarity_matrix[i, j] = self.hamming_distance(
                    histograms[i], histograms[j], threshold
                )

        # Print similarity matrix
        for i in range(num_images):
            print(f"Picture_{i+1}: ", end="")
            similarities = [f"{similarity_matrix[i, j]:.1f}" for j in range(num_images)]
            print(", ".join(similarities))

        # Visualize results
        self._visualize_lbp_features(images, lbp_features, image_paths)
        self._visualize_histograms(histograms, image_paths)
        self._visualize_similarity_matrix(similarity_matrix, image_paths)

        print(f"\n[OK] All outputs saved to {self.output_dir}/\n")

    def _visualize_lbp_features(self, images: List[np.ndarray],
                                lbp_features: List[np.ndarray],
                                image_paths: List[str]) -> None:
        """
        Create side-by-side visualization of original images and LBP features.

        Args:
            images: List of original grayscale images
            lbp_features: List of LBP feature maps
            image_paths: List of image paths for naming
        """
        num_images = len(images)
        fig, axes = plt.subplots(num_images, 3, figsize=(15, 4 * num_images))

        if num_images == 1:
            axes = axes.reshape(1, -1)

        for idx in range(num_images):
            img_name = Path(image_paths[idx]).stem

            # Original image
            axes[idx, 0].imshow(images[idx], cmap='gray')
            axes[idx, 0].set_title(f'{img_name} - Original')
            axes[idx, 0].axis('off')

            # LBP features
            axes[idx, 1].imshow(lbp_features[idx], cmap='jet')
            axes[idx, 1].set_title(f'{img_name} - LBP Features')
            axes[idx, 1].axis('off')

            # LBP histogram visualization
            axes[idx, 2].imshow(lbp_features[idx], cmap='gray')
            axes[idx, 2].set_title(f'{img_name} - LBP (Grayscale)')
            axes[idx, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'lbp_features_comparison.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

    def _visualize_histograms(self, histograms: List[np.ndarray],
                             image_paths: List[str]) -> None:
        """
        Create multi-panel histogram visualization.

        Args:
            histograms: List of LBP histograms
            image_paths: List of image paths for naming
        """
        num_images = len(histograms)
        fig, axes = plt.subplots(num_images, 1, figsize=(12, 3 * num_images))

        if num_images == 1:
            axes = [axes]

        for idx in range(num_images):
            img_name = Path(image_paths[idx]).stem

            axes[idx].plot(histograms[idx], color='blue', linewidth=2)
            axes[idx].set_title(f'LBP Histogram - {img_name}')
            axes[idx].set_xlabel('LBP Value (0-255)')
            axes[idx].set_ylabel('Normalized Frequency')
            axes[idx].set_xlim([0, 256])
            axes[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'lbp_histograms.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

    def _visualize_similarity_matrix(self, similarity_matrix: np.ndarray,
                                    image_paths: List[str]) -> None:
        """
        Create heatmap visualization of similarity matrix.

        Args:
            similarity_matrix: Pairwise similarity matrix
            image_paths: List of image paths for labeling
        """
        num_images = len(image_paths)
        image_names = [Path(p).stem for p in image_paths]

        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='YlGnBu', aspect='auto',
                      vmin=0, vmax=100)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity (%)', rotation=270, labelpad=20)

        # Set ticks and labels
        ax.set_xticks(np.arange(num_images))
        ax.set_yticks(np.arange(num_images))
        ax.set_xticklabels(image_names, rotation=45, ha='right')
        ax.set_yticklabels(image_names)

        # Add similarity values as text
        for i in range(num_images):
            for j in range(num_images):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)

        ax.set_title('LBP-based Image Similarity Matrix (%)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'similarity_matrix_heatmap.png', dpi=150,
                   bbox_inches='tight')
        plt.close()


def main():
    """Main entry point for LBP image matching."""
    parser = argparse.ArgumentParser(
        description='Local Binary Patterns - Texture-based Image Matching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python lbp_matcher.py --images I0.jpg I1.jpeg I2.jpg I3.jpg I4.jpg
  python lbp_matcher.py --images I0.jpg I1.jpeg --threshold 0.02
        """
    )

    parser.add_argument('--images', nargs='+', required=True,
                       help='Paths to input images for matching')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Hamming distance threshold (default: 0.01)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')

    args = parser.parse_args()

    # Initialize matcher
    matcher = LBPMatcher(output_dir=args.output_dir)

    print("\n" + "="*70)
    print("Local Binary Patterns - Image Matching System")
    print("="*70)

    try:
        # Process images and compute similarities
        matcher.process_images(args.images, args.threshold)

        print("="*70)
        print("[SUCCESS] LBP matching completed successfully!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
