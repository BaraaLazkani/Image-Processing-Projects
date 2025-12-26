"""
OpenCV Basics - Fundamental Image Processing Operations

This module demonstrates basic OpenCV operations including image loading,
color conversions, thresholding, histogram analysis, and gamma correction.
Consolidates exercises 1-3 from introductory OpenCV coursework.

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import cv2
import matplotlib.pyplot as plt


class OpenCVBasics:
    """Class implementing fundamental OpenCV image processing operations."""

    def __init__(self, output_dir: str = "output"):
        """
        Initialize OpenCV Basics processor.

        Args:
            output_dir: Directory path for saving output images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def run_exercise_1(self, image_path: str) -> None:
        """
        Exercise 1: Basic image loading and display operations.

        Demonstrates:
        - Loading images in grayscale and color modes
        - Basic image display and file I/O

        Args:
            image_path: Path to input image file
        """
        print(f"\n{'='*60}")
        print("Exercise 1: Basic Image Loading and Display")
        print(f"{'='*60}")

        # Load image in grayscale mode
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image in color mode (BGR)
        image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

        print(f"Loaded image: {image_path}")
        print(f"Grayscale shape: {image_gray.shape}")
        print(f"Color shape: {image_color.shape}")

        # Save outputs
        output_base = Path(image_path).stem
        cv2.imwrite(str(self.output_dir / f"{output_base}_ex1_grayscale.png"), image_gray)
        cv2.imwrite(str(self.output_dir / f"{output_base}_ex1_color.png"), image_color)

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(image_gray, cmap='gray')
        axes[0].set_title('Grayscale Mode')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Color Mode (RGB)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_base}_ex1_comparison.png", dpi=150)
        plt.close()

        print(f"[OK] Outputs saved to {self.output_dir}/")

    def run_exercise_2(self, image_path: str, threshold_value: int = 127,
                      gamma: float = 0.5) -> None:
        """
        Exercise 2: Color conversion, thresholding, and gamma correction.

        Demonstrates:
        - BGR to grayscale conversion
        - Binary thresholding
        - Gamma correction (power-law transformation)

        Args:
            image_path: Path to input image file
            threshold_value: Binary threshold value (0-255)
            gamma: Gamma correction exponent
        """
        print(f"\n{'='*60}")
        print("Exercise 2: Color Conversion and Transformations")
        print(f"{'='*60}")

        # Load color image
        image_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_color is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Convert to grayscale
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        print(f"Converted {image_color.shape} color image to {image_gray.shape} grayscale")

        # Apply binary thresholding
        _, image_binary = cv2.threshold(image_gray, threshold_value, 255,
                                        cv2.THRESH_BINARY)
        print(f"Applied binary threshold at value {threshold_value}")

        # Apply gamma correction (power-law transformation)
        # Normalize to [0, 1], apply gamma, then scale back
        image_normalized = image_gray.astype(np.float64) / 255.0
        image_gamma_corrected = np.power(image_normalized, gamma)

        print(f"Applied gamma correction (gamma={gamma})")
        print(f"  Gamma-corrected range: [{np.min(image_gamma_corrected):.3f}, "
              f"{np.max(image_gamma_corrected):.3f}]")

        # Save individual outputs
        output_base = Path(image_path).stem
        cv2.imwrite(str(self.output_dir / f"{output_base}_ex2_grayscale.png"),
                   image_gray)
        cv2.imwrite(str(self.output_dir / f"{output_base}_ex2_binary.png"),
                   image_binary)

        # Save gamma-corrected (scale back to 0-255 for saving)
        image_gamma_uint8 = (image_gamma_corrected * 255).astype(np.uint8)
        cv2.imwrite(str(self.output_dir / f"{output_base}_ex2_gamma.png"),
                   image_gamma_uint8)

        # Create multi-panel visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Color Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(image_gray, cmap='gray')
        axes[0, 1].set_title('Grayscale Conversion')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(image_binary, cmap='gray')
        axes[1, 0].set_title(f'Binary Threshold (T={threshold_value})')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(image_gamma_corrected, cmap='gray')
        axes[1, 1].set_title(f'Gamma Correction (gamma={gamma})')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{output_base}_ex2_comparison.png", dpi=150)
        plt.close()

        print(f"[OK] Outputs saved to {self.output_dir}/")

    def run_exercise_3(self, image_path: str) -> None:
        """
        Exercise 3: Histogram computation and visualization.

        Demonstrates:
        - Computing image histograms
        - Visualizing intensity distributions
        - Understanding image statistics

        Args:
            image_path: Path to input image file
        """
        print(f"\n{'='*60}")
        print("Exercise 3: Histogram Analysis")
        print(f"{'='*60}")

        # Load image in grayscale
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image_gray is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Compute histogram using OpenCV
        histogram = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

        # Compute basic statistics
        mean_intensity = np.mean(image_gray)
        std_intensity = np.std(image_gray)
        min_intensity = np.min(image_gray)
        max_intensity = np.max(image_gray)

        print(f"Image statistics:")
        print(f"  Mean intensity: {mean_intensity:.2f}")
        print(f"  Std deviation: {std_intensity:.2f}")
        print(f"  Min intensity: {min_intensity}")
        print(f"  Max intensity: {max_intensity}")

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        axes[0, 0].imshow(image_gray, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Histogram
        axes[0, 1].plot(histogram, color='black', linewidth=2)
        axes[0, 1].set_title('Histogram of Original Image')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_xlim([0, 256])
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram equalization
        image_equalized = cv2.equalizeHist(image_gray)
        axes[1, 0].imshow(image_equalized, cmap='gray')
        axes[1, 0].set_title('Histogram Equalized Image')
        axes[1, 0].axis('off')

        # Equalized histogram
        histogram_eq = cv2.calcHist([image_equalized], [0], None, [256], [0, 256])
        axes[1, 1].plot(histogram_eq, color='blue', linewidth=2)
        axes[1, 1].set_title('Histogram After Equalization')
        axes[1, 1].set_xlabel('Pixel Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_xlim([0, 256])
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save outputs
        output_base = Path(image_path).stem
        plt.savefig(self.output_dir / f"{output_base}_ex3_histogram_analysis.png", dpi=150)
        plt.close()

        cv2.imwrite(str(self.output_dir / f"{output_base}_ex3_equalized.png"),
                   image_equalized)

        # Save histogram data to text file
        with open(self.output_dir / f"{output_base}_ex3_statistics.txt", 'w') as f:
            f.write(f"Image Statistics for {image_path}\n")
            f.write(f"{'='*50}\n")
            f.write(f"Mean intensity: {mean_intensity:.2f}\n")
            f.write(f"Standard deviation: {std_intensity:.2f}\n")
            f.write(f"Min intensity: {min_intensity}\n")
            f.write(f"Max intensity: {max_intensity}\n")
            f.write(f"Image shape: {image_gray.shape}\n")

        print(f"[OK] Outputs saved to {self.output_dir}/")


def main():
    """Main entry point for OpenCV Basics exercises."""
    parser = argparse.ArgumentParser(
        description='OpenCV Basics - Fundamental Image Processing Operations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python opencv_basics.py --exercise 1 --image picka.png
  python opencv_basics.py --exercise 2 --image picka.png --threshold 100 --gamma 0.7
  python opencv_basics.py --exercise 3 --image rose.jpg
  python opencv_basics.py --exercise all --image picka.png
        """
    )

    parser.add_argument('--exercise', type=str, choices=['1', '2', '3', 'all'],
                       default='all',
                       help='Exercise number to run (1, 2, 3, or all)')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image file')
    parser.add_argument('--threshold', type=int, default=127,
                       help='Binary threshold value for Exercise 2 (default: 127)')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Gamma correction value for Exercise 2 (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results (default: output)')

    args = parser.parse_args()

    # Initialize processor
    processor = OpenCVBasics(output_dir=args.output_dir)

    print("\n" + "="*60)
    print("OpenCV Basics - Image Processing Exercises")
    print("="*60)

    try:
        # Run selected exercises
        if args.exercise in ['1', 'all']:
            processor.run_exercise_1(args.image)

        if args.exercise in ['2', 'all']:
            processor.run_exercise_2(args.image, args.threshold, args.gamma)

        if args.exercise in ['3', 'all']:
            processor.run_exercise_3(args.image)

        print(f"\n{'='*60}")
        print("[SUCCESS] All exercises completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
