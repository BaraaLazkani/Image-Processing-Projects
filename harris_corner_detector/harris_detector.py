"""
Harris Corner Detector - Interest Point Detection

This module implements the Harris Corner Detection algorithm for identifying
corner points in images. Corners are regions with high intensity variation
in multiple directions, making them useful features for object recognition
and image matching.

The Harris detector computes a corner response function R based on the
structure tensor (matrix M) of image gradients:
    R = det(M) - k * trace(M)^2
where k is typically 0.04-0.06.

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HarrisCornerDetector:
    """Harris Corner Detection implementation."""

    # Gaussian filter kernels
    GAUSSIAN_5X5 = np.array([
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]
    ])

    GAUSSIAN_13X13 = np.array([
        [0.0000, 0.0000, 0.0001, 0.0001, 0.0003, 0.0004, 0.0004, 0.0004, 0.0003, 0.0001, 0.0001, 0.0000, 0.0000],
        [0.0000, 0.0001, 0.0002, 0.0006, 0.0011, 0.0015, 0.0018, 0.0015, 0.0011, 0.0006, 0.0002, 0.0001, 0.0000],
        [0.0001, 0.0002, 0.0007, 0.0018, 0.0033, 0.0048, 0.0054, 0.0048, 0.0033, 0.0018, 0.0007, 0.0002, 0.0001],
        [0.0001, 0.0006, 0.0018, 0.0042, 0.0079, 0.0114, 0.0129, 0.0114, 0.0079, 0.0042, 0.0018, 0.0006, 0.0001],
        [0.0003, 0.0011, 0.0033, 0.0079, 0.0147, 0.0213, 0.0242, 0.0213, 0.0147, 0.0079, 0.0033, 0.0011, 0.0003],
        [0.0004, 0.0015, 0.0048, 0.0114, 0.0213, 0.0311, 0.0352, 0.0311, 0.0213, 0.0114, 0.0048, 0.0015, 0.0004],
        [0.0004, 0.0018, 0.0054, 0.0129, 0.0242, 0.0352, 0.0399, 0.0352, 0.0242, 0.0129, 0.0054, 0.0018, 0.0004],
        [0.0004, 0.0015, 0.0048, 0.0114, 0.0213, 0.0311, 0.0352, 0.0311, 0.0213, 0.0114, 0.0048, 0.0015, 0.0004],
        [0.0003, 0.0011, 0.0033, 0.0079, 0.0147, 0.0213, 0.0242, 0.0213, 0.0147, 0.0079, 0.0033, 0.0011, 0.0003],
        [0.0001, 0.0006, 0.0018, 0.0042, 0.0079, 0.0114, 0.0129, 0.0114, 0.0079, 0.0042, 0.0018, 0.0006, 0.0001],
        [0.0001, 0.0002, 0.0007, 0.0018, 0.0033, 0.0048, 0.0054, 0.0048, 0.0033, 0.0018, 0.0007, 0.0002, 0.0001],
        [0.0000, 0.0001, 0.0002, 0.0006, 0.0011, 0.0015, 0.0018, 0.0015, 0.0011, 0.0006, 0.0002, 0.0001, 0.0000],
        [0.0000, 0.0000, 0.0001, 0.0001, 0.0003, 0.0004, 0.0004, 0.0004, 0.0003, 0.0001, 0.0001, 0.0000, 0.0000]
    ])

    # Prewitt gradient kernels
    KERNEL_PREWITT_X = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])

    KERNEL_PREWITT_Y = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])

    def __init__(self, image_path: str, target_size: Tuple[int, int] = (200, 200),
                 k_value: float = 0.05, corner_threshold: float = 0.01,
                 output_dir: str = "output"):
        """
        Initialize Harris Corner Detector.

        Args:
            image_path: Path to input image
            target_size: Resize dimensions (width, height)
            k_value: Harris detector free parameter (typically 0.04-0.06)
            corner_threshold: Threshold for corner response (as fraction of max)
            output_dir: Directory for saving outputs
        """
        self.k_value = k_value
        self.corner_threshold = corner_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load and preprocess image
        self.image_color = self._load_image(image_path, target_size)
        self.image_gray = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2GRAY)

        # Storage for intermediate results
        self.smoothed = None
        self.gradient_x = None
        self.gradient_y = None
        self.M11 = None
        self.M12 = None
        self.M22 = None
        self.corner_response = None
        self.corners = None

    @staticmethod
    def _load_image(image_path: str, size: Tuple[int, int]) -> np.ndarray:
        """Load and resize image."""
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        return cv2.resize(image, size)

    @staticmethod
    def _convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Perform 2D convolution with zero-padding.

        Args:
            image: Input image
            kernel: Convolution kernel

        Returns:
            Convolved image
        """
        rows, cols = image.shape
        kernel_rows, kernel_cols = kernel.shape
        output = np.zeros((rows, cols))

        # Pad borders with original values
        for i in range(rows):
            for j in range(cols):
                output[i, j] = image[i, j]

        # Convolve interior region
        pad_r = kernel_rows // 2
        pad_c = kernel_cols // 2

        for i in range(pad_r, rows - pad_r - 1):
            for j in range(pad_c, cols - pad_c - 1):
                window = image[i - pad_r:i + pad_r + 1, j - pad_c:j + pad_c + 1]
                output[i, j] = np.sum(window * kernel)

        return output

    def compute_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image gradients using Prewitt operator.

        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        print("Computing image gradients...")

        # Apply Gaussian smoothing first
        self.smoothed = self._convolve(self.image_gray.astype(float), self.GAUSSIAN_5X5)

        # Compute gradients using Prewitt operator
        self.gradient_x = self._convolve(self.smoothed, self.KERNEL_PREWITT_X)
        self.gradient_y = self._convolve(self.smoothed, self.KERNEL_PREWITT_Y)

        print(f"  Gradient X range: [{np.min(self.gradient_x):.2f}, {np.max(self.gradient_x):.2f}]")
        print(f"  Gradient Y range: [{np.min(self.gradient_y):.2f}, {np.max(self.gradient_y):.2f}]")

        return self.gradient_x, self.gradient_y

    def compute_structure_tensor(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute structure tensor (M matrix) from gradients.

        The structure tensor is a 2x2 matrix at each pixel:
            M = [[M11, M12],
                 [M12, M22]]
        where M11 = sum(Ix^2), M12 = sum(Ix*Iy), M22 = sum(Iy^2)
        over a local window (via Gaussian weighting).

        Returns:
            Tuple of (M11, M12, M22)
        """
        print("Computing structure tensor...")

        # Compute products of derivatives
        Ixx = self.gradient_x * self.gradient_x
        Iyy = self.gradient_y * self.gradient_y
        Ixy = self.gradient_x * self.gradient_y

        # Apply Gaussian weighting (sum over local window)
        self.M11 = self._convolve(Ixx, self.GAUSSIAN_13X13)
        self.M12 = self._convolve(Ixy, self.GAUSSIAN_13X13)
        self.M22 = self._convolve(Iyy, self.GAUSSIAN_13X13)

        print(f"  M11 range: [{np.min(self.M11):.2f}, {np.max(self.M11):.2f}]")
        print(f"  M12 range: [{np.min(self.M12):.2f}, {np.max(self.M12):.2f}]")
        print(f"  M22 range: [{np.min(self.M22):.2f}, {np.max(self.M22):.2f}]")

        return self.M11, self.M12, self.M22

    def compute_corner_response(self) -> np.ndarray:
        """
        Compute Harris corner response function.

        R = det(M) - k * trace(M)^2
        where det(M) = M11*M22 - M12^2 and trace(M) = M11 + M22

        Positive R indicates corners, negative indicates edges,
        small absolute values indicate flat regions.

        Returns:
            Corner response map
        """
        print(f"Computing Harris corner response (k={self.k_value})...")

        # Compute determinant and trace
        det_M = self.M11 * self.M22 - self.M12 * self.M12
        trace_M = self.M11 + self.M22

        # Harris corner response
        self.corner_response = det_M - self.k_value * (trace_M ** 2)

        print(f"  Response range: [{np.min(self.corner_response):.2f}, "
              f"{np.max(self.corner_response):.2f}]")

        num_positive = np.sum(self.corner_response > 0)
        print(f"  Positive responses: {num_positive}/{self.corner_response.size} pixels")

        return self.corner_response

    def detect_corners(self) -> np.ndarray:
        """
        Detect corners by thresholding corner response.

        Args:
            None (uses self.corner_threshold)

        Returns:
            Binary corner map (1 = corner, 0 = non-corner)
        """
        print(f"Detecting corners (threshold={self.corner_threshold})...")

        # Normalize response to [0, 1]
        response_max = np.max(self.corner_response)
        response_normalized = self.corner_response / response_max if response_max > 0 else self.corner_response

        # Threshold to detect corners
        self.corners = (response_normalized > self.corner_threshold).astype(np.uint8)

        num_corners = np.sum(self.corners)
        print(f"  Detected {num_corners} corner pixels")

        return self.corners

    def visualize_and_save(self, image_name: str = "result") -> None:
        """
        Create comprehensive visualization of Harris corner detection pipeline.

        Args:
            image_name: Base name for output files
        """
        print(f"\nCreating visualizations...")

        # Create multi-panel visualization of pipeline stages
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Row 1: Original, Gradients
        axes[0, 0].imshow(self.image_gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(np.abs(self.gradient_x), cmap='gray')
        axes[0, 1].set_title('Gradient X (Prewitt)')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.abs(self.gradient_y), cmap='gray')
        axes[0, 2].set_title('Gradient Y (Prewitt)')
        axes[0, 2].axis('off')

        # Row 2: Structure tensor elements
        axes[1, 0].imshow(self.M11, cmap='jet')
        axes[1, 0].set_title('Structure Tensor M11 (Ix^2)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.M12, cmap='jet')
        axes[1, 1].set_title('Structure Tensor M12 (Ix*Iy)')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(self.M22, cmap='jet')
        axes[1, 2].set_title('Structure Tensor M22 (Iy^2)')
        axes[1, 2].axis('off')

        # Row 3: Corner response and detected corners
        axes[2, 0].imshow(self.corner_response, cmap='jet')
        axes[2, 0].set_title(f'Harris Corner Response (k={self.k_value})')
        axes[2, 0].axis('off')

        axes[2, 1].imshow(self.corners, cmap='gray')
        axes[2, 1].set_title(f'Detected Corners (threshold={self.corner_threshold})')
        axes[2, 1].axis('off')

        # Overlay corners on original image
        image_with_corners = self.image_color.copy()
        corner_coords = np.where(self.corners == 1)
        for y, x in zip(corner_coords[0], corner_coords[1]):
            cv2.circle(image_with_corners, (x, y), 3, (0, 0, 255), -1)

        axes[2, 2].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
        axes[2, 2].set_title(f'Corners Overlaid ({np.sum(self.corners)} corners)')
        axes[2, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_harris_pipeline.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

        # Save individual outputs
        cv2.imwrite(str(self.output_dir / f'{image_name}_gradient_x.png'),
                   np.abs(self.gradient_x).astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_gradient_y.png'),
                   np.abs(self.gradient_y).astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_corners_overlay.png'),
                   image_with_corners)

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(self.image_gray, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Harris Corners ({np.sum(self.corners)} corners)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_comparison.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

        print(f"[OK] All outputs saved to {self.output_dir}/")

    def run(self) -> np.ndarray:
        """
        Execute full Harris corner detection pipeline.

        Returns:
            Binary corner map
        """
        print("\n" + "="*70)
        print("Harris Corner Detection Pipeline")
        print("="*70 + "\n")

        self.compute_gradients()
        self.compute_structure_tensor()
        self.compute_corner_response()
        self.detect_corners()

        return self.corners


def main():
    """Main entry point for Harris corner detection."""
    parser = argparse.ArgumentParser(
        description='Harris Corner Detector - Interest Point Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python harris_detector.py --image board.jpg
  python harris_detector.py --image board.jpg --k 0.04 --threshold 0.02
  python harris_detector.py --image board.jpg --size 300 300
        """
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--size', type=int, nargs=2, default=[200, 200],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Resize image to WIDTH HEIGHT (default: 200 200)')
    parser.add_argument('--k', type=float, default=0.05,
                       help='Harris detector k parameter (default: 0.05)')
    parser.add_argument('--threshold', type=float, default=0.01,
                       help='Corner detection threshold (default: 0.01)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = HarrisCornerDetector(
            image_path=args.image,
            target_size=tuple(args.size),
            k_value=args.k,
            corner_threshold=args.threshold,
            output_dir=args.output_dir
        )

        # Run detection
        corners = detector.run()

        # Visualize and save results
        image_name = Path(args.image).stem
        detector.visualize_and_save(image_name)

        print("\n" + "="*70)
        print(f"[SUCCESS] Detected {np.sum(corners)} corners!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
