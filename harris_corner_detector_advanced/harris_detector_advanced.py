"""
Advanced Harris Corner Detector - Enhanced Interest Point Detection

This module implements an enhanced version of the Harris Corner Detector
with Non-Maximum Suppression (NMS) for cleaner corner detection results.

Key differences from basic Harris detector:
1. Uses alternative corner response formula: R = det(M) / trace(M)
   (instead of standard R = det(M) - k*trace(M)^2)
2. Applies Non-Maximum Suppression to eliminate clustered detections
3. Provides enhanced visualization with circle overlays

The alternative formulation is invariant to image scale and doesn't
require tuning the k parameter.

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import Tuple, List
import numpy as np
import cv2
import matplotlib.pyplot as plt


class AdvancedHarrisDetector:
    """Advanced Harris Corner Detector with Non-Maximum Suppression."""

    # Gaussian filter (5x5, sigma=1)
    GAUSSIAN_5X5 = np.array([
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]
    ])

    # Gaussian filter (13x13, sigma=2)
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
    KERNEL_PREWITT_Y = np.array([[-1, -1, -1],
                                  [0, 0, 0],
                                  [1, 1, 1]])

    KERNEL_PREWITT_X = np.array([[-1, 0, 1],
                                  [-1, 0, 1],
                                  [-1, 0, 1]])

    def __init__(self, image_path: str, target_size: Tuple[int, int] = (200, 200),
                 corner_threshold: float = 0.25, nms_window_size: int = 15,
                 output_dir: str = "output"):
        """
        Initialize Advanced Harris Corner Detector.

        Args:
            image_path: Path to input image
            target_size: Resize dimensions (width, height)
            corner_threshold: Threshold for corner response (as fraction of max)
            nms_window_size: Window size for non-maximum suppression
            output_dir: Directory for saving outputs
        """
        self.corner_threshold = corner_threshold
        self.nms_window_size = nms_window_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load and preprocess image
        self.image_color = self._load_image(image_path, target_size)
        self.image_gray = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Storage for intermediate results
        self.smoothed = None
        self.gradient_x = None
        self.gradient_y = None
        self.M11 = None
        self.M12 = None
        self.M22 = None
        self.corner_response = None
        self.response_after_nms = None
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
        Perform 2D convolution.

        Args:
            image: Input image
            kernel: Convolution kernel

        Returns:
            Convolved image
        """
        rows, cols = image.shape
        kernel_rows, kernel_cols = kernel.shape
        output = np.zeros((rows, cols), dtype=np.float32)

        # Initialize with original values
        for i in range(rows):
            for j in range(cols):
                output[i, j] = image[i, j]

        # Convolve interior
        pad_r = kernel_rows // 2
        pad_c = kernel_cols // 2

        for i in range(pad_r, rows - pad_r):
            for j in range(pad_c, cols - pad_c):
                window = image[i - pad_r:i + pad_r + 1, j - pad_c:j + pad_c + 1]
                output[i, j] = np.sum(window * kernel)

        return output

    @staticmethod
    def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        """
        Safely divide two arrays, handling division by zero.

        Args:
            numerator: Numerator array
            denominator: Denominator array

        Returns:
            Result of division, with zeros where denominator is zero
        """
        result = np.zeros_like(numerator)
        for i in range(numerator.shape[0]):
            for j in range(numerator.shape[1]):
                if denominator[i, j] != 0:
                    result[i, j] = numerator[i, j] / denominator[i, j]
        return result

    def compute_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute image gradients using Prewitt operator.

        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        print("Computing image gradients...")

        # Apply Gaussian smoothing
        self.smoothed = self._convolve(self.image_gray, self.GAUSSIAN_5X5)

        # Compute gradients
        self.gradient_x = self._convolve(self.smoothed, self.KERNEL_PREWITT_X)
        self.gradient_y = self._convolve(self.smoothed, self.KERNEL_PREWITT_Y)

        print(f"  Gradient X range: [{np.min(self.gradient_x):.2f}, {np.max(self.gradient_x):.2f}]")
        print(f"  Gradient Y range: [{np.min(self.gradient_y):.2f}, {np.max(self.gradient_y):.2f}]")

        return self.gradient_x, self.gradient_y

    def compute_structure_tensor(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute structure tensor from gradients.

        Returns:
            Tuple of (M11, M12, M22)
        """
        print("Computing structure tensor...")

        # Compute products of derivatives
        Ixx = self.gradient_x * self.gradient_x
        Iyy = self.gradient_y * self.gradient_y
        Ixy = self.gradient_x * self.gradient_y

        # Apply Gaussian weighting
        self.M11 = self._convolve(Ixx, self.GAUSSIAN_13X13)
        self.M12 = self._convolve(Ixy, self.GAUSSIAN_13X13)
        self.M22 = self._convolve(Iyy, self.GAUSSIAN_13X13)

        print(f"  M11 range: [{np.min(self.M11):.2f}, {np.max(self.M11):.2f}]")
        print(f"  M12 range: [{np.min(self.M12):.2f}, {np.max(self.M12):.2f}]")
        print(f"  M22 range: [{np.min(self.M22):.2f}, {np.max(self.M22):.2f}]")

        return self.M11, self.M12, self.M22

    def compute_corner_response(self) -> np.ndarray:
        """
        Compute corner response using alternative formulation.

        Uses R = det(M) / trace(M) instead of standard Harris formula.
        This formulation is scale-invariant and doesn't require k parameter.

        Returns:
            Corner response map
        """
        print("Computing corner response (det(M)/trace(M) formulation)...")

        # Compute determinant and trace
        det_M = self.M11 * self.M22 - self.M12 * self.M12
        trace_M = self.M11 + self.M22

        # Alternative corner response: det(M) / trace(M)
        self.corner_response = self._safe_divide(det_M, trace_M)

        print(f"  Response range: [{np.min(self.corner_response):.2f}, "
              f"{np.max(self.corner_response):.2f}]")

        return self.corner_response

    def apply_nms(self) -> np.ndarray:
        """
        Apply Non-Maximum Suppression to corner response.

        Only keeps local maxima within the specified window size.

        Returns:
            Corner response after NMS
        """
        print(f"Applying non-maximum suppression (window size={self.nms_window_size})...")

        rows, cols = self.corner_response.shape
        self.response_after_nms = np.zeros((rows, cols), dtype=np.float32)

        window_half = self.nms_window_size // 2

        for i in range(window_half, rows - window_half):
            for j in range(window_half, cols - window_half):
                # Extract window
                window = self.corner_response[i - window_half:i + window_half + 1,
                                             j - window_half:j + window_half + 1]
                max_val = np.max(window)

                # Keep only if local maximum
                if self.corner_response[i, j] == max_val:
                    self.response_after_nms[i, j] = self.corner_response[i, j]

        num_local_maxima = np.sum(self.response_after_nms > 0)
        print(f"  Local maxima after NMS: {num_local_maxima}")

        return self.response_after_nms

    def detect_corners(self) -> np.ndarray:
        """
        Detect corners by thresholding NMS response.

        Returns:
            Binary corner map
        """
        print(f"Detecting corners (threshold={self.corner_threshold})...")

        # Normalize response
        max_response = np.max(self.response_after_nms)
        if max_response > 0:
            self.response_after_nms /= max_response

        # Threshold to detect corners
        self.corners = (self.response_after_nms >= self.corner_threshold).astype(np.float32)

        num_corners = np.sum(self.corners)
        print(f"  Detected {int(num_corners)} corners")

        return self.corners

    def visualize_and_save(self, image_name: str = "result") -> None:
        """
        Create comprehensive visualization of advanced Harris detection pipeline.

        Args:
            image_name: Base name for output files
        """
        print(f"\nCreating visualizations...")

        # Get corner coordinates
        corner_rows, corner_cols = np.where(self.corners == 1)
        corner_points = list(zip(corner_cols, corner_rows))

        # Create image with corner overlays
        image_with_corners = self.image_color.copy()
        circle_radius = 5
        circle_color = (0, 0, 255)  # Red in BGR
        circle_thickness = 2

        for point in corner_points:
            cv2.circle(image_with_corners, point, circle_radius, circle_color, circle_thickness)

        # Multi-panel visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(self.image_gray, cmap='gray')
        axes[0, 0].set_title('Original Grayscale Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(np.abs(self.gradient_x), cmap='gray')
        axes[0, 1].set_title('Gradient X')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(np.abs(self.gradient_y), cmap='gray')
        axes[0, 2].set_title('Gradient Y')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(self.corner_response, cmap='jet')
        axes[1, 0].set_title('Corner Response (det/trace)')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.response_after_nms, cmap='jet')
        axes[1, 1].set_title(f'After NMS ({self.nms_window_size}x{self.nms_window_size})')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Detected Corners ({int(np.sum(self.corners))})')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_harris_advanced_pipeline.png',
                   dpi=150, bbox_inches='tight')
        plt.close()

        # Save individual outputs
        cv2.imwrite(str(self.output_dir / f'{image_name}_corners_overlay.png'),
                   image_with_corners)

        # Side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(self.image_gray, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'Detected Corners ({int(np.sum(self.corners))} corners)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_comparison.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

        print(f"[OK] All outputs saved to {self.output_dir}/")

    def run(self) -> np.ndarray:
        """
        Execute full advanced Harris corner detection pipeline.

        Returns:
            Binary corner map
        """
        print("\n" + "="*70)
        print("Advanced Harris Corner Detection Pipeline")
        print("="*70 + "\n")

        self.compute_gradients()
        self.compute_structure_tensor()
        self.compute_corner_response()
        self.apply_nms()
        self.detect_corners()

        return self.corners


def main():
    """Main entry point for advanced Harris corner detection."""
    parser = argparse.ArgumentParser(
        description='Advanced Harris Corner Detector with Non-Maximum Suppression',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python harris_detector_advanced.py --image board_1.jpg
  python harris_detector_advanced.py --image board_2.jpg --threshold 0.3 --nms-window 20
  python harris_detector_advanced.py --image board_1.jpg --size 300 300
        """
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--size', type=int, nargs=2, default=[200, 200],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Resize image to WIDTH HEIGHT (default: 200 200)')
    parser.add_argument('--threshold', type=float, default=0.25,
                       help='Corner detection threshold (default: 0.25)')
    parser.add_argument('--nms-window', type=int, default=15,
                       help='NMS window size (default: 15)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = AdvancedHarrisDetector(
            image_path=args.image,
            target_size=tuple(args.size),
            corner_threshold=args.threshold,
            nms_window_size=args.nms_window,
            output_dir=args.output_dir
        )

        # Run detection
        corners = detector.run()

        # Visualize and save results
        image_name = Path(args.image).stem
        detector.visualize_and_save(image_name)

        print("\n" + "="*70)
        print(f"[SUCCESS] Detected {int(np.sum(corners))} corners!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
