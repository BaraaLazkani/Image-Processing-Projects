"""
Canny Edge Detector - Multi-Stage Edge Detection

This module implements the Canny edge detection algorithm, a multi-stage
process for detecting edges in images with good detection, localization,
and minimal response properties.

The Canny algorithm consists of five stages:
1. Gaussian smoothing - noise reduction
2. Gradient computation - find intensity gradients
3. Non-maximum suppression - thin edges to single pixel width
4. Double thresholding - classify pixels as strong/weak/non-edges
5. Edge tracking by hysteresis - connect weak edges to strong edges

Author: Computer Vision Course Project
"""

import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt


class CannyEdgeDetector:
    """Canny edge detection implementation."""

    # 5x5 Gaussian filter (sigma=1)
    GAUSSIAN_KERNEL = np.array([
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
        [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
        [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]
    ])

    # Sobel kernels for gradient computation
    SOBEL_X = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    SOBEL_Y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    def __init__(self, image_path: str, target_size: Tuple[int, int] = (200, 200),
                 low_threshold: float = 0.03, high_threshold: float = 0.3,
                 output_dir: str = "output"):
        """
        Initialize Canny Edge Detector.

        Args:
            image_path: Path to input image
            target_size: Resize dimensions (width, height)
            low_threshold: Low threshold for double thresholding
            high_threshold: High threshold for double thresholding
            output_dir: Directory for saving outputs
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load and preprocess image
        self.image_color = self._load_image(image_path, target_size)
        self.image_gray = cv2.cvtColor(self.image_color, cv2.COLOR_BGR2GRAY)

        # Storage for intermediate results
        self.smoothed = None
        self.gradient_x = None
        self.gradient_y = None
        self.gradient_magnitude = None
        self.gradient_direction = None
        self.quantized_direction = None
        self.nms_result = None
        self.thresholded = None
        self.edges = None

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

    def gaussian_smoothing(self) -> np.ndarray:
        """
        Stage 1: Apply Gaussian smoothing to reduce noise.

        Returns:
            Smoothed image
        """
        print("Stage 1: Gaussian smoothing...")
        self.smoothed = self._convolve(self.image_gray.astype(np.float32),
                                      self.GAUSSIAN_KERNEL)
        return self.smoothed

    def compute_gradients(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stage 2: Compute intensity gradients using Sobel operator.

        Returns:
            Tuple of (gradient_magnitude, gradient_direction_degrees)
        """
        print("Stage 2: Computing gradients...")

        # Compute gradients using Sobel
        self.gradient_x = self._convolve(self.smoothed, self.SOBEL_X)
        self.gradient_y = self._convolve(self.smoothed, self.SOBEL_Y)

        # Compute magnitude and direction
        self.gradient_magnitude = np.sqrt(self.gradient_x**2 + self.gradient_y**2)
        self.gradient_magnitude /= np.max(self.gradient_magnitude)  # Normalize

        self.gradient_direction = np.arctan2(self.gradient_y, self.gradient_x) * 180 / np.pi

        print(f"  Gradient magnitude range: [0.0, 1.0] (normalized)")
        print(f"  Gradient direction range: [{np.min(self.gradient_direction):.1f}°, "
              f"{np.max(self.gradient_direction):.1f}°]")

        return self.gradient_magnitude, self.gradient_direction

    def _quantize_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Quantize gradient directions to nearest 45-degree angle.

        Args:
            angles: Gradient directions in degrees

        Returns:
            Quantized angles (0, 45, 90, 135, 180, 225, 270, 315)
        """
        rows, cols = angles.shape
        quantized = np.zeros_like(angles)

        for i in range(rows):
            for j in range(cols):
                angle = angles[i, j] + 180  # Shift to [0, 360]

                if ((angle >= 0) and (angle < 22.5)) or ((angle >= 337.5) and (angle <= 360)):
                    quantized[i, j] = 0
                elif (angle >= 22.5) and (angle < 67.5):
                    quantized[i, j] = 45
                elif (angle >= 67.5) and (angle < 112.5):
                    quantized[i, j] = 90
                elif (angle >= 112.5) and (angle < 157.5):
                    quantized[i, j] = 135
                elif (angle >= 157.5) and (angle < 202.5):
                    quantized[i, j] = 180
                elif (angle >= 202.5) and (angle < 247.5):
                    quantized[i, j] = 225
                elif (angle >= 247.5) and (angle < 292.5):
                    quantized[i, j] = 270
                elif (angle >= 292.5) and (angle < 337.5):
                    quantized[i, j] = 315

        return quantized

    def non_maximum_suppression(self) -> np.ndarray:
        """
        Stage 3: Thin edges by suppressing non-maximum pixels.

        Only keep pixels that are local maxima in the gradient direction.

        Returns:
            Edge map after NMS
        """
        print("Stage 3: Non-maximum suppression...")

        # Quantize gradient directions
        self.quantized_direction = self._quantize_angles(self.gradient_direction)

        rows, cols = self.gradient_magnitude.shape
        self.nms_result = np.zeros((rows, cols), dtype=np.float32)

        # Suppress non-maxima
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                angle = self.quantized_direction[i, j]
                magnitude = self.gradient_magnitude[i, j]

                # Compare with neighbors along gradient direction
                if angle in [0, 180]:
                    neighbors = [self.gradient_magnitude[i, j - 1], self.gradient_magnitude[i, j + 1]]
                elif angle in [45, 225]:
                    neighbors = [self.gradient_magnitude[i - 1, j + 1], self.gradient_magnitude[i + 1, j - 1]]
                elif angle in [90, 270]:
                    neighbors = [self.gradient_magnitude[i - 1, j], self.gradient_magnitude[i + 1, j]]
                else:  # 135, 315
                    neighbors = [self.gradient_magnitude[i - 1, j - 1], self.gradient_magnitude[i + 1, j + 1]]

                # Keep only if local maximum
                if magnitude >= max(neighbors):
                    self.nms_result[i, j] = magnitude

        # Trim border
        self.nms_result = self.nms_result[1:-1, 1:-1]

        num_edges = np.sum(self.nms_result > 0)
        print(f"  Remaining edge pixels after NMS: {num_edges}")

        return self.nms_result

    def double_threshold(self) -> np.ndarray:
        """
        Stage 4: Classify pixels as strong/weak/non-edges using double threshold.

        Classifications:
        - 0: Non-edge (< low_threshold)
        - 1: Weak edge (>= low_threshold and < high_threshold)
        - 2: Strong edge (>= high_threshold)

        Returns:
            Classified edge map
        """
        print(f"Stage 4: Double thresholding (low={self.low_threshold}, high={self.high_threshold})...")

        self.thresholded = np.zeros_like(self.nms_result)

        # Classify pixels
        weak_mask = (self.nms_result >= self.low_threshold) & (self.nms_result < self.high_threshold)
        strong_mask = self.nms_result >= self.high_threshold

        self.thresholded[weak_mask] = 1
        self.thresholded[strong_mask] = 2

        num_weak = np.sum(self.thresholded == 1)
        num_strong = np.sum(self.thresholded == 2)
        print(f"  Strong edges: {num_strong}, Weak edges: {num_weak}")

        return self.thresholded

    def edge_tracking_by_hysteresis(self) -> np.ndarray:
        """
        Stage 5: Connect weak edges to strong edges via hysteresis.

        Weak edges are kept only if connected to strong edges.

        Returns:
            Final edge map
        """
        print("Stage 5: Edge tracking by hysteresis...")

        rows, cols = self.thresholded.shape
        self.edges = np.zeros_like(self.thresholded)

        # Keep all strong edges
        self.edges[self.thresholded >= self.high_threshold] = self.thresholded[self.thresholded >= self.high_threshold]

        # Check weak edges for connectivity to strong edges
        for i in range(rows):
            for j in range(cols):
                if (self.thresholded[i, j] >= self.low_threshold) and \
                   (self.thresholded[i, j] < self.high_threshold):

                    # Check 8-connected neighborhood for strong edges
                    connected_to_strong = False
                    try:
                        for x in range(-1, 2):
                            for y in range(-1, 2):
                                if (0 <= i + x < rows) and (0 <= j + y < cols):
                                    if self.thresholded[i + x, j + y] > self.high_threshold:
                                        connected_to_strong = True
                                        break
                            if connected_to_strong:
                                break
                    except:
                        continue

                    if connected_to_strong:
                        self.edges[i, j] = self.thresholded[i, j]

        num_final_edges = np.sum(self.edges > 0)
        print(f"  Final edge pixels after hysteresis: {num_final_edges}")

        return self.edges

    def visualize_and_save(self, image_name: str = "result") -> None:
        """
        Create comprehensive visualization of Canny edge detection pipeline.

        Args:
            image_name: Base name for output files
        """
        print(f"\nCreating visualizations...")

        # Create multi-panel pipeline visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        axes[0, 0].imshow(self.image_gray, cmap='gray')
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(self.smoothed, cmap='gray')
        axes[0, 1].set_title('2. Gaussian Smoothed')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(self.gradient_magnitude, cmap='gray')
        axes[0, 2].set_title('3. Gradient Magnitude')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(self.nms_result, cmap='gray')
        axes[1, 0].set_title('4. After Non-Maximum Suppression')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(self.thresholded, cmap='gray')
        axes[1, 1].set_title(f'5. After Double Threshold (L={self.low_threshold}, H={self.high_threshold})')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(self.edges, cmap='gray')
        axes[1, 2].set_title('6. Final Edges (After Hysteresis)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_canny_pipeline.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

        # Save individual stages
        cv2.imwrite(str(self.output_dir / f'{image_name}_1_smoothed.png'),
                   (self.smoothed / np.max(self.smoothed) * 255).astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_2_gradient.png'),
                   (self.gradient_magnitude * 255).astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_3_nms.png'),
                   (self.nms_result / np.max(self.nms_result) * 255).astype(np.uint8)
                   if np.max(self.nms_result) > 0 else self.nms_result.astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_4_thresholded.png'),
                   (self.thresholded / 2 * 255).astype(np.uint8))
        cv2.imwrite(str(self.output_dir / f'{image_name}_5_final_edges.png'),
                   (self.edges > 0).astype(np.uint8) * 255)

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].imshow(self.image_gray, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(self.edges, cmap='gray')
        axes[1].set_title(f'Detected Edges ({np.sum(self.edges > 0)} pixels)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'{image_name}_comparison.png', dpi=150,
                   bbox_inches='tight')
        plt.close()

        print(f"[OK] All outputs saved to {self.output_dir}/")

    def run(self) -> np.ndarray:
        """
        Execute full Canny edge detection pipeline.

        Returns:
            Final edge map
        """
        print("\n" + "="*70)
        print("Canny Edge Detection Pipeline")
        print("="*70 + "\n")

        self.gaussian_smoothing()
        self.compute_gradients()
        self.non_maximum_suppression()
        self.double_threshold()
        self.edge_tracking_by_hysteresis()

        return self.edges


def main():
    """Main entry point for Canny edge detection."""
    parser = argparse.ArgumentParser(
        description='Canny Edge Detector - Multi-Stage Edge Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python canny_detector.py --image city.jpg
  python canny_detector.py --image city.jpg --low 0.03 --high 0.3
  python canny_detector.py --image I2.jpg --low 0.02 --high 0.1 --size 300 300
        """
    )

    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--size', type=int, nargs=2, default=[200, 200],
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Resize image to WIDTH HEIGHT (default: 200 200)')
    parser.add_argument('--low', type=float, default=0.03,
                       help='Low threshold for double thresholding (default: 0.03)')
    parser.add_argument('--high', type=float, default=0.3,
                       help='High threshold for double thresholding (default: 0.3)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory (default: output)')

    args = parser.parse_args()

    try:
        # Initialize detector
        detector = CannyEdgeDetector(
            image_path=args.image,
            target_size=tuple(args.size),
            low_threshold=args.low,
            high_threshold=args.high,
            output_dir=args.output_dir
        )

        # Run detection
        edges = detector.run()

        # Visualize and save results
        image_name = Path(args.image).stem
        detector.visualize_and_save(image_name)

        print("\n" + "="*70)
        print(f"[SUCCESS] Detected {np.sum(edges > 0)} edge pixels!")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        raise


if __name__ == "__main__":
    main()
