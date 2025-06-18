"""
Image Utility Functions
Contains helper functions for image processing and file operations.
"""

import cv2
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

class ImageUtils:
    def __init__(self):
        """Initialize the image utilities."""
        pass
        
    def load_image(self, file_path, grayscale=True):
        """
        Load an image from file.
        
        Args:
            file_path: Path to the image file
            grayscale: Whether to load as grayscale
            
        Returns:
            Loaded image as numpy array
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")
            
        if grayscale:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
            
        return image
        
    def save_image(self, image, file_path, quality=95):
        """
        Save an image to file.
        
        Args:
            image: Image to save
            file_path: Output file path
            quality: JPEG quality (1-100)
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save image
        cv2.imwrite(file_path, image)
        
    def resize_image(self, image, width=None, height=None, interpolation=cv2.INTER_LINEAR):
        """
        Resize an image while maintaining aspect ratio.
        
        Args:
            image: Input image
            width: Target width (None to maintain aspect ratio)
            height: Target height (None to maintain aspect ratio)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image
            
        if width is None:
            # Calculate width based on height
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
            
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        return resized
        
    def normalize_image(self, image, min_val=0, max_val=255):
        """
        Normalize image values to specified range.
        
        Args:
            image: Input image
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            Normalized image
        """
        img_min = image.min()
        img_max = image.max()
        
        if img_max == img_min:
            return np.full_like(image, min_val)
            
        normalized = (image - img_min) / (img_max - img_min) * (max_val - min_val) + min_val
        return normalized.astype(image.dtype)
        
    def apply_colormap(self, image, colormap='hot'):
        """
        Apply a colormap to a grayscale image.
        
        Args:
            image: Input grayscale image
            colormap: Matplotlib colormap name
            
        Returns:
            Colored image
        """
        # Normalize image to 0-1 range
        normalized = self.normalize_image(image, 0, 1)
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(normalized)
        
        # Convert to uint8
        colored = (colored[:, :, :3] * 255).astype(np.uint8)
        
        return colored
        
    def calculate_image_statistics(self, image):
        """
        Calculate basic statistics for an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        stats = {
            'mean': np.mean(image),
            'std': np.std(image),
            'min': np.min(image),
            'max': np.max(image),
            'median': np.median(image),
            'shape': image.shape,
            'dtype': str(image.dtype)
        }
        
        return stats
        
    def create_histogram(self, image, bins=256):
        """
        Create histogram for an image.
        
        Args:
            image: Input image
            bins: Number of histogram bins
            
        Returns:
            Tuple of (histogram, bin_edges)
        """
        return np.histogram(image.flatten(), bins=bins, range=(0, 255))
        
    def plot_histogram(self, image, title="Image Histogram", bins=256):
        """
        Plot histogram for an image.
        
        Args:
            image: Input image
            title: Plot title
            bins: Number of histogram bins
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(image.flatten(), bins=bins, range=(0, 255), alpha=0.7, color='blue')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        return fig
        
    def compare_images(self, image1, image2, title1="Image 1", title2="Image 2"):
        """
        Create a side-by-side comparison of two images.
        
        Args:
            image1: First image
            image2: Second image
            title1: Title for first image
            title2: Title for second image
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot first image
        if len(image1.shape) == 2:
            ax1.imshow(image1, cmap='hot')
        else:
            ax1.imshow(image1)
        ax1.set_title(title1)
        ax1.axis('off')
        
        # Plot second image
        if len(image2.shape) == 2:
            ax2.imshow(image2, cmap='hot')
        else:
            ax2.imshow(image2)
        ax2.set_title(title2)
        ax2.axis('off')
        
        plt.tight_layout()
        return fig
        
    def create_montage(self, images, titles=None, cols=3, figsize=(15, 10)):
        """
        Create a montage of multiple images.
        
        Args:
            images: List of images
            titles: List of titles (optional)
            cols: Number of columns
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
            
        for i, image in enumerate(images):
            row = i // cols
            col = i % cols
            
            if len(image.shape) == 2:
                axes[row, col].imshow(image, cmap='hot')
            else:
                axes[row, col].imshow(image)
                
            if titles and i < len(titles):
                axes[row, col].set_title(titles[i])
            axes[row, col].axis('off')
            
        # Hide empty subplots
        for i in range(n_images, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
            
        plt.tight_layout()
        return fig
        
    def validate_image_path(self, file_path):
        """
        Validate if a file path points to a valid image file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if valid, False otherwise
        """
        if not os.path.exists(file_path):
            return False
            
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext not in valid_extensions:
            return False
            
        # Try to open the image
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True
        except:
            return False
            
    def get_supported_formats(self):
        """
        Get list of supported image formats.
        
        Returns:
            List of supported file extensions
        """
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'] 