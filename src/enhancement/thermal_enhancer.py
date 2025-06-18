"""
Thermal Image Enhancement Module
Contains various algorithms for enhancing thermal images to improve defect detection.
"""

import cv2
import numpy as np
from skimage import exposure, filters, morphology
from scipy import ndimage
import matplotlib.pyplot as plt

class ThermalImageEnhancer:
    def __init__(self):
        """Initialize the thermal image enhancer."""
        self.enhancement_methods = {
            'histogram_equalization': self.histogram_equalization,
            'adaptive_histogram_equalization': self.adaptive_histogram_equalization,
            'gaussian_filtering': self.gaussian_filtering,
            'edge_enhancement': self.edge_enhancement,
            'temperature_mapping': self.temperature_mapping,
            'multi_scale_enhancement': self.multi_scale_enhancement
        }
        
    def enhance(self, image, method='histogram_equalization', **kwargs):
        """
        Enhance a thermal image using the specified method.
        
        Args:
            image: Input thermal image (grayscale)
            method: Enhancement method to use
            **kwargs: Additional parameters for the enhancement method
            
        Returns:
            Enhanced image
        """
        if method not in self.enhancement_methods:
            raise ValueError(f"Unknown enhancement method: {method}")
            
        return self.enhancement_methods[method](image, **kwargs)
        
    def histogram_equalization(self, image, **kwargs):
        """
        Apply histogram equalization to improve contrast.
        
        Args:
            image: Input thermal image
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image with improved contrast
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(image)
        
        return enhanced
        
    def adaptive_histogram_equalization(self, image, clip_limit=2.0, tile_grid_size=(8, 8), **kwargs):
        """
        Apply adaptive histogram equalization for local contrast enhancement.
        
        Args:
            image: Input thermal image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            **kwargs: Additional parameters
            
        Returns:
            Enhanced image with local contrast improvement
        """
        # Ensure image is in uint8 format
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        # Apply adaptive histogram equalization
        enhanced = clahe.apply(image)
        
        return enhanced
        
    def gaussian_filtering(self, image, sigma=1.0, **kwargs):
        """
        Apply Gaussian filtering to reduce noise while preserving edges.
        
        Args:
            image: Input thermal image
            sigma: Standard deviation for Gaussian kernel
            **kwargs: Additional parameters
            
        Returns:
            Denoised image
        """
        # Apply Gaussian filter
        enhanced = ndimage.gaussian_filter(image, sigma=sigma)
        
        return enhanced.astype(image.dtype)
        
    def edge_enhancement(self, image, kernel_size=3, alpha=0.5, **kwargs):
        """
        Enhance edges in the thermal image to highlight defect boundaries.
        
        Args:
            image: Input thermal image
            kernel_size: Size of the Laplacian kernel
            alpha: Weight for edge enhancement
            **kwargs: Additional parameters
            
        Returns:
            Edge-enhanced image
        """
        # Ensure image is in float format for processing
        if image.dtype != np.float64:
            image = image.astype(np.float64)
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        # Apply Laplacian edge detection
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F, ksize=kernel_size)
        
        # Normalize Laplacian
        laplacian = np.abs(laplacian)
        laplacian = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min())
        
        # Enhance edges
        enhanced = image + alpha * laplacian * 255
        
        # Clip values to valid range
        enhanced = np.clip(enhanced, 0, 255)
        
        return enhanced.astype(np.uint8)
        
    def temperature_mapping(self, image, colormap='hot', **kwargs):
        """
        Convert thermal image to false-color representation for better visualization.
        
        Args:
            image: Input thermal image
            colormap: Matplotlib colormap to use
            **kwargs: Additional parameters
            
        Returns:
            False-color enhanced image
        """
        # Normalize image to 0-1 range
        normalized = (image - image.min()) / (image.max() - image.min())
        
        # Apply colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(normalized)
        
        # Convert to uint8 format
        enhanced = (colored[:, :, :3] * 255).astype(np.uint8)
        
        # Convert to grayscale for consistency
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        
        return enhanced_gray
        
    def multi_scale_enhancement(self, image, scales=[1, 2, 4], weights=[0.5, 0.3, 0.2], **kwargs):
        """
        Apply multi-scale enhancement combining different enhancement levels.
        
        Args:
            image: Input thermal image
            scales: List of scale factors for multi-scale processing
            weights: Weights for each scale level
            **kwargs: Additional parameters
            
        Returns:
            Multi-scale enhanced image
        """
        enhanced_scales = []
        
        for scale in scales:
            # Resize image
            if scale != 1:
                h, w = image.shape
                new_h, new_w = h // scale, w // scale
                resized = cv2.resize(image, (new_w, new_h))
            else:
                resized = image.copy()
                
            # Apply histogram equalization to this scale
            scale_enhanced = self.histogram_equalization(resized)
            
            # Resize back to original size
            if scale != 1:
                scale_enhanced = cv2.resize(scale_enhanced, (w, h))
                
            enhanced_scales.append(scale_enhanced)
            
        # Combine scales with weights
        enhanced = np.zeros_like(image, dtype=np.float64)
        for i, (scale_img, weight) in enumerate(zip(enhanced_scales, weights)):
            enhanced += weight * scale_img.astype(np.float64)
            
        return enhanced.astype(np.uint8)
        
    def morphological_enhancement(self, image, operation='open', kernel_size=3, **kwargs):
        """
        Apply morphological operations to enhance thermal image features.
        
        Args:
            image: Input thermal image
            operation: Morphological operation ('open', 'close', 'tophat', 'blackhat')
            kernel_size: Size of morphological kernel
            **kwargs: Additional parameters
            
        Returns:
            Morphologically enhanced image
        """
        # Create kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        if operation == 'open':
            enhanced = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif operation == 'close':
            enhanced = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif operation == 'tophat':
            enhanced = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        elif operation == 'blackhat':
            enhanced = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
            
        return enhanced
        
    def unsharp_masking(self, image, sigma=1.0, amount=1.0, threshold=0, **kwargs):
        """
        Apply unsharp masking to enhance fine details.
        
        Args:
            image: Input thermal image
            sigma: Standard deviation for Gaussian blur
            amount: Strength of the enhancement
            threshold: Minimum brightness change
            **kwargs: Additional parameters
            
        Returns:
            Unsharp masked image
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        
        # Calculate sharpened image
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
            
        return sharpened
        
    def get_enhancement_info(self, method):
        """
        Get information about an enhancement method.
        
        Args:
            method: Enhancement method name
            
        Returns:
            Dictionary with method information
        """
        info = {
            'histogram_equalization': {
                'description': 'Improves global contrast by equalizing image histogram',
                'best_for': 'Low contrast thermal images',
                'parameters': {}
            },
            'adaptive_histogram_equalization': {
                'description': 'Improves local contrast using adaptive histogram equalization',
                'best_for': 'Images with varying lighting conditions',
                'parameters': {'clip_limit': 'Contrast limiting threshold', 'tile_grid_size': 'Grid size for local processing'}
            },
            'gaussian_filtering': {
                'description': 'Reduces noise while preserving edges',
                'best_for': 'Noisy thermal images',
                'parameters': {'sigma': 'Standard deviation of Gaussian kernel'}
            },
            'edge_enhancement': {
                'description': 'Enhances edges to highlight defect boundaries',
                'best_for': 'Defect detection and boundary analysis',
                'parameters': {'kernel_size': 'Laplacian kernel size', 'alpha': 'Enhancement strength'}
            },
            'temperature_mapping': {
                'description': 'Converts to false-color representation for better visualization',
                'best_for': 'Temperature analysis and visualization',
                'parameters': {'colormap': 'Matplotlib colormap name'}
            },
            'multi_scale_enhancement': {
                'description': 'Combines enhancements at multiple scales',
                'best_for': 'Complex thermal images with features at different scales',
                'parameters': {'scales': 'List of scale factors', 'weights': 'Weights for each scale'}
            }
        }
        
        return info.get(method, {'description': 'Unknown method', 'best_for': 'N/A', 'parameters': {}}) 