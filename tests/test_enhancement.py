"""
Unit tests for thermal image enhancement module.
"""

import unittest
import numpy as np
import cv2
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhancement.thermal_enhancer import ThermalImageEnhancer

class TestThermalEnhancer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = ThermalImageEnhancer()
        
        # Create a test image
        self.test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
    def test_histogram_equalization(self):
        """Test histogram equalization."""
        enhanced = self.enhancer.histogram_equalization(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_adaptive_histogram_equalization(self):
        """Test adaptive histogram equalization."""
        enhanced = self.enhancer.adaptive_histogram_equalization(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_gaussian_filtering(self):
        """Test Gaussian filtering."""
        enhanced = self.enhancer.gaussian_filtering(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, self.test_image.dtype)
        
    def test_edge_enhancement(self):
        """Test edge enhancement."""
        enhanced = self.enhancer.edge_enhancement(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_temperature_mapping(self):
        """Test temperature mapping."""
        enhanced = self.enhancer.temperature_mapping(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_multi_scale_enhancement(self):
        """Test multi-scale enhancement."""
        enhanced = self.enhancer.multi_scale_enhancement(self.test_image)
        
        self.assertEqual(enhanced.shape, self.test_image.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        
    def test_invalid_method(self):
        """Test that invalid enhancement method raises error."""
        with self.assertRaises(ValueError):
            self.enhancer.enhance(self.test_image, method='invalid_method')
            
    def test_enhancement_info(self):
        """Test getting enhancement method information."""
        info = self.enhancer.get_enhancement_info('histogram_equalization')
        
        self.assertIn('description', info)
        self.assertIn('best_for', info)
        self.assertIn('parameters', info)

if __name__ == '__main__':
    unittest.main() 