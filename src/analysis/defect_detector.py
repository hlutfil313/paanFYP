"""
Defect Detection Module
Contains algorithms for detecting defects in thermal images.
"""

import cv2
import numpy as np
from skimage import measure, morphology, filters
from scipy import ndimage
import matplotlib.pyplot as plt

class DefectDetector:
    def __init__(self):
        """Initialize the defect detector."""
        self.detection_methods = {
            'threshold_based': self.threshold_based_detection,
            'contour_based': self.contour_based_detection,
            'blob_detection': self.blob_detection,
            'edge_based': self.edge_based_detection,
            'region_growing': self.region_growing_detection
        }
        
    def detect_defects(self, image, method='threshold_based', **kwargs):
        """
        Detect defects in a thermal image using the specified method.
        
        Args:
            image: Input thermal image
            method: Detection method to use
            **kwargs: Additional parameters for the detection method
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        if method not in self.detection_methods:
            raise ValueError(f"Unknown detection method: {method}")
            
        return self.detection_methods[method](image, **kwargs)
        
    def threshold_based_detection(self, image, threshold_percentile=95, min_area=50, **kwargs):
        """
        Detect defects using threshold-based segmentation.
        
        Args:
            image: Input thermal image
            threshold_percentile: Percentile for threshold calculation
            min_area: Minimum area for defect regions
            **kwargs: Additional parameters
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        # Calculate threshold based on percentile
        threshold = np.percentile(image, threshold_percentile)
        
        # Create binary mask
        binary = image > threshold
        
        # Remove small objects
        binary = morphology.remove_small_objects(binary, min_size=min_area)
        
        # Find connected components
        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        # Extract bounding boxes
        defects = []
        for region in regions:
            minr, minc, maxr, maxc = region.bbox
            defects.append((minc, minr, maxc - minc, maxr - minr))
            
        return defects
        
    def contour_based_detection(self, image, blur_kernel=5, threshold_percentile=90, min_area=30, **kwargs):
        """
        Detect defects using contour detection.
        
        Args:
            image: Input thermal image
            blur_kernel: Kernel size for Gaussian blur
            threshold_percentile: Percentile for threshold calculation
            min_area: Minimum area for defect regions
            **kwargs: Additional parameters
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
        
        # Calculate threshold
        threshold = np.percentile(blurred, threshold_percentile)
        
        # Create binary image
        _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append((x, y, w, h))
                
        return defects
        
    def blob_detection(self, image, min_threshold=50, max_threshold=200, min_area=20, max_area=1000, **kwargs):
        """
        Detect defects using blob detection.
        
        Args:
            image: Input thermal image
            min_threshold: Minimum threshold for blob detection
            max_threshold: Maximum threshold for blob detection
            min_area: Minimum blob area
            max_area: Maximum blob area
            **kwargs: Additional parameters
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        # Create blob detector
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = min_threshold
        params.maxThreshold = max_threshold
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(image)
        
        # Convert keypoints to bounding boxes
        defects = []
        for keypoint in keypoints:
            x = int(keypoint.pt[0] - keypoint.size / 2)
            y = int(keypoint.pt[1] - keypoint.size / 2)
            size = int(keypoint.size)
            defects.append((x, y, size, size))
            
        return defects
        
    def edge_based_detection(self, image, low_threshold=50, high_threshold=150, min_area=25, **kwargs):
        """
        Detect defects using edge-based detection.
        
        Args:
            image: Input thermal image
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
            min_area: Minimum area for defect regions
            **kwargs: Additional parameters
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Dilate edges to connect components
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        defects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                defects.append((x, y, w, h))
                
        return defects
        
    def region_growing_detection(self, image, seed_threshold_percentile=95, similarity_threshold=10, min_area=40, **kwargs):
        """
        Detect defects using region growing algorithm.
        
        Args:
            image: Input thermal image
            seed_threshold_percentile: Percentile for seed point selection
            similarity_threshold: Threshold for region growing
            min_area: Minimum area for defect regions
            **kwargs: Additional parameters
            
        Returns:
            List of defect regions as (x, y, w, h) tuples
        """
        # Calculate seed threshold
        seed_threshold = np.percentile(image, seed_threshold_percentile)
        
        # Find seed points
        seed_points = np.where(image >= seed_threshold)
        
        # Initialize visited mask
        visited = np.zeros_like(image, dtype=bool)
        
        defects = []
        
        # Perform region growing from each seed point
        for i in range(len(seed_points[0])):
            y, x = seed_points[0][i], seed_points[1][i]
            
            if visited[y, x]:
                continue
                
            # Grow region
            region = self._grow_region(image, x, y, visited, similarity_threshold)
            
            if len(region) > min_area:
                # Calculate bounding box
                y_coords = [p[1] for p in region]
                x_coords = [p[0] for p in region]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                defects.append((min_x, min_y, max_x - min_x, max_y - min_y))
                
        return defects
        
    def _grow_region(self, image, start_x, start_y, visited, threshold):
        """
        Grow a region from a seed point.
        
        Args:
            image: Input image
            start_x: Starting x coordinate
            start_y: Starting y coordinate
            visited: Visited mask
            threshold: Similarity threshold
            
        Returns:
            List of region points
        """
        height, width = image.shape
        seed_value = image[start_y, start_x]
        region = []
        stack = [(start_x, start_y)]
        
        while stack:
            x, y = stack.pop()
            
            if (x < 0 or x >= width or y < 0 or y >= height or 
                visited[y, x] or abs(image[y, x] - seed_value) > threshold):
                continue
                
            visited[y, x] = True
            region.append((x, y))
            
            # Add neighbors to stack
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                stack.append((x + dx, y + dy))
                
        return region
        
    def analyze_defect_statistics(self, image, defects):
        """
        Analyze statistics of detected defects.
        
        Args:
            image: Input thermal image
            defects: List of defect regions
            
        Returns:
            Dictionary with defect statistics
        """
        if not defects:
            return {'count': 0, 'total_area': 0, 'avg_temperature': 0}
            
        total_area = 0
        temperatures = []
        
        for x, y, w, h in defects:
            area = w * h
            total_area += area
            
            # Extract region and calculate average temperature
            region = image[y:y+h, x:x+w]
            avg_temp = np.mean(region)
            temperatures.append(avg_temp)
            
        stats = {
            'count': len(defects),
            'total_area': total_area,
            'avg_temperature': np.mean(temperatures),
            'max_temperature': np.max(temperatures),
            'min_temperature': np.min(temperatures),
            'temperature_std': np.std(temperatures)
        }
        
        return stats
        
    def get_detection_info(self, method):
        """
        Get information about a detection method.
        
        Args:
            method: Detection method name
            
        Returns:
            Dictionary with method information
        """
        info = {
            'threshold_based': {
                'description': 'Detects defects using intensity thresholding',
                'best_for': 'High contrast thermal images with clear defect boundaries',
                'parameters': {'threshold_percentile': 'Percentile for threshold calculation', 'min_area': 'Minimum defect area'}
            },
            'contour_based': {
                'description': 'Detects defects using contour analysis',
                'best_for': 'Thermal images with well-defined defect shapes',
                'parameters': {'blur_kernel': 'Gaussian blur kernel size', 'threshold_percentile': 'Threshold percentile', 'min_area': 'Minimum defect area'}
            },
            'blob_detection': {
                'description': 'Detects defects using blob analysis',
                'best_for': 'Circular or blob-like defects',
                'parameters': {'min_threshold': 'Minimum blob threshold', 'max_threshold': 'Maximum blob threshold', 'min_area': 'Minimum blob area', 'max_area': 'Maximum blob area'}
            },
            'edge_based': {
                'description': 'Detects defects using edge analysis',
                'best_for': 'Defects with strong edge boundaries',
                'parameters': {'low_threshold': 'Lower Canny threshold', 'high_threshold': 'Upper Canny threshold', 'min_area': 'Minimum defect area'}
            },
            'region_growing': {
                'description': 'Detects defects using region growing algorithm',
                'best_for': 'Connected defect regions with similar intensity',
                'parameters': {'seed_threshold_percentile': 'Seed point threshold percentile', 'similarity_threshold': 'Region growing similarity threshold', 'min_area': 'Minimum defect area'}
            }
        }
        
        return info.get(method, {'description': 'Unknown method', 'best_for': 'N/A', 'parameters': {}}) 