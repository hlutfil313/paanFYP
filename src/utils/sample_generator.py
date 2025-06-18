"""
Sample Image Generator
Generates sample thermal and RGB images for testing the application.
"""

import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class SampleGenerator:
    def __init__(self):
        """Initialize the sample generator."""
        self.output_dir = "data/samples"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_thermal_image(self, width=512, height=512, defects=True):
        """
        Generate a sample thermal image.
        
        Args:
            width: Image width
            height: Image height
            defects: Whether to include simulated defects
            
        Returns:
            Generated thermal image
        """
        # Create base thermal image with temperature gradient
        x = np.linspace(0, 1, width)
        y = np.linspace(0, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Create temperature field with some variation
        temperature = 50 + 30 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        temperature += 20 * np.random.normal(0, 0.1, (height, width))
        
        # Add simulated defects if requested
        if defects:
            temperature = self._add_simulated_defects(temperature)
            
        # Convert to uint8 (0-255 range)
        thermal_image = ((temperature - temperature.min()) / 
                        (temperature.max() - temperature.min()) * 255).astype(np.uint8)
        
        return thermal_image
        
    def generate_rgb_image(self, width=512, height=512, defects=True):
        """
        Generate a sample RGB image.
        
        Args:
            width: Image width
            height: Image height
            defects: Whether to include simulated defects
            
        Returns:
            Generated RGB image
        """
        # Create base RGB image
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background texture
        for i in range(3):
            noise = np.random.normal(128, 30, (height, width))
            rgb_image[:, :, i] = np.clip(noise, 0, 255).astype(np.uint8)
            
        # Add some structural elements
        # Vertical lines
        for i in range(0, width, 50):
            rgb_image[:, i:i+2, :] = [100, 100, 100]
            
        # Horizontal lines
        for i in range(0, height, 50):
            rgb_image[i:i+2, :, :] = [100, 100, 100]
            
        # Add simulated defects if requested
        if defects:
            rgb_image = self._add_rgb_defects(rgb_image)
            
        return rgb_image
        
    def _add_simulated_defects(self, temperature):
        """Add simulated defects to thermal image."""
        height, width = temperature.shape
        
        # Add hot spots (defects)
        num_defects = np.random.randint(2, 6)
        for _ in range(num_defects):
            # Random defect position
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            
            # Random defect size
            size = np.random.randint(10, 30)
            
            # Create hot spot
            for i in range(max(0, y - size), min(height, y + size)):
                for j in range(max(0, x - size), min(width, x + size)):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance < size:
                        # Increase temperature at defect location
                        temperature[i, j] += 15 * np.exp(-distance / (size / 3))
                        
        # Add cold spots (defects)
        num_cold_defects = np.random.randint(1, 4)
        for _ in range(num_cold_defects):
            x = np.random.randint(50, width - 50)
            y = np.random.randint(50, height - 50)
            size = np.random.randint(8, 20)
            
            for i in range(max(0, y - size), min(height, y + size)):
                for j in range(max(0, x - size), min(width, x + size)):
                    distance = np.sqrt((i - y)**2 + (j - x)**2)
                    if distance < size:
                        # Decrease temperature at defect location
                        temperature[i, j] -= 10 * np.exp(-distance / (size / 3))
                        
        return temperature
        
    def _add_rgb_defects(self, rgb_image):
        """Add simulated defects to RGB image."""
        height, width = rgb_image.shape[:2]
        
        # Add some visible defects (cracks, stains, etc.)
        num_defects = np.random.randint(1, 4)
        for _ in range(num_defects):
            defect_type = np.random.choice(['crack', 'stain', 'hole'])
            
            if defect_type == 'crack':
                # Add a crack
                start_x = np.random.randint(0, width)
                start_y = np.random.randint(0, height)
                length = np.random.randint(20, 60)
                angle = np.random.uniform(0, 2 * np.pi)
                
                for i in range(length):
                    x = int(start_x + i * np.cos(angle))
                    y = int(start_y + i * np.sin(angle))
                    if 0 <= x < width and 0 <= y < height:
                        rgb_image[y, x, :] = [50, 50, 50]  # Dark crack
                        
            elif defect_type == 'stain':
                # Add a stain
                x = np.random.randint(50, width - 50)
                y = np.random.randint(50, height - 50)
                size = np.random.randint(15, 35)
                
                for i in range(max(0, y - size), min(height, y + size)):
                    for j in range(max(0, x - size), min(width, x + size)):
                        distance = np.sqrt((i - y)**2 + (j - x)**2)
                        if distance < size:
                            # Add dark stain
                            rgb_image[i, j, :] = np.clip(rgb_image[i, j, :] * 0.7, 0, 255).astype(np.uint8)
                            
            elif defect_type == 'hole':
                # Add a hole
                x = np.random.randint(50, width - 50)
                y = np.random.randint(50, height - 50)
                size = np.random.randint(8, 20)
                
                for i in range(max(0, y - size), min(height, y + size)):
                    for j in range(max(0, x - size), min(width, x + size)):
                        distance = np.sqrt((i - y)**2 + (j - x)**2)
                        if distance < size:
                            # Make it black (hole)
                            rgb_image[i, j, :] = [0, 0, 0]
                            
        return rgb_image
        
    def generate_sample_dataset(self, num_samples=5):
        """
        Generate a complete sample dataset.
        
        Args:
            num_samples: Number of sample pairs to generate
        """
        print(f"Generating {num_samples} sample image pairs...")
        
        for i in range(num_samples):
            # Generate thermal image
            thermal = self.generate_thermal_image(defects=True)
            thermal_path = os.path.join(self.output_dir, f"thermal_sample_{i+1:02d}.png")
            cv2.imwrite(thermal_path, thermal)
            
            # Generate corresponding RGB image
            rgb = self.generate_rgb_image(defects=True)
            rgb_path = os.path.join(self.output_dir, f"rgb_sample_{i+1:02d}.png")
            cv2.imwrite(rgb_path, rgb)
            
            print(f"Generated: {thermal_path}, {rgb_path}")
            
        # Generate one pair without defects for comparison
        thermal_no_defects = self.generate_thermal_image(defects=False)
        rgb_no_defects = self.generate_rgb_image(defects=False)
        
        thermal_no_defects_path = os.path.join(self.output_dir, "thermal_no_defects.png")
        rgb_no_defects_path = os.path.join(self.output_dir, "rgb_no_defects.png")
        
        cv2.imwrite(thermal_no_defects_path, thermal_no_defects)
        cv2.imwrite(rgb_no_defects_path, rgb_no_defects)
        
        print(f"Generated control images: {thermal_no_defects_path}, {rgb_no_defects_path}")
        print("Sample dataset generation complete!")
        
    def create_comparison_image(self, thermal_path, rgb_path, output_path):
        """
        Create a side-by-side comparison image.
        
        Args:
            thermal_path: Path to thermal image
            rgb_path: Path to RGB image
            output_path: Output path for comparison image
        """
        # Load images
        thermal = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Resize to same height
        height = min(thermal.shape[0], rgb.shape[0])
        thermal_resized = cv2.resize(thermal, (int(thermal.shape[1] * height / thermal.shape[0]), height))
        rgb_resized = cv2.resize(rgb, (int(rgb.shape[1] * height / rgb.shape[0]), height))
        
        # Create comparison image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        ax1.imshow(thermal_resized, cmap='hot')
        ax1.set_title('Thermal Image')
        ax1.axis('off')
        
        ax2.imshow(rgb_resized)
        ax2.set_title('RGB Image')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison image saved: {output_path}")

def main():
    """Generate sample dataset."""
    generator = SampleGenerator()
    generator.generate_sample_dataset(num_samples=5)
    
    # Create a comparison image
    thermal_path = os.path.join(generator.output_dir, "thermal_sample_01.png")
    rgb_path = os.path.join(generator.output_dir, "rgb_sample_01.png")
    comparison_path = os.path.join(generator.output_dir, "comparison_sample_01.png")
    
    if os.path.exists(thermal_path) and os.path.exists(rgb_path):
        generator.create_comparison_image(thermal_path, rgb_path, comparison_path)

if __name__ == "__main__":
    main() 