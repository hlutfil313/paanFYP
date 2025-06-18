# Thermal Image Enhancement Application - User Guide

## Overview

The Thermal Image Enhancement Application is designed to enhance thermal images and compare them with RGB images to demonstrate the effectiveness of thermal imaging in defect detection. This tool helps prove that thermal images can provide valuable insights that may not be visible in standard RGB images.

## Features

### Core Functionality
- **Image Loading**: Load both thermal and RGB images from your computer
- **Thermal Enhancement**: Apply various enhancement algorithms to thermal images
- **Defect Detection**: Identify potential defects using thermal analysis
- **Comparison View**: Side-by-side comparison of original and enhanced images
- **Export Results**: Save enhanced images and comparison results

### Enhancement Algorithms
1. **Histogram Equalization**: Improves global contrast
2. **Adaptive Histogram Equalization**: Enhances local contrast
3. **Gaussian Filtering**: Reduces noise while preserving edges
4. **Edge Enhancement**: Highlights defect boundaries
5. **Temperature Mapping**: Converts to false-color representation
6. **Multi-scale Enhancement**: Combines enhancements at multiple scales

### Defect Detection Methods
1. **Threshold-based**: Uses intensity thresholding
2. **Contour-based**: Analyzes image contours
3. **Blob Detection**: Identifies blob-like defects
4. **Edge-based**: Uses edge analysis
5. **Region Growing**: Grows regions from seed points

## Installation

### Prerequisites
- Python 3.7 or higher
- Windows 10/11 (tested on Windows 10)

### Quick Installation
1. Download or clone the project
2. Open Command Prompt or PowerShell in the project directory
3. Run the installation script:
   ```bash
   python install.py
   ```

### Manual Installation
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate sample data:
   ```bash
   python src/utils/sample_generator.py
   ```

## Usage

### Starting the Application
Run the application using one of these methods:
```bash
python main.py
```
or
```bash
python run.py
```

### Basic Workflow

#### 1. Loading Images
1. Click "Browse" next to "Thermal Image" to select your thermal image
2. Optionally, click "Browse" next to "RGB Image" to select a corresponding RGB image for comparison
3. The application will automatically load and display the images

#### 2. Enhancing Thermal Images
1. Select an enhancement method from the dropdown menu
2. Click "Enhance Thermal Image"
3. The enhanced image will appear in the right panel
4. Compare the original and enhanced images

#### 3. Detecting Defects
1. Click "Detect Defects" to analyze the thermal image
2. Detected defects will be highlighted with green rectangles
3. Check the status bar for information about detected defects

#### 4. Comparing Images
1. Load both thermal and RGB images
2. Click "Compare Images" to open a comparison window
3. Analyze the differences between thermal and RGB representations

#### 5. Saving Results
1. Click "Save Enhanced Image" to save the enhanced thermal image
2. Choose the save location and format (PNG recommended)

### Advanced Usage

#### Enhancement Method Selection
- **Histogram Equalization**: Best for low contrast thermal images
- **Adaptive Histogram Equalization**: Best for images with varying lighting
- **Gaussian Filtering**: Best for noisy thermal images
- **Edge Enhancement**: Best for defect detection and boundary analysis
- **Temperature Mapping**: Best for temperature analysis and visualization
- **Multi-scale Enhancement**: Best for complex images with features at different scales

#### Defect Detection Tuning
Different detection methods work better for different types of defects:
- **Threshold-based**: Good for high contrast defects
- **Contour-based**: Good for well-defined defect shapes
- **Blob Detection**: Good for circular or blob-like defects
- **Edge-based**: Good for defects with strong boundaries
- **Region Growing**: Good for connected defect regions

## File Formats

### Supported Input Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff, .tif)

### Output Formats
- PNG (recommended for lossless quality)
- JPEG (for smaller file sizes)

## Sample Data

The application includes a sample data generator that creates synthetic thermal and RGB images with simulated defects. To generate sample data:

```bash
python src/utils/sample_generator.py
```

Sample images will be saved in the `data/samples/` directory.

## Troubleshooting

### Common Issues

#### Application Won't Start
- Ensure Python 3.7+ is installed
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify you're running the script from the project root directory

#### Images Won't Load
- Check that the image file exists and is not corrupted
- Ensure the image format is supported
- Try converting the image to PNG format

#### Enhancement Not Working
- Ensure the thermal image is loaded
- Try a different enhancement method
- Check that the image is in grayscale format

#### Defect Detection Issues
- Try different detection methods
- Adjust the enhancement method first
- Ensure the image has sufficient contrast

### Performance Tips
- Use images with dimensions under 2048x2048 for best performance
- Close other applications to free up memory
- Use PNG format for better quality and processing

## Technical Details

### Image Processing Pipeline
1. **Loading**: Images are loaded and converted to appropriate format
2. **Preprocessing**: Basic normalization and format conversion
3. **Enhancement**: Selected algorithm is applied
4. **Analysis**: Defect detection is performed
5. **Visualization**: Results are displayed and can be saved

### Algorithm Details
Each enhancement algorithm is optimized for thermal imaging:
- **Histogram Equalization**: Spreads pixel values across the full range
- **Adaptive Histogram Equalization**: Applies local contrast enhancement
- **Gaussian Filtering**: Smooths noise while preserving important features
- **Edge Enhancement**: Uses Laplacian operator to highlight edges
- **Temperature Mapping**: Applies false-color colormaps for better visualization

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the console output for error messages
3. Ensure all dependencies are properly installed
4. Try with sample data first to verify functionality

## Contributing

To contribute to the project:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is open source and available under the MIT License. 