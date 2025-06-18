# Thermal Image Enhancement for Defect Detection

## Purpose
This application is designed to enhance thermal images and compare them with RGB images to demonstrate the effectiveness of thermal imaging in defect detection. The goal is to prove that thermal images can provide valuable insights that may not be visible in standard RGB images.

## 🚀 Quick Start (For Friends/Users)

### **Super Easy Setup (Windows)**
1. Download or clone this repository
2. **Double-click `setup.bat`** - This will install everything and start the app automatically!

### **Manual Setup**
1. Install Python 3.7 or higher
2. Open Command Prompt/PowerShell in the project folder
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Generate sample data (optional):
   ```bash
   python src/utils/sample_generator.py
   ```
5. Run the application:
   ```bash
   python main.py
   ```

## Features
- **Image Loading**: Load both thermal and RGB images from your computer
- **Thermal Enhancement**: Apply various enhancement algorithms to thermal images
- **Comparison View**: Side-by-side comparison of original and enhanced images
- **Defect Detection**: Highlight potential defects using thermal analysis
- **Export Results**: Save enhanced images and comparison results

## Installation
1. Install Python 3.8 or higher
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main application:
   ```bash
   python main.py
   ```
2. Use the GUI to:
   - Select thermal image path
   - Select RGB image path (optional, for comparison)
   - Choose enhancement algorithm
   - View results and save enhanced images

## Project Structure
```
thermal_image_enhancer/
├── main.py                 # Main application entry point
├── setup.bat              # One-click setup for Windows
├── src/                    # Source code directory
│   ├── gui/               # GUI components
│   ├── enhancement/       # Image enhancement algorithms
│   ├── utils/            # Utility functions
│   └── analysis/         # Defect detection analysis
├── data/                  # Sample data and results
├── tests/                 # Unit tests
├── docs/                  # Documentation
└── requirements.txt       # Python dependencies
```

## Thermal Image Enhancement Algorithms
- **Histogram Equalization**: Improves contrast in thermal images
- **Adaptive Histogram Equalization**: Local contrast enhancement
- **Gaussian Filtering**: Noise reduction
- **Edge Enhancement**: Highlight defect boundaries
- **Temperature Mapping**: Convert to false-color representation

## Contributing
Feel free to contribute by adding new enhancement algorithms or improving the defect detection capabilities. 