#!/usr/bin/env python3
"""
Launcher script for Thermal Image Enhancement Application
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'opencv-python',
        'numpy',
        'matplotlib',
        'Pillow',
        'scikit-image',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main launcher function."""
    print("Thermal Image Enhancement for Defect Detection")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found!")
        print("Please make sure you're running this script from the project root directory.")
        sys.exit(1)
    
    # Run the main application
    try:
        print("Starting application...")
        subprocess.run([sys.executable, 'main.py'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    main() 