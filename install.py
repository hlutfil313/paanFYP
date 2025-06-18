#!/usr/bin/env python3
"""
Installation script for Thermal Image Enhancement Application
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def generate_sample_data():
    """Generate sample data for testing."""
    print("Generating sample data...")
    try:
        # Add src to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        from utils.sample_generator import SampleGenerator
        
        generator = SampleGenerator()
        generator.generate_sample_dataset(num_samples=3)
        print("Sample data generated successfully!")
        return True
    except Exception as e:
        print(f"Error generating sample data: {e}")
        return False

def run_tests():
    """Run unit tests."""
    print("Running unit tests...")
    try:
        subprocess.check_call([sys.executable, "-m", "unittest", "discover", "tests"])
        print("Tests passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Some tests failed: {e}")
        return False

def main():
    """Main installation function."""
    print("Thermal Image Enhancement Application - Installation")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required!")
        sys.exit(1)
    
    print(f"Python version: {sys.version}")
    
    # Install dependencies
    if not install_dependencies():
        print("Failed to install dependencies. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Generate sample data
    if not generate_sample_data():
        print("Failed to generate sample data.")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("Some tests failed, but installation can continue.")
    
    print("\nInstallation completed successfully!")
    print("\nTo run the application:")
    print("  python main.py")
    print("  or")
    print("  python run.py")
    print("\nSample images are available in the data/samples/ directory.")

if __name__ == "__main__":
    main() 