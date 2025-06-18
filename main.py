#!/usr/bin/env python3
"""
Thermal Image Enhancement Application
Main entry point for the thermal image enhancement and defect detection tool.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gui.main_window import ThermalImageEnhancerGUI

def main():
    """Main application entry point."""
    try:
        # Create the main application window
        root = tk.Tk()
        root.title("Thermal Image Enhancement for Defect Detection")
        root.geometry("1200x800")
        
        # Set application icon and styling
        style = ttk.Style()
        style.theme_use('clam')
        
        # Initialize and run the GUI
        app = ThermalImageEnhancerGUI(root)
        root.mainloop()
        
    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Error", f"Failed to start application: {e}")

if __name__ == "__main__":
    main() 