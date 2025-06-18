"""
Main GUI Window for Thermal Image Enhancement Application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import cv2
from PIL import Image, ImageTk
import os

from enhancement.thermal_enhancer import ThermalImageEnhancer
from analysis.defect_detector import DefectDetector
from utils.image_utils import ImageUtils

class ThermalImageEnhancerGUI:
    def __init__(self, root):
        self.root = root
        self.thermal_enhancer = ThermalImageEnhancer()
        self.defect_detector = DefectDetector()
        self.image_utils = ImageUtils()
        
        # Image data
        self.thermal_image = None
        self.rgb_image = None
        self.enhanced_image = None
        
        # GUI variables
        self.thermal_path_var = tk.StringVar()
        self.rgb_path_var = tk.StringVar()
        self.enhancement_method_var = tk.StringVar(value="histogram_equalization")
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the main user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Thermal Image Enhancement for Defect Detection", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # File selection frame
        self.create_file_selection_frame(main_frame)
        
        # Control frame
        self.create_control_frame(main_frame)
        
        # Image display frame
        self.create_image_display_frame(main_frame)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def create_file_selection_frame(self, parent):
        """Create the file selection controls."""
        file_frame = ttk.LabelFrame(parent, text="Image Selection", padding="10")
        file_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # Thermal image selection
        ttk.Label(file_frame, text="Thermal Image:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        thermal_entry = ttk.Entry(file_frame, textvariable=self.thermal_path_var, width=50)
        thermal_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(file_frame, text="Browse", command=self.browse_thermal_image).grid(row=0, column=2)
        
        # RGB image selection
        ttk.Label(file_frame, text="RGB Image (Optional):").grid(row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(10, 0))
        rgb_entry = ttk.Entry(file_frame, textvariable=self.rgb_path_var, width=50)
        rgb_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 10), pady=(10, 0))
        ttk.Button(file_frame, text="Browse", command=self.browse_rgb_image).grid(row=1, column=2, pady=(10, 0))
        
    def create_control_frame(self, parent):
        """Create the enhancement control panel."""
        control_frame = ttk.LabelFrame(parent, text="Enhancement Controls", padding="10")
        control_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))
        
        # Enhancement method selection
        ttk.Label(control_frame, text="Enhancement Method:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        methods = [
            "histogram_equalization",
            "adaptive_histogram_equalization", 
            "gaussian_filtering",
            "edge_enhancement",
            "temperature_mapping",
            "multi_scale_enhancement"
        ]
        method_combo = ttk.Combobox(control_frame, textvariable=self.enhancement_method_var, 
                                   values=methods, state="readonly", width=25)
        method_combo.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Enhancement buttons
        ttk.Button(control_frame, text="Enhance Thermal Image", 
                  command=self.enhance_thermal_image).grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="Detect Defects", 
                  command=self.detect_defects).grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="Compare Images", 
                  command=self.compare_images).grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(control_frame, text="Save Enhanced Image", 
                  command=self.save_enhanced_image).grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
    def create_image_display_frame(self, parent):
        """Create the image display area."""
        display_frame = ttk.LabelFrame(parent, text="Image Display", padding="10")
        display_frame.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure for image display
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, display_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Set titles
        self.ax1.set_title("Original Thermal Image")
        self.ax2.set_title("Enhanced Image")
        
        # Initial display
        self.update_image_display()
        
    def browse_thermal_image(self):
        """Browse for thermal image file."""
        file_path = filedialog.askopenfilename(
            title="Select Thermal Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.thermal_path_var.set(file_path)
            self.load_thermal_image()
            
    def browse_rgb_image(self):
        """Browse for RGB image file."""
        file_path = filedialog.askopenfilename(
            title="Select RGB Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.rgb_path_var.set(file_path)
            self.load_rgb_image()
            
    def load_thermal_image(self):
        """Load the thermal image from file."""
        try:
            file_path = self.thermal_path_var.get()
            if file_path and os.path.exists(file_path):
                self.thermal_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if self.thermal_image is not None:
                    self.status_var.set(f"Thermal image loaded: {os.path.basename(file_path)}")
                    self.update_image_display()
                else:
                    messagebox.showerror("Error", "Failed to load thermal image")
            else:
                messagebox.showerror("Error", "Please select a valid thermal image file")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading thermal image: {e}")
            
    def load_rgb_image(self):
        """Load the RGB image from file."""
        try:
            file_path = self.rgb_path_var.get()
            if file_path and os.path.exists(file_path):
                self.rgb_image = cv2.imread(file_path)
                if self.rgb_image is not None:
                    self.rgb_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2RGB)
                    self.status_var.set(f"RGB image loaded: {os.path.basename(file_path)}")
                else:
                    messagebox.showerror("Error", "Failed to load RGB image")
            else:
                messagebox.showerror("Error", "Please select a valid RGB image file")
        except Exception as e:
            messagebox.showerror("Error", f"Error loading RGB image: {e}")
            
    def enhance_thermal_image(self):
        """Enhance the thermal image using selected method."""
        if self.thermal_image is None:
            messagebox.showwarning("Warning", "Please load a thermal image first")
            return
            
        try:
            method = self.enhancement_method_var.get()
            self.status_var.set(f"Enhancing image using {method}...")
            self.root.update()
            
            self.enhanced_image = self.thermal_enhancer.enhance(
                self.thermal_image, method
            )
            
            self.status_var.set(f"Image enhanced using {method}")
            self.update_image_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error enhancing image: {e}")
            self.status_var.set("Enhancement failed")
            
    def detect_defects(self):
        """Detect defects in the thermal image."""
        if self.thermal_image is None:
            messagebox.showwarning("Warning", "Please load a thermal image first")
            return
            
        try:
            self.status_var.set("Detecting defects...")
            self.root.update()
            
            # Use enhanced image if available, otherwise use original
            image_to_analyze = self.enhanced_image if self.enhanced_image is not None else self.thermal_image
            
            defects = self.defect_detector.detect_defects(image_to_analyze)
            
            if defects:
                self.status_var.set(f"Found {len(defects)} potential defects")
                # Highlight defects on the image
                self.highlight_defects(defects)
            else:
                self.status_var.set("No defects detected")
                messagebox.showinfo("Defect Detection", "No defects were detected in the image")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting defects: {e}")
            self.status_var.set("Defect detection failed")
            
    def compare_images(self):
        """Compare thermal and RGB images."""
        if self.thermal_image is None:
            messagebox.showwarning("Warning", "Please load a thermal image first")
            return
            
        if self.rgb_image is None:
            messagebox.showwarning("Warning", "Please load an RGB image for comparison")
            return
            
        try:
            self.status_var.set("Comparing images...")
            self.root.update()
            
            # Create comparison window
            self.create_comparison_window()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error comparing images: {e}")
            self.status_var.set("Comparison failed")
            
    def save_enhanced_image(self):
        """Save the enhanced image."""
        if self.enhanced_image is None:
            messagebox.showwarning("Warning", "No enhanced image to save")
            return
            
        try:
            file_path = filedialog.asksaveasfilename(
                title="Save Enhanced Image",
                defaultextension=".png",
                filetypes=[
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("All files", "*.*")
                ]
            )
            
            if file_path:
                cv2.imwrite(file_path, self.enhanced_image)
                self.status_var.set(f"Enhanced image saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Enhanced image saved successfully")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error saving image: {e}")
            
    def update_image_display(self):
        """Update the image display."""
        self.ax1.clear()
        self.ax2.clear()
        
        # Display original thermal image
        if self.thermal_image is not None:
            self.ax1.imshow(self.thermal_image, cmap='hot')
            self.ax1.set_title("Original Thermal Image")
            self.ax1.axis('off')
        
        # Display enhanced image
        if self.enhanced_image is not None:
            self.ax2.imshow(self.enhanced_image, cmap='hot')
            self.ax2.set_title("Enhanced Image")
            self.ax2.axis('off')
        else:
            self.ax2.set_title("Enhanced Image (Not Available)")
            self.ax2.axis('off')
            
        self.canvas.draw()
        
    def highlight_defects(self, defects):
        """Highlight detected defects on the image."""
        if self.enhanced_image is not None:
            image_to_highlight = self.enhanced_image.copy()
        else:
            image_to_highlight = self.thermal_image.copy()
            
        # Convert to RGB for highlighting
        if len(image_to_highlight.shape) == 2:
            image_to_highlight = cv2.cvtColor(image_to_highlight, cv2.COLOR_GRAY2RGB)
            
        # Draw rectangles around defects
        for defect in defects:
            x, y, w, h = defect
            cv2.rectangle(image_to_highlight, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
        # Update display
        self.ax2.clear()
        self.ax2.imshow(image_to_highlight)
        self.ax2.set_title("Enhanced Image with Defects Highlighted")
        self.ax2.axis('off')
        self.canvas.draw()
        
    def create_comparison_window(self):
        """Create a new window for image comparison."""
        comparison_window = tk.Toplevel(self.root)
        comparison_window.title("Image Comparison")
        comparison_window.geometry("1000x600")
        
        # Create comparison figure
        comp_fig, (comp_ax1, comp_ax2) = plt.subplots(1, 2, figsize=(10, 5))
        comp_canvas = FigureCanvasTkAgg(comp_fig, comparison_window)
        comp_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Display thermal image
        if self.enhanced_image is not None:
            comp_ax1.imshow(self.enhanced_image, cmap='hot')
            comp_ax1.set_title("Enhanced Thermal Image")
        else:
            comp_ax1.imshow(self.thermal_image, cmap='hot')
            comp_ax1.set_title("Original Thermal Image")
        comp_ax1.axis('off')
        
        # Display RGB image
        comp_ax2.imshow(self.rgb_image)
        comp_ax2.set_title("RGB Image")
        comp_ax2.axis('off')
        
        comp_canvas.draw() 