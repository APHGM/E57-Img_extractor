import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pye57
import numpy as np
from PIL import Image
import os
import json

def explore_e57_structure(e57):
    """
    Explore the structure of an E57 file to understand what data is available
    """
    print("Exploring E57 file structure...")
    print(f"Root node type: {type(e57.root)}")
    
    # Try different methods to access children
    children = None
    print("\nTrying different methods to access file structure:")
    
    # Method 1: children_names
    if hasattr(e57.root, 'children_names'):
        try:
            children = e57.root.children_names
            print(f"Method 1 (children_names): Found {len(children)} children")
        except Exception as e:
            print(f"Method 1 failed: {e}")
    
    # Method 2: list() conversion
    if children is None and hasattr(e57.root, '__iter__'):
        try:
            children = list(e57.root)
            print(f"Method 2 (list()): Found {len(children)} children")
        except Exception as e:
            print(f"Method 2 failed: {e}")
    
    # Method 3: Try to access common E57 elements directly
    if children is None:
        print("Method 3: Trying direct access to common elements...")
        common_elements = ["data3D", "images", "images2D", "cameraImages", "visualReferenceRepresentation"]
        found_elements = []
        
        for element in common_elements:
            try:
                if hasattr(e57.root, '__getitem__'):
                    test_element = e57.root[element]
                    found_elements.append(element)
                    print(f"  Found '{element}': {type(test_element)}")
            except Exception:
                pass
        
        if found_elements:
            children = found_elements
            print(f"Method 3: Found {len(found_elements)} elements via direct access")
    
    # Method 4: Try to read the entire file structure
    if children is None:
        print("Method 4: Attempting to read file structure...")
        try:
            # Try to get all data from the file
            all_data = e57.read_scan(0)  # Try reading first scan
            print(f"Method 4: Successfully read scan data")
            children = ["data3D"]  # Assume data3D exists if we can read scans
        except Exception as e:
            print(f"Method 4 failed: {e}")
    
    # List all top-level elements
    print("\nTop-level elements:")
    if children is not None:
        for key in children:
            try:
                element_type = type(e57.root[key])
                print(f"  - {key}: {element_type}")
            except Exception as e:
                print(f"  - {key}: (could not get type: {e})")
    else:
        print("  (Could not list children of root node)")
    
    # Check for data3D scans
    if hasattr(e57.root, '__getitem__') and children and "data3D" in children:
        try:
            data3d = e57.root["data3D"]
            print(f"\nFound {len(data3d)} data3D scans")
            for i, scan in enumerate(data3d):
                print(f"  Scan {i}:")
                scan_children = None
                if hasattr(scan, 'children_names'):
                    scan_children = scan.children_names
                elif hasattr(scan, '__iter__'):
                    try:
                        scan_children = list(scan)
                    except Exception:
                        scan_children = None
                if scan_children is not None:
                    for key in scan_children:
                        try:
                            print(f"    - {key}: {type(scan[key])}")
                        except Exception:
                            print(f"    - {key}: (could not get type)")
                else:
                    print("    (Could not list children of scan node)")
        except Exception as e:
            print(f"Error accessing data3D: {e}")
    
    # Check for images
    if hasattr(e57.root, '__getitem__') and children and "images" in children:
        try:
            images = e57.root["images"]
            print(f"\nFound {len(images)} images")
        except Exception as e:
            print(f"Error accessing images: {e}")
    else:
        print("\nNo 'images' element found in root")
    
    # Check for images2D (spherical images)
    if hasattr(e57.root, '__getitem__') and children and "images2D" in children:
        try:
            images2d = e57.root["images2D"]
            print(f"\nFound {len(images2d)} images2D (spherical images)")
        except Exception as e:
            print(f"Error accessing images2D: {e}")
    else:
        print("\nNo 'images2D' element found in root")
    
    return e57.root, children

def try_direct_image_extraction(e57, output_folder):
    """
    Try to extract images using direct methods when structure exploration fails
    """
    print("\nTrying direct image extraction methods...")
    extracted_count = 0
    
    # Method 1: Try to read all scans and look for images
    try:
        scan_count = e57.scan_count
        print(f"Found {scan_count} scans in file")
        
        for scan_idx in range(scan_count):
            try:
                scan_data = e57.read_scan(scan_idx, ignore_missing_fields=True)
                print(f"Scan {scan_idx}: {type(scan_data)}")
                
                # Check if scan_data contains image information
                if hasattr(scan_data, 'keys'):
                    print(f"  Scan {scan_idx} fields:")
                    for key in scan_data.keys():
                        value = scan_data[key]
                        print(f"    - {key}: {type(value)}")
                        
                        # If this looks like image data, try to save it
                        if key in ['image', 'images', 'visualReferenceRepresentation', 'cameraImage', 'photo']:
                            try:
                                img_data = value
                                if hasattr(img_data, 'shape'):  # numpy array
                                    img_filename = f"scan_{scan_idx:03d}_{key}.jpg"
                                    Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                                    print(f"      Saved: {img_filename}")
                                    extracted_count += 1
                            except Exception as e:
                                print(f"      Error saving {key}: {e}")
                        
                        # Check if the value is a dictionary/object that might contain images
                        elif hasattr(value, 'keys') and isinstance(value, dict):
                            print(f"      Examining nested field {key}:")
                            for sub_key, sub_value in value.items():
                                print(f"        - {sub_key}: {type(sub_value)}")
                                if sub_key in ['image', 'images', 'visualReferenceRepresentation', 'cameraImage', 'photo']:
                                    try:
                                        if hasattr(sub_value, 'shape'):  # numpy array
                                            img_filename = f"scan_{scan_idx:03d}_{key}_{sub_key}.jpg"
                                            Image.fromarray(sub_value).save(os.path.join(output_folder, img_filename))
                                            print(f"          Saved: {img_filename}")
                                            extracted_count += 1
                                    except Exception as e:
                                        print(f"          Error saving {sub_key}: {e}")
                                
            except Exception as e:
                print(f"Error reading scan {scan_idx}: {e}")
                continue
                
    except Exception as e:
        print(f"Error accessing scan count: {e}")
    
    # Method 2: Try to access images directly if they exist
    try:
        if hasattr(e57.root, '__getitem__'):
            # Try common image element names
            image_elements = ['images', 'cameraImages', 'visualReferenceRepresentation', 'photos', 'cameraImage']
            for element_name in image_elements:
                try:
                    images = e57.root[element_name]
                    print(f"Found {len(images)} images in '{element_name}'")
                    
                    for i, img in enumerate(images):
                        try:
                            img_data = e57.read_image(img)
                            img_filename = f"direct_{element_name}_{i:03d}.jpg"
                            Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                            print(f"Saved: {img_filename}")
                            extracted_count += 1
                        except Exception as e:
                            print(f"Error processing image {i}: {e}")
                            continue
                            
                except Exception:
                    pass  # Element doesn't exist, try next
                    
    except Exception as e:
        print(f"Error in direct image access: {e}")
    
    # Method 3: Try to access scan structure directly
    try:
        if hasattr(e57.root, '__getitem__') and 'data3D' in e57.root:
            data3d = e57.root["data3D"]
            print(f"Examining data3D structure directly...")
            
            for scan_idx, scan in enumerate(data3d):
                try:
                    # Try to access scan elements directly
                    scan_children = None
                    if hasattr(scan, 'children_names'):
                        scan_children = scan.children_names
                    elif hasattr(scan, '__iter__'):
                        try:
                            scan_children = list(scan)
                        except Exception:
                            scan_children = None
                    
                    if scan_children:
                        print(f"  Scan {scan_idx} children:")
                        for child_name in scan_children:
                            try:
                                child_value = scan[child_name]
                                print(f"    - {child_name}: {type(child_value)}")
                                
                                # Check if this child contains images
                                if child_name in ['images', 'cameraImage', 'visualReferenceRepresentation', 'photo']:
                                    try:
                                        if hasattr(child_value, '__len__') and len(child_value) > 0:
                                            for img_idx, img in enumerate(child_value):
                                                try:
                                                    img_data = e57.read_image(img)
                                                    img_filename = f"scan_{scan_idx:03d}_{child_name}_{img_idx:03d}.jpg"
                                                    Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                                                    print(f"      Saved: {img_filename}")
                                                    extracted_count += 1
                                                except Exception as e:
                                                    print(f"      Error processing image {img_idx}: {e}")
                                        else:
                                            # Single image
                                            img_data = e57.read_image(child_value)
                                            img_filename = f"scan_{scan_idx:03d}_{child_name}.jpg"
                                            Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                                            print(f"      Saved: {img_filename}")
                                            extracted_count += 1
                                    except Exception as e:
                                        print(f"      Error processing {child_name}: {e}")
                                        
                            except Exception as e:
                                print(f"    - {child_name}: Error accessing - {e}")
                    else:
                        print(f"  Scan {scan_idx}: Could not list children")
                        
                except Exception as e:
                    print(f"Error examining scan {scan_idx}: {e}")
                    continue
                    
    except Exception as e:
        print(f"Error in data3D direct access: {e}")
    
    return extracted_count

def extract_images_from_e57(e57_path, output_folder=None):
    """
    Extracts all 2D images and their location data from an E57 file
    
    Args:
        e57_path (str): Path to input E57 file
        output_folder (str): Folder to save extracted images and metadata (optional)
    """
    
    # Create output folder if not provided
    if output_folder is None:
        base_name = os.path.splitext(os.path.basename(e57_path))[0]
        output_folder = f"{base_name}_extracted_images"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    
    # Open the E57 file
    e57 = pye57.E57(e57_path)
    
    # Explore the file structure first
    root, children = explore_e57_structure(e57)
    
    extracted_count = 0
    
    # Try to extract images from the "images" element if it exists
    if hasattr(root, '__getitem__') and children and "images" in children:
        images = root["images"]
        print(f"\nProcessing {len(images)} images from 'images' element...")
        
        for i, img in enumerate(images):
            try:
                # Extract image data
                img_data = e57.read_image(img)
                
                # Get image metadata
                img_name = img["name"].value()
                img_guid = img["guid"].value()
                img_mime = img["visualReferenceRepresentation"]["mimeType"].value()
                
                # Get pose information (position and orientation)
                pose = img["pose"]
                translation = pose["translation"]
                x = translation["x"].value()
                y = translation["y"].value()
                z = translation["z"].value()
                
                rotation = pose["rotation"]
                rotation_w = rotation["w"].value()
                rotation_x = rotation["x"].value()
                rotation_y = rotation["y"].value()
                rotation_z = rotation["z"].value()
                
                # Create output filename
                if img_mime == "image/jpeg":
                    ext = "jpg"
                elif img_mime == "image/png":
                    ext = "png"
                else:
                    ext = "bin"
                    
                img_filename = f"image_{i:03d}_{img_guid[:8]}.{ext}"
                meta_filename = f"image_{i:03d}_{img_guid[:8]}.json"
                
                # Save image
                if img_mime in ["image/jpeg", "image/png"]:
                    Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                else:
                    with open(os.path.join(output_folder, img_filename), "wb") as f:
                        f.write(img_data.tobytes())
                
                # Save metadata
                metadata = {
                    "guid": img_guid,
                    "name": img_name,
                    "mime_type": img_mime,
                    "position": {
                        "x": x,
                        "y": y,
                        "z": z
                    },
                    "orientation": {
                        "w": rotation_w,
                        "x": rotation_x,
                        "y": rotation_y,
                        "z": rotation_z
                    }
                }
                
                with open(os.path.join(output_folder, meta_filename), "w") as f:
                    json.dump(metadata, f, indent=4)
                    
                print(f"Saved image {i+1}/{len(images)}: {img_filename}")
                extracted_count += 1
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue
    
    # Try to extract images from images2D (spherical images)
    if hasattr(root, '__getitem__') and children and "images2D" in children:
        images2d = root["images2D"]
        print(f"\nProcessing {len(images2d)} spherical images from 'images2D' element...")
        
        for i, img in enumerate(images2d):
            try:
                # Get image metadata
                img_name = img["name"].value()
                img_guid = img["guid"].value()
                
                # Get pose information (position and orientation)
                pose = img["pose"]
                translation = pose["translation"]
                x = translation["x"].value()
                y = translation["y"].value()
                z = translation["z"].value()
                
                rotation = pose["rotation"]
                rotation_w = rotation["w"].value()
                rotation_x = rotation["x"].value()
                rotation_y = rotation["y"].value()
                rotation_z = rotation["z"].value()
                
                # Get spherical representation data
                spherical = img["sphericalRepresentation"]
                # Read the JPEG blob data directly
                jpeg_blob = spherical["jpegImage"]
                
                # Get blob length and read the data
                blob_length = jpeg_blob.byteCount()
                img_data = bytearray(blob_length)
                jpeg_blob.read(img_data, 0, blob_length)
                img_data = bytes(img_data)
                
                image_height = spherical["imageHeight"].value()
                image_width = spherical["imageWidth"].value()
                pixel_height = spherical["pixelHeight"].value()
                pixel_width = spherical["pixelWidth"].value()
                
                # Create output filename
                img_filename = f"sphere_{i:03d}_{img_guid[:8]}.jpg"
                meta_filename = f"sphere_{i:03d}_{img_guid[:8]}.json"
                
                # Save image (raw bytes as JPEG)
                with open(os.path.join(output_folder, img_filename), "wb") as f:
                    f.write(img_data)
                
                # Save metadata
                metadata = {
                    "guid": img_guid,
                    "name": img_name,
                    "type": "spherical",
                    "position": {
                        "x": x,
                        "y": y,
                        "z": z
                    },
                    "orientation": {
                        "w": rotation_w,
                        "x": rotation_x,
                        "y": rotation_y,
                        "z": rotation_z
                    },
                    "spherical_data": {
                        "image_height": image_height,
                        "image_width": image_width,
                        "pixel_height": pixel_height,
                        "pixel_width": pixel_width
                    }
                }
                
                with open(os.path.join(output_folder, meta_filename), "w") as f:
                    json.dump(metadata, f, indent=4)
                    
                print(f"Saved spherical image {i+1}/{len(images2d)}: {img_filename}")
                extracted_count += 1
                
            except Exception as e:
                print(f"Error processing spherical image {i}: {str(e)}")
                continue
    
    # Try to extract images from data3D scans if no images were found
    if extracted_count == 0 and hasattr(root, '__getitem__') and children and "data3D" in children:
        print("\nNo images found in 'images' element. Checking data3D scans for images...")
        
        data3d = root["data3D"]
        for scan_idx, scan in enumerate(data3d):
            try:
                # Check if this scan has images
                scan_children = None
                if hasattr(scan, 'children_names'):
                    scan_children = scan.children_names
                elif hasattr(scan, '__iter__'):
                    try:
                        scan_children = list(scan)
                    except Exception:
                        scan_children = None
                
                if scan_children and "images" in scan_children:
                    scan_images = scan["images"]
                    print(f"Found {len(scan_images)} images in scan {scan_idx}")
                    
                    for img_idx, img in enumerate(scan_images):
                        try:
                            # Extract image data
                            img_data = e57.read_image(img)
                            
                            # Get image metadata
                            img_name = img["name"].value() if "name" in img else f"scan_{scan_idx}_img_{img_idx}"
                            img_guid = img["guid"].value() if "guid" in img else f"scan_{scan_idx}_img_{img_idx}"
                            
                            # Get MIME type
                            if "visualReferenceRepresentation" in img and "mimeType" in img["visualReferenceRepresentation"]:
                                img_mime = img["visualReferenceRepresentation"]["mimeType"].value()
                            else:
                                img_mime = "image/jpeg"  # Default assumption
                            
                            # Get pose information if available
                            pose_data = {}
                            if "pose" in img:
                                pose = img["pose"]
                                if "translation" in pose:
                                    translation = pose["translation"]
                                    pose_data["position"] = {
                                        "x": translation["x"].value() if "x" in translation else 0,
                                        "y": translation["y"].value() if "y" in translation else 0,
                                        "z": translation["z"].value() if "z" in translation else 0
                                    }
                                
                                if "rotation" in pose:
                                    rotation = pose["rotation"]
                                    pose_data["orientation"] = {
                                        "w": rotation["w"].value() if "w" in rotation else 1,
                                        "x": rotation["x"].value() if "x" in rotation else 0,
                                        "y": rotation["y"].value() if "y" in rotation else 0,
                                        "z": rotation["z"].value() if "z" in rotation else 0
                                    }
                            
                            # Create output filename
                            if img_mime == "image/jpeg":
                                ext = "jpg"
                            elif img_mime == "image/png":
                                ext = "png"
                            else:
                                ext = "bin"
                                
                            img_filename = f"scan_{scan_idx:03d}_img_{img_idx:03d}_{img_guid[:8]}.{ext}"
                            meta_filename = f"scan_{scan_idx:03d}_img_{img_idx:03d}_{img_guid[:8]}.json"
                            
                            # Save image
                            if img_mime in ["image/jpeg", "image/png"]:
                                Image.fromarray(img_data).save(os.path.join(output_folder, img_filename))
                            else:
                                with open(os.path.join(output_folder, img_filename), "wb") as f:
                                    f.write(img_data.tobytes())
                            
                            # Save metadata
                            metadata = {
                                "scan_index": scan_idx,
                                "image_index": img_idx,
                                "guid": img_guid,
                                "name": img_name,
                                "mime_type": img_mime,
                                **pose_data
                            }
                            
                            with open(os.path.join(output_folder, meta_filename), "w") as f:
                                json.dump(metadata, f, indent=4)
                                
                            print(f"Saved image from scan {scan_idx}, image {img_idx}: {img_filename}")
                            extracted_count += 1
                            
                        except Exception as e:
                            print(f"Error processing image {img_idx} in scan {scan_idx}: {str(e)}")
                            continue
                            
            except Exception as e:
                print(f"Error processing scan {scan_idx}: {str(e)}")
                continue
    
    if extracted_count == 0:
        print("\nNo images found in the E57 file.")
        print("The file may not contain 2D images, or they may be stored in an unexpected location.")
        print("Available data types in this E57 file:")
        if children is not None:
            for key in children:
                print(f"  - {key}")
        else:
            print("  (Could not list children of root node)")
        
        # Try direct extraction methods as a fallback
        direct_count = try_direct_image_extraction(e57, output_folder)
        if direct_count > 0:
            print(f"\nSuccessfully extracted {direct_count} images using direct methods!")
        else:
            print("\nNo images could be extracted using any method.")
    else:
        print(f"\nSuccessfully extracted {extracted_count} images to {output_folder}")

# [Previous functions: explore_e57_structure, try_direct_image_extraction, extract_images_from_e57 remain exactly the same]

class E57ImageExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("E57 Image Extractor")
        self.root.geometry("600x400")
        
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabel', padding=5)
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input file selection
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input E57 File", padding="10")
        self.input_frame.pack(fill=tk.X, pady=5)
        
        self.input_entry = ttk.Entry(self.input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.browse_button = ttk.Button(
            self.input_frame, 
            text="Browse...", 
            command=self.browse_input_file
        )
        self.browse_button.pack(side=tk.RIGHT)
        
        # Output folder selection
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output Folder", padding="10")
        self.output_frame.pack(fill=tk.X, pady=5)
        
        self.output_entry = ttk.Entry(self.output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.output_browse_button = ttk.Button(
            self.output_frame, 
            text="Browse...", 
            command=self.browse_output_folder
        )
        self.output_browse_button.pack(side=tk.RIGHT)
        
        # Progress and log
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Progress", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = tk.Text(self.log_frame, height=10, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.scrollbar = ttk.Scrollbar(self.log_text)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.log_text.yview)
        
        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        
        self.extract_button = ttk.Button(
            self.button_frame, 
            text="Extract Images", 
            command=self.start_extraction,
            state=tk.NORMAL
        )
        self.extract_button.pack(side=tk.LEFT, padx=5)
        
        self.clear_button = ttk.Button(
            self.button_frame, 
            text="Clear Log", 
            command=self.clear_log
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        self.exit_button = ttk.Button(
            self.button_frame, 
            text="Exit", 
            command=self.root.quit
        )
        self.exit_button.pack(side=tk.RIGHT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(
            self.main_frame, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Redirect stdout to log
        self.redirect_stdout()
    
    def redirect_stdout(self):
        """Redirect stdout to the log text widget"""
        class StdoutRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, message):
                self.text_widget.insert(tk.END, message)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
            
            def flush(self):
                pass
        
        import sys
        sys.stdout = StdoutRedirector(self.log_text)
    
    def browse_input_file(self):
        """Open file dialog to select E57 file"""
        file_path = filedialog.askopenfilename(
            title="Select E57 File",
            filetypes=[("E57 Files", "*.e57"), ("All Files", "*.*")]
        )
        if file_path:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, file_path)
            
            # Suggest output folder based on input file name
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            suggested_folder = f"{base_name}_images"
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, suggested_folder)
    
    def browse_output_folder(self):
        """Open folder dialog to select output directory and create basename subfolder"""
        # Get the initial directory from the input file if it exists
        initial_dir = None
        if self.input_entry.get():
            input_dir = os.path.dirname(self.input_entry.get())
            if os.path.exists(input_dir):
                initial_dir = input_dir
        
        # Let user select parent folder
        parent_folder = filedialog.askdirectory(
            title="Select Parent Folder for Output",
            initialdir=initial_dir
        )
        
        if parent_folder:  # Only proceed if a folder was selected
            # Get basename from input file or use default
            if self.input_entry.get():
                base_name = os.path.splitext(os.path.basename(self.input_entry.get()))[0]
                folder_name = f"{base_name}_images"
            else:
                folder_name = "extracted_images"
            
            # Create full output path
            output_folder = os.path.join(parent_folder, folder_name)
            
            # Create the directory if it doesn't exist
            try:
                os.makedirs(output_folder, exist_ok=True)
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, output_folder)
                print(f"Created output folder: {output_folder}")
            except Exception as e:
                print(f"Error creating output folder: {e}")
                messagebox.showerror("Error", f"Could not create output folder:\n{e}")
    
    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.delete(1.0, tk.END)
    
    def start_extraction(self):
        """Start the extraction process in a separate thread"""
        input_file = self.input_entry.get()
        output_folder = self.output_entry.get()
        
        if not input_file:
            messagebox.showerror("Error", "Please select an input E57 file")
            return
        
        if not output_folder:
            messagebox.showerror("Error", "Please select an output folder")
            return
        
        # Disable buttons during extraction
        self.extract_button.config(state=tk.DISABLED)
        self.browse_button.config(state=tk.DISABLED)
        self.output_browse_button.config(state=tk.DISABLED)
        
        # Start extraction in a separate thread
        self.status_var.set("Extracting images...")
        
        def extraction_thread():
            try:
                extract_images_from_e57(input_file, output_folder)
            except Exception as e:
                print(f"Error during extraction: {str(e)}")
            finally:
                # Re-enable buttons when done
                self.root.after(0, lambda: self.extract_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.browse_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.output_browse_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Extraction complete"))
        
        thread = threading.Thread(target=extraction_thread, daemon=True)
        thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    app = E57ImageExtractorApp(root)
    
    # Set application icon (optional)
    try:
        root.iconbitmap(default='extractor_icon.ico')  # Provide your own icon file
    except:
        pass
    
    root.mainloop()
