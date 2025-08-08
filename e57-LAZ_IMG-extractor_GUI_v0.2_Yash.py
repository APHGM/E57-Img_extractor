import gc
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import pye57
import numpy as np
from PIL import Image
import os
import json
import glob
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import psutil

import gc


def inspect_e57_fields(e57_path):
    """
    Simple function to inspect all fields in an E57 file
    """
    try:
        e57 = pye57.E57(e57_path)
        scan_count = e57.scan_count
        print(f"File: {os.path.basename(e57_path)}")
        print(f"Scans: {scan_count}")
        for scan_idx in range(min(scan_count, 3)):  # Check first 3 scans
            print(f"\n=== SCAN {scan_idx} FIELDS ===")
            try:
                scan_data = e57.read_scan(scan_idx, ignore_missing_fields=True)
                print(f"Available fields: {list(scan_data.keys())}")
                for field_name in scan_data.keys():
                    field_data = scan_data[field_name]
                    try:
                        if hasattr(field_data, '__len__') and len(field_data) > 0:
                            field_array = np.array(field_data)
                            print(f"  {field_name}:")
                            print(f"    Length: {len(field_data)}")
                            print(f"    Type: {type(field_data)}")
                            print(f"    DType: {field_array.dtype}")
                            if np.issubdtype(field_array.dtype, np.number):
                                print(
                                    f"    Range: {field_array.min()} to {field_array.max()}")
                                print(f"    Sample: {field_array[:5]}")
                        else:
                            print(f"  {field_name}: Empty or no length")
                    except Exception as e:
                        print(f"  {field_name}: Error - {e}")
            except Exception as e:
                print(f"Error reading scan {scan_idx}: {e}")
    except Exception as e:
        print(f"Error opening E57 file: {e}")


def extract_point_cloud_to_laz_streaming(e57_path, output_folder=None, max_points_per_chunk=2_000_000, max_workers=4):
    """
    Memory-optimized version that streams data directly to disk instead of loading everything into RAM
    Args:
        e57_path (str): Path to input E57 file
        output_folder (str): Folder to save LAZ file (optional)
        max_points_per_chunk (int): Maximum number of points per chunk (default: 2M)
        max_workers (int): Maximum number of worker processes (default: 4)
    """
    import os
    import tempfile
    import shutil
    import numpy as np
    try:
        import laspy
        import pye57
    except ImportError as e:
        print(f"Error: Required package missing - {e}")
        print("Install with: pip install laspy pye57")
        return False

    if output_folder is None:
        output_folder = os.path.dirname(e57_path)

    # Create output filename
    base_name = os.path.splitext(os.path.basename(e57_path))[0]
    laz_path = os.path.join(output_folder, f"{base_name}.laz")

    # Conservative settings for memory management
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    # Adjust chunk size based on available memory
    if available_memory_gb < 2:
        max_points_per_chunk = min(max_points_per_chunk, 500_000)  # 500k points for low memory
        max_workers = min(max_workers, 2)
    elif available_memory_gb < 4:
        max_points_per_chunk = min(max_points_per_chunk, 1_000_000)  # 1M points for medium memory
        max_workers = min(max_workers, 3)
    
    optimal_workers = min(max_workers, cpu_count(), 4)  # Cap at 4 workers to save memory

    print(f"\nMemory-Optimized E57 to LAZ Conversion")
    print(f"Input: {os.path.basename(e57_path)}")
    print(f"Output: {laz_path}")
    print(f"Available memory: {available_memory_gb:.1f} GB")
    print(f"Max points per chunk: {max_points_per_chunk:,}")
    print(f"Using {optimal_workers} worker processes (memory-conservative)")

    # Create temporary directory for chunks
    tmp_dir = tempfile.mkdtemp(prefix="e57_chunks_")
    chunk_files = []

    try:
        # Use pye57 to extract point cloud data
        e57 = pye57.E57(e57_path)
        scan_count = e57.scan_count
        print(f"Found {scan_count} scans in E57 file")

        total_points_processed = 0
        chunk_index = 0

        print(f"\n=== STREAMING EXTRACTION (Memory-Optimized) ===")

        # Process each scan individually without loading all data
        for scan_idx in range(scan_count):
            print(f"\nProcessing scan {scan_idx + 1}/{scan_count}...")

            # Force garbage collection between scans
            gc.collect()
            
            # Check memory before processing each scan
            current_memory_gb = psutil.virtual_memory().available / (1024**3)
            print(f"  Available memory: {current_memory_gb:.1f} GB")

            try:
                # Read scan data
                scan_data = e57.read_scan(scan_idx, ignore_missing_fields=True, intensity=True, colors=True)

                # Extract coordinate data - try different possible field names
                x_data = None
                y_data = None  
                z_data = None

                coord_names = [
                    ('cartesianX', 'cartesianY', 'cartesianZ'),
                    ('x', 'y', 'z'),
                    ('X', 'Y', 'Z')
                ]

                for x_name, y_name, z_name in coord_names:
                    if x_name in scan_data and y_name in scan_data and z_name in scan_data:
                        x_data = scan_data[x_name]
                        y_data = scan_data[y_name]
                        z_data = scan_data[z_name]
                        print(f"  Found coordinates using fields: {x_name}, {y_name}, {z_name}")
                        break

                if x_data is None or y_data is None or z_data is None:
                    print(f"  No coordinate data found in scan {scan_idx}")
                    continue

                n_pts = len(x_data)
                print(f"  Found {n_pts:,} points in scan {scan_idx}")

                # Extract additional data
                intensity_data = scan_data.get("intensity", [0] * n_pts)
                red_data = scan_data.get("colorRed", [0] * n_pts)
                green_data = scan_data.get("colorGreen", [0] * n_pts)
                blue_data = scan_data.get("colorBlue", [0] * n_pts)

                # Process in small batches to avoid memory buildup
                batch_size = min(50_000, max_points_per_chunk // 20)  # Very small batches
                current_chunk_points = []

                print(f"  Processing in micro-batches of {batch_size:,} points")

                for start_idx in range(0, n_pts, batch_size):
                    end_idx = min(start_idx + batch_size, n_pts)
                    
                    if start_idx % (batch_size * 10) == 0:  # Progress every 10 batches
                        print(f"    Processing batch: points {start_idx:,} to {end_idx:,}")

                    # Extract only current batch data
                    batch_points = []
                    for i in range(start_idx, end_idx):
                        try:
                            # Handle different data types safely
                            def safe_float(data, idx):
                                try:
                                    if isinstance(data, (list, tuple)):
                                        return float(data[idx]) if data[idx] is not None else 0.0
                                    elif hasattr(data, '__getitem__'):
                                        return float(data[idx]) if data[idx] is not None else 0.0
                                    else:
                                        return 0.0
                                except (IndexError, ValueError, TypeError):
                                    return 0.0

                            def safe_int(data, idx):
                                try:
                                    if isinstance(data, (list, tuple)):
                                        val = data[idx] if data[idx] is not None else 0
                                    elif hasattr(data, '__getitem__'):
                                        val = data[idx] if data[idx] is not None else 0
                                    else:
                                        val = 0
                                    return int(val)
                                except (IndexError, ValueError, TypeError):
                                    return 0

                            point = {
                                "x": safe_float(x_data, i),
                                "y": safe_float(y_data, i),
                                "z": safe_float(z_data, i),
                                "intensity": safe_int(intensity_data, i),
                                "red": safe_int(red_data, i),
                                "green": safe_int(green_data, i),
                                "blue": safe_int(blue_data, i)
                            }
                            batch_points.append(point)
                        except Exception as point_error:
                            continue  # Skip invalid points

                    # Add batch to current chunk
                    current_chunk_points.extend(batch_points)

                    # Write chunk immediately when it reaches max size
                    if len(current_chunk_points) >= max_points_per_chunk:
                        chunk_file = os.path.join(tmp_dir, f"{base_name}_chunk_{chunk_index:03d}.laz")
                        write_chunk_to_laz_immediate(current_chunk_points, chunk_file)
                        chunk_files.append(chunk_file)
                        chunk_index += 1
                        total_points_processed += len(current_chunk_points)
                        print(f"      ✓ Wrote chunk {chunk_index} with {len(current_chunk_points):,} points")
                        current_chunk_points.clear()  # Clear immediately to free memory
                        gc.collect()  # Force garbage collection

                    # Clear batch data immediately
                    del batch_points
                    
                    # Periodic garbage collection during processing
                    if start_idx % (batch_size * 20) == 0:
                        gc.collect()

                # Write remaining points as final chunk for this scan
                if current_chunk_points:
                    chunk_file = os.path.join(tmp_dir, f"{base_name}_chunk_{chunk_index:03d}.laz")
                    write_chunk_to_laz_immediate(current_chunk_points, chunk_file)
                    chunk_files.append(chunk_file)
                    chunk_index += 1
                    total_points_processed += len(current_chunk_points)
                    print(f"      ✓ Wrote final chunk {chunk_index} with {len(current_chunk_points):,} points")
                    current_chunk_points.clear()
                    gc.collect()

                print(f"  ✓ Completed scan {scan_idx}")

                # Clear scan data immediately after processing
                del scan_data, x_data, y_data, z_data, intensity_data, red_data, green_data, blue_data
                gc.collect()

            except Exception as scan_error:
                print(f"  Error processing scan {scan_idx}: {scan_error}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n=== EXTRACTION COMPLETE ===")
        print(f"Total points processed: {total_points_processed:,}")
        print(f"Total chunks created: {len(chunk_files)}")

        if not chunk_files:
            print("❌ No chunks were created - no point cloud data found")
            return False

        # Merge chunks using streaming method
        print(f"\n=== MERGING CHUNKS ===")
        if len(chunk_files) == 1:
            # Only one chunk, just copy it
            shutil.copy(chunk_files[0], laz_path)
            print(f"✓ Saved single chunk to: {laz_path}")
        else:
            # Merge multiple chunks using streaming
            merge_laz_chunks_streaming(chunk_files, laz_path)

        # Get file size and final info
        if os.path.exists(laz_path):
            file_size = os.path.getsize(laz_path)
            size_mb = file_size / (1024 * 1024)
            print(f"\n✓ SUCCESS!")
            print(f"  Final file: {laz_path}")
            print(f"  Total points: {total_points_processed:,}")
            print(f"  File size: {size_mb:.1f} MB")
            print(f"  Chunks processed: {len(chunk_files)}")
            return True
        else:
            print("❌ Final LAZ file was not created")
            return False

    except Exception as e:
        print(f"❌ Error exporting point cloud: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {tmp_dir}")
        except:
            pass


def write_chunk_to_laz_immediate(chunk_data, chunk_file):
    """
    Write chunk data immediately to LAZ file without storing in memory
    This function processes data in small batches to minimize RAM usage
    """
    import laspy
    import numpy as np
    
    if not chunk_data:
        return
    
    total_points = len(chunk_data)
    
    # Process in smaller sub-batches to avoid memory spikes
    sub_batch_size = min(25_000, total_points)  # Process 25k points at a time max
    
    # Pre-allocate arrays
    x_data = np.empty(total_points, dtype=np.float64)
    y_data = np.empty(total_points, dtype=np.float64)
    z_data = np.empty(total_points, dtype=np.float64)
    intensity_data = np.empty(total_points, dtype=np.uint16)
    red_data = np.empty(total_points, dtype=np.uint16)
    green_data = np.empty(total_points, dtype=np.uint16)
    blue_data = np.empty(total_points, dtype=np.uint16)
    
    # Fill arrays in sub-batches to control memory usage
    for i in range(0, total_points, sub_batch_size):
        end_i = min(i + sub_batch_size, total_points)
        
        for j in range(i, end_i):
            point = chunk_data[j]
            x_data[j] = point["x"]
            y_data[j] = point["y"]
            z_data[j] = point["z"]
            
            # Handle intensity conversion
            intensity_val = point.get("intensity", 0)
            if intensity_val <= 1.0:
                intensity_data[j] = int(intensity_val * 65535)
            elif intensity_val <= 255:
                intensity_data[j] = int(intensity_val * 256)
            else:
                intensity_data[j] = min(65535, max(0, int(intensity_val)))
            
            # Handle color conversion
            for color_name, color_array in [("red", red_data), ("green", green_data), ("blue", blue_data)]:
                color_val = point.get(color_name, 0)
                if color_val <= 1.0:
                    color_array[j] = int(color_val * 65535)
                elif color_val <= 255:
                    color_array[j] = int(color_val * 257)
                else:
                    color_array[j] = min(65535, max(0, int(color_val)))
        
        # Periodic garbage collection during array filling
        if i % (sub_batch_size * 4) == 0:
            gc.collect()
    
    try:
        # Determine point format based on available data
        has_color = np.any(red_data > 0) or np.any(green_data > 0) or np.any(blue_data > 0)
        has_intensity = np.any(intensity_data > 0)
        
        if has_color and has_intensity:
            point_format = 3  # XYZ + RGB + Intensity
        elif has_color:
            point_format = 3  # XYZ + RGB
        elif has_intensity:
            point_format = 1  # XYZ + Intensity
        else:
            point_format = 0  # XYZ only
        
        # Create LAZ file
        header = laspy.LasHeader(point_format=point_format, version="1.2")
        header.x_scale = 0.001  # 1mm precision
        header.y_scale = 0.001
        header.z_scale = 0.001
        
        # Set offset to minimize coordinate values
        header.x_offset = np.min(x_data)
        header.y_offset = np.min(y_data)
        header.z_offset = np.min(z_data)
        
        las = laspy.LasData(header)
        las.x = x_data
        las.y = y_data
        las.z = z_data
        
        if has_intensity:
            las.intensity = intensity_data
        if has_color:
            las.red = red_data
            las.green = green_data
            las.blue = blue_data
        
        # Write file
        las.write(chunk_file)
        
    except Exception as e:
        print(f"Error writing chunk to {chunk_file}: {e}")
        raise
    finally:
        # Clear memory immediately
        del x_data, y_data, z_data, intensity_data, red_data, green_data, blue_data
        if 'las' in locals():
            del las
        gc.collect()


def merge_laz_chunks_streaming(chunk_paths, output_path):
    """
    Merge LAZ chunks using streaming to minimize memory usage
    Reads and writes one chunk at a time instead of loading everything into memory
    """
    import laspy
    import numpy as np
    
    if not chunk_paths:
        return
    
    print(f"Merging {len(chunk_paths)} chunks using streaming method...")
    
    # Read first chunk to get format info
    first_chunk = laspy.read(chunk_paths[0])
    header = laspy.LasHeader(point_format=first_chunk.header.point_format, version="1.2")
    header.x_scale = 0.001
    header.y_scale = 0.001  
    header.z_scale = 0.001
    
    # Calculate total points and find global bounds (read chunks one by one)
    total_points = 0
    min_x = min_y = min_z = float('inf')
    
    print("  Calculating bounds and total points...")
    for i, chunk_path in enumerate(chunk_paths):
        print(f"    Scanning chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
        chunk = laspy.read(chunk_path)
        total_points += len(chunk.points)
        min_x = min(min_x, np.min(chunk.x))
        min_y = min(min_y, np.min(chunk.y))
        min_z = min(min_z, np.min(chunk.z))
        del chunk  # Free memory immediately
        gc.collect()
    
    print(f"  Total points to merge: {total_points:,}")
    
    header.x_offset = min_x
    header.y_offset = min_y
    header.z_offset = min_z
    
    # Create output LAS file and write chunks sequentially
    print("  Writing merged file...")
    try:
        with laspy.open(output_path, mode='w', header=header) as writer:
            for i, chunk_path in enumerate(chunk_paths):
                print(f"    Merging chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
                
                try:
                    chunk = laspy.read(chunk_path)
                    
                    # Write chunk data directly to output file
                    writer.write_points(chunk.points)
                    
                    # Free memory immediately
                    del chunk
                    gc.collect()
                    
                except Exception as chunk_error:
                    print(f"    Error processing chunk {chunk_path}: {chunk_error}")
                    continue
        
        print(f"✓ Merged file saved: {output_path}")
        
    except Exception as e:
        print(f"Error creating merged file: {e}")
        raise
    
    # Clear first chunk from memory
    del first_chunk
    gc.collect()


def write_chunk_to_laz_wrapper(args):
    """
    Wrapper function for multiprocessing - unpacks arguments and calls write_chunk_to_laz
    """
    chunk_data, chunk_file, chunk_index, total_chunks = args
    try:
        print(
            f"  Processing chunk {chunk_index+1}/{total_chunks} with {len(chunk_data):,} points...")
        write_chunk_to_laz(chunk_data, chunk_file)
        return True, chunk_index, len(chunk_data), None
    except Exception as e:
        return False, chunk_index, len(chunk_data), str(e)


def write_chunk_to_laz(chunk_data, chunk_file):
    """
    Write a chunk of point data to a LAZ file.
    Args:
        chunk_data (list): List of point dictionaries with x, y, z, intensity, red, green, blue
        chunk_file (str): Path to output chunk file
    """
    import laspy
    import numpy as np
    if not chunk_data:
        return
    # Convert chunk data to arrays
    points = np.array([(p["x"], p["y"], p["z"]) for p in chunk_data])
    # Check if we have color/intensity data
    has_color = any(p.get("red") is not None and p.get(
        "green") is not None and p.get("blue") is not None for p in chunk_data)
    has_intensity = any(p.get("intensity") is not None for p in chunk_data)
    # Determine point format
    if has_color and has_intensity:
        point_format = 3  # XYZ + RGB + Intensity
    elif has_color:
        point_format = 3  # XYZ + RGB
    elif has_intensity:
        point_format = 1  # XYZ + Intensity
    else:
        point_format = 0  # XYZ only
    # Create LAS file
    header = laspy.LasHeader(point_format=point_format, version="1.2")
    header.x_scale = 0.001  # 1mm precision
    header.y_scale = 0.001
    header.z_scale = 0.001
    # Set offset to minimize coordinate values
    header.x_offset = np.min(points[:, 0])
    header.y_offset = np.min(points[:, 1])
    header.z_offset = np.min(points[:, 2])
    las = laspy.LasData(header)
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    # Add color data if available
    # Add color data if available
    if has_color:
        red_data = np.array([p.get("red", 0)
                            for p in chunk_data], dtype=np.float64)
        green_data = np.array([p.get("green", 0)
                               for p in chunk_data], dtype=np.float64)
        blue_data = np.array([p.get("blue", 0)
                              for p in chunk_data], dtype=np.float64)
        # Convert to 16-bit if necessary - using float64 to avoid overflow
        if red_data.max() <= 1.0:
            red_data = (red_data * 65535.0).astype(np.uint16)
            green_data = (green_data * 65535.0).astype(np.uint16)
            blue_data = (blue_data * 65535.0).astype(np.uint16)
        elif red_data.max() <= 255:
            red_data = (red_data * 257.0).astype(np.uint16)
            green_data = (green_data * 257.0).astype(np.uint16)
            blue_data = (blue_data * 257.0).astype(np.uint16)
        else:
            red_data = np.clip(red_data, 0, 65535).astype(np.uint16)
            green_data = np.clip(green_data, 0, 65535).astype(np.uint16)
            blue_data = np.clip(blue_data, 0, 65535).astype(np.uint16)
        las.red = red_data
        las.green = green_data
        las.blue = blue_data
    # Add intensity data if available
    # Add intensity data if available
    if has_intensity:
        intensity_data = np.array([p.get("intensity", 0)
                                   for p in chunk_data], dtype=np.float64)
        # Convert to 16-bit if necessary
        if intensity_data.max() <= 1.0:
            intensity_data = (intensity_data * 65535).astype(np.uint16)
        elif intensity_data.max() <= 255:
            intensity_data = (intensity_data * 256).astype(np.uint16)
        else:
            intensity_data = intensity_data.astype(np.uint16)
        las.intensity = intensity_data
    # Save chunk
    las.write(chunk_file)


def merge_laz_chunks(chunk_paths, output_path):
    """
    Merge multiple LAZ chunk files into a single LAZ file.
    Args:
        chunk_paths (list): List of paths to chunk files
        output_path (str): Path to output merged file
    """
    import laspy
    import numpy as np
    if not chunk_paths:
        return
    print(f"Merging {len(chunk_paths)} chunks into final LAZ file...")
    # Read first chunk to get header template
    first_chunk = laspy.read(chunk_paths[0])
    total_points = sum(len(laspy.read(chunk_path).points)
                       for chunk_path in chunk_paths)
    print(f"Total points to merge: {total_points:,}")
    # Create output header
    header = laspy.LasHeader(
        point_format=first_chunk.header.point_format, version="1.2")
    header.x_scale = 0.001
    header.y_scale = 0.001
    header.z_scale = 0.001
    # Pre-allocate arrays for better memory efficiency
    all_x = np.empty(total_points, dtype=np.float64)
    all_y = np.empty(total_points, dtype=np.float64)
    all_z = np.empty(total_points, dtype=np.float64)
    has_colors = hasattr(first_chunk, 'red')
    has_intensity = hasattr(first_chunk, 'intensity')
    if has_colors:
        all_red = np.empty(total_points, dtype=np.uint16)
        all_green = np.empty(total_points, dtype=np.uint16)
        all_blue = np.empty(total_points, dtype=np.uint16)
    if has_intensity:
        all_intensity = np.empty(total_points, dtype=np.uint16)
    # Read and combine all chunks
    current_idx = 0
    for i, chunk_path in enumerate(chunk_paths):
        print(
            f"  Reading chunk {i+1}/{len(chunk_paths)}: {os.path.basename(chunk_path)}")
        chunk = laspy.read(chunk_path)
        chunk_size = len(chunk.points)
        # Copy coordinate data
        all_x[current_idx:current_idx + chunk_size] = chunk.x
        all_y[current_idx:current_idx + chunk_size] = chunk.y
        all_z[current_idx:current_idx + chunk_size] = chunk.z
        # Copy color data if available
        if has_colors:
            all_red[current_idx:current_idx + chunk_size] = chunk.red
            all_green[current_idx:current_idx + chunk_size] = chunk.green
            all_blue[current_idx:current_idx + chunk_size] = chunk.blue
        # Copy intensity data if available
        if has_intensity:
            all_intensity[current_idx:current_idx +
                          chunk_size] = chunk.intensity
        current_idx += chunk_size
    # Set offset to minimize coordinate values
    header.x_offset = np.min(all_x)
    header.y_offset = np.min(all_y)
    header.z_offset = np.min(all_z)
    # Create final LAS file
    las = laspy.LasData(header)
    las.x = all_x
    las.y = all_y
    las.z = all_z
    if has_colors:
        las.red = all_red
        las.green = all_green
        las.blue = all_blue
    if has_intensity:
        las.intensity = all_intensity
    print(f"Writing merged file: {output_path}")
    las.write(output_path)


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
        common_elements = ["data3D", "images", "images2D",
                           "cameraImages", "visualReferenceRepresentation"]
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
            print(
                f"Method 3: Found {len(found_elements)} elements via direct access")
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
                                    Image.fromarray(img_data).save(
                                        os.path.join(output_folder, img_filename))
                                    print(f"      Saved: {img_filename}")
                                    extracted_count += 1
                            except Exception as e:
                                print(f"      Error saving {key}: {e}")
                        # Check if the value is a dictionary/object that might contain images
                        elif hasattr(value, 'keys') and isinstance(value, dict):
                            print(f"      Examining nested field {key}:")
                            for sub_key, sub_value in value.items():
                                print(
                                    f"        - {sub_key}: {type(sub_value)}")
                                if sub_key in ['image', 'images', 'visualReferenceRepresentation', 'cameraImage', 'photo']:
                                    try:
                                        if hasattr(sub_value, 'shape'):  # numpy array
                                            img_filename = f"scan_{scan_idx:03d}_{key}_{sub_key}.jpg"
                                            Image.fromarray(sub_value).save(
                                                os.path.join(output_folder, img_filename))
                                            print(
                                                f"          Saved: {img_filename}")
                                            extracted_count += 1
                                    except Exception as e:
                                        print(
                                            f"          Error saving {sub_key}: {e}")
            except Exception as e:
                print(f"Error reading scan {scan_idx}: {e}")
                continue
    except Exception as e:
        print(f"Error accessing scan count: {e}")
    # Method 2: Try to access images directly if they exist
    try:
        if hasattr(e57.root, '__getitem__'):
            # Try common image element names
            image_elements = ['images', 'cameraImages',
                              'visualReferenceRepresentation', 'photos', 'cameraImage']
            for element_name in image_elements:
                try:
                    images = e57.root[element_name]
                    print(f"Found {len(images)} images in '{element_name}'")
                    for i, img in enumerate(images):
                        try:
                            img_data = e57.read_image(img)
                            img_filename = f"direct_{element_name}_{i:03d}.jpg"
                            Image.fromarray(img_data).save(
                                os.path.join(output_folder, img_filename))
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
                                print(
                                    f"    - {child_name}: {type(child_value)}")
                                # Check if this child contains images
                                if child_name in ['images', 'cameraImage', 'visualReferenceRepresentation', 'photo']:
                                    try:
                                        if hasattr(child_value, '__len__') and len(child_value) > 0:
                                            for img_idx, img in enumerate(child_value):
                                                try:
                                                    img_data = e57.read_image(
                                                        img)
                                                    img_filename = f"scan_{scan_idx:03d}_{child_name}_{img_idx:03d}.jpg"
                                                    Image.fromarray(img_data).save(
                                                        os.path.join(output_folder, img_filename))
                                                    print(
                                                        f"      Saved: {img_filename}")
                                                    extracted_count += 1
                                                except Exception as e:
                                                    print(
                                                        f"      Error processing image {img_idx}: {e}")
                                        else:
                                            # Single image
                                            img_data = e57.read_image(
                                                child_value)
                                            img_filename = f"scan_{scan_idx:03d}_{child_name}.jpg"
                                            Image.fromarray(img_data).save(
                                                os.path.join(output_folder, img_filename))
                                            print(
                                                f"      Saved: {img_filename}")
                                            extracted_count += 1
                                    except Exception as e:
                                        print(
                                            f"      Error processing {child_name}: {e}")
                            except Exception as e:
                                print(
                                    f"    - {child_name}: Error accessing - {e}")
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
    print(f"\n{'='*60}")
    print(f"Processing file: {os.path.basename(e57_path)}")
    print(f"{'='*60}")
    # Create output folder if not provided
    if output_folder is None:
        base_name = os.path.splitext(os.path.basename(e57_path))[0]
        output_folder = f"{base_name}_images"
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder: {output_folder}")
    try:
        # Open the E57 file
        e57 = pye57.E57(e57_path)
    except Exception as e:
        print(f"Error opening E57 file: {e}")
        return 0
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
                    Image.fromarray(img_data).save(
                        os.path.join(output_folder, img_filename))
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
        print(
            f"\nProcessing {len(images2d)} spherical images from 'images2D' element...")
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
                print(
                    f"Saved spherical image {i+1}/{len(images2d)}: {img_filename}")
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
                    print(
                        f"Found {len(scan_images)} images in scan {scan_idx}")
                    for img_idx, img in enumerate(scan_images):
                        try:
                            # Extract image data
                            img_data = e57.read_image(img)
                            # Get image metadata
                            img_name = img["name"].value(
                            ) if "name" in img else f"scan_{scan_idx}_img_{img_idx}"
                            img_guid = img["guid"].value(
                            ) if "guid" in img else f"scan_{scan_idx}_img_{img_idx}"
                            # Get MIME type
                            if "visualReferenceRepresentation" in img and "mimeType" in img["visualReferenceRepresentation"]:
                                img_mime = img["visualReferenceRepresentation"]["mimeType"].value(
                                )
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
                                Image.fromarray(img_data).save(
                                    os.path.join(output_folder, img_filename))
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
                            print(
                                f"Saved image from scan {scan_idx}, image {img_idx}: {img_filename}")
                            extracted_count += 1
                        except Exception as e:
                            print(
                                f"Error processing image {img_idx} in scan {scan_idx}: {str(e)}")
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
            print(
                f"\nSuccessfully extracted {direct_count} images using direct methods!")
            extracted_count = direct_count
        else:
            print("\nNo images could be extracted using any method.")
    else:
        print(
            f"\nSuccessfully extracted {extracted_count} images to {output_folder}")
    return extracted_count


def process_batch_e57_files(input_folder, output_base_folder, extract_pc=False, max_workers=8):
    """
    Process all E57 files in a folder and create separate output folders for each
    Args:
        input_folder (str): Folder containing E57 files
        output_base_folder (str): Base folder where individual file folders will be created
        extract_pc (bool): Whether to extract point clouds as LAZ files
    """
    # Find all E57 files in the input folder
    e57_files = []
    for ext in ['*.e57', '*.E57']:
        e57_files.extend(glob.glob(os.path.join(input_folder, ext)))
    if not e57_files:
        print(f"No E57 files found in {input_folder}")
        return 0, 0
    print(f"\nFound {len(e57_files)} E57 files to process:")
    for i, file_path in enumerate(e57_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    # Create base output folder if it doesn't exist
    os.makedirs(output_base_folder, exist_ok=True)
    total_files_processed = 0
    total_images_extracted = 0
    # Process each E57 file
    for i, e57_file in enumerate(e57_files, 1):
        try:
            # Get the base name without extension
            base_name = os.path.splitext(os.path.basename(e57_file))[0]
            # Create individual output folder for this file
            file_output_folder = os.path.join(output_base_folder, base_name)
            print(f"\n{'='*80}")
            print(f"Processing file {i}/{len(e57_files)}: {base_name}")
            print(f"{'='*80}")
            # Extract images from this E57 file
            extracted_count = extract_images_from_e57(
                e57_file, file_output_folder)
            # Extract point cloud if requested
            if extract_pc:
                print(f"\nExtracting point cloud for {base_name}...")
                pc_success = extract_point_cloud_to_laz_streaming(
    e57_file, file_output_folder, max_workers=max_workers)
                if not pc_success:
                    print(
                        f"Warning: Point cloud extraction failed for {base_name}")
            total_files_processed += 1
            total_images_extracted += extracted_count
            print(
                f"\nCompleted processing {base_name}: {extracted_count} images extracted")
        except Exception as e:
            print(f"\nError processing {os.path.basename(e57_file)}: {str(e)}")
            continue
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Files processed: {total_files_processed}/{len(e57_files)}")
    print(f"Total images extracted: {total_images_extracted}")
    print(f"Output location: {output_base_folder}")
    return total_files_processed, total_images_extracted


class E57BatchImageExtractorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("E57 Batch Image Extractor")
        self.root.geometry("800x650")
        self.root.minsize(600, 500)  # Set minimum window size
        # Configure styles
        self.style = ttk.Style()
        self.style.configure('TButton', padding=5)
        self.style.configure('TLabel', padding=5)
        # Create main container with scrollbar
        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(
            self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window(
            (0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        # Pack canvas and scrollbar
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        # Create main frame inside scrollable frame
        self.main_frame = ttk.Frame(self.scrollable_frame, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        # Bind mousewheel to canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)
        # Mode selection
        self.mode_frame = ttk.LabelFrame(
            self.main_frame, text="Processing Mode", padding="10")
        self.mode_frame.pack(fill=tk.X, pady=5)
        self.mode_var = tk.StringVar(value="batch")
        self.batch_radio = ttk.Radiobutton(
            self.mode_frame, text="Batch Process (Folder of E57 files)",
            variable=self.mode_var, value="batch", command=self.update_ui_mode
        )
        self.batch_radio.pack(anchor=tk.W)
        self.single_radio = ttk.Radiobutton(
            self.mode_frame, text="Single File",
            variable=self.mode_var, value="single", command=self.update_ui_mode
        )
        self.single_radio.pack(anchor=tk.W)
        # Input selection
        self.input_frame = ttk.LabelFrame(
            self.main_frame, text="Input", padding="10")
        self.input_frame.pack(fill=tk.X, pady=5)
        self.input_entry = ttk.Entry(self.input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X,
                              expand=True, padx=(0, 5))
        self.browse_input_button = ttk.Button(
            self.input_frame,
            text="Browse Folder...",
            command=self.browse_input
        )
        self.browse_input_button.pack(side=tk.RIGHT)
        # Output folder selection
        self.output_frame = ttk.LabelFrame(
            self.main_frame, text="Output Base Folder", padding="10")
        self.output_frame.pack(fill=tk.X, pady=5)
        self.output_entry = ttk.Entry(self.output_frame)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X,
                               expand=True, padx=(0, 5))
        self.output_browse_button = ttk.Button(
            self.output_frame,
            text="Browse...",
            command=self.browse_output_folder
        )
        self.output_browse_button.pack(side=tk.RIGHT)
        # Options frame
        self.options_frame = ttk.LabelFrame(
            self.main_frame, text="Extraction Options", padding="10")
        self.options_frame.pack(fill=tk.X, pady=5)
        # Point cloud extraction checkbox
        self.extract_point_cloud_var = tk.BooleanVar(value=False)
        self.pc_checkbox = ttk.Checkbutton(
            self.options_frame,
            text="Extract Point Cloud as LAZ files (saved in same folders as images)",
            variable=self.extract_point_cloud_var
        )
        self.pc_checkbox.pack(side=tk.LEFT, padx=5)
        # Add info label for point cloud requirements
        # Add info label for point cloud requirements
        self.pc_info_label = ttk.Label(
            self.options_frame,
            text="Note: Only requires 'pip install laspy' for point cloud extraction",
            font=('TkDefaultFont', 8),
            foreground='gray'
        )
        self.pc_info_label.pack(side=tk.LEFT, padx=(10, 5))
        # Multiprocessing options - ADD THIS SECTION HERE
        self.mp_frame = ttk.Frame(self.options_frame)
        self.mp_frame.pack(fill=tk.X, pady=5)
        ttk.Label(self.mp_frame, text="Max Workers:").pack(side=tk.LEFT)
        self.max_workers_var = tk.IntVar(value=min(8, mp.cpu_count()))
        self.workers_spinbox = ttk.Spinbox(
            self.mp_frame,
            from_=1,
            to=min(16, mp.cpu_count()),
            width=5,
            textvariable=self.max_workers_var
        )
        self.workers_spinbox.pack(side=tk.LEFT, padx=(5, 10))
        ttk.Label(
            self.mp_frame,
            text=f"(Available cores: {mp.cpu_count()})",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).pack(side=tk.LEFT)
        # Progress and log
        self.log_frame = ttk.LabelFrame(
            self.main_frame, text="Progress", padding="10")
        self.log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        # Progress bar
        self.progress_var = tk.StringVar()
        self.progress_label = ttk.Label(
            self.log_frame, textvariable=self.progress_var)
        self.progress_label.pack(fill=tk.X, pady=(0, 5))
        self.progress_bar = ttk.Progressbar(self.log_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))
        # Create frame for text and scrollbar
        self.log_text_frame = ttk.Frame(self.log_frame)
        self.log_text_frame.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(self.log_text_frame, height=15, wrap=tk.WORD)
        self.log_scrollbar = ttk.Scrollbar(
            self.log_text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        # Buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=5)
        self.extract_button = ttk.Button(
            self.button_frame,
            text="Start Processing",
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
        # Initialize UI mode
        self.update_ui_mode()
        # Redirect stdout to log
        self.redirect_stdout()
        # Update canvas scroll region after all widgets are created
        self.root.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        if event.delta:
            # Windows
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        else:
            # Linux
            if event.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif event.num == 5:
                self.canvas.yview_scroll(1, "units")

    def update_ui_mode(self):
        """Update UI elements based on selected mode"""
        mode = self.mode_var.get()
        if mode == "batch":
            self.browse_input_button.config(text="Browse Folder...")
            self.input_frame.config(text="Input Folder (containing E57 files)")
            self.output_frame.config(
                text="Output Base Folder (subfolders will be created)")
            self.extract_button.config(text="Start Batch Processing")
        else:
            self.browse_input_button.config(text="Browse File...")
            self.input_frame.config(text="Input E57 File")
            self.output_frame.config(text="Output Folder")
            self.extract_button.config(text="Extract Images")

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

    def browse_input(self):
        """Browse for input file or folder based on mode"""
        mode = self.mode_var.get()
        if mode == "batch":
            # Browse for folder
            folder_path = filedialog.askdirectory(
                title="Select Folder Containing E57 Files"
            )
            if folder_path:
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, folder_path)
                # Suggest output folder
                suggested_output = os.path.join(
                    folder_path, "extracted_images")
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, suggested_output)
        else:
            # Browse for single file
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
        """Open folder dialog to select output directory"""
        # Get the initial directory from the input if it exists
        initial_dir = None
        input_path = self.input_entry.get()
        if input_path:
            if os.path.isfile(input_path):
                initial_dir = os.path.dirname(input_path)
            elif os.path.isdir(input_path):
                initial_dir = input_path
        folder_path = filedialog.askdirectory(
            title="Select Output Folder",
            initialdir=initial_dir
        )
        if folder_path:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder_path)

    def clear_log(self):
        """Clear the log text widget"""
        self.log_text.delete(1.0, tk.END)
        self.progress_bar['value'] = 0
        self.progress_var.set("")

    def update_progress(self, current, total, message=""):
        """Update progress bar and message"""
        if total > 0:
            progress = (current / total) * 100
            self.progress_bar['value'] = progress
        self.progress_var.set(f"{message} ({current}/{total})")
        self.root.update_idletasks()

    def start_extraction(self):
        """Start the extraction process in a separate thread"""
        input_path = self.input_entry.get()
        output_path = self.output_entry.get()
        extract_pc = self.extract_point_cloud_var.get()
        mode = self.mode_var.get()
        if not input_path:
            messagebox.showerror(
                "Error", "Please select an input file or folder")
            return
        if not output_path:
            messagebox.showerror("Error", "Please select an output folder")
            return
        # Validate input based on mode
        if mode == "batch":
            if not os.path.isdir(input_path):
                messagebox.showerror(
                    "Error", "Please select a valid input folder")
                return
        else:
            if not os.path.isfile(input_path):
                messagebox.showerror("Error", "Please select a valid E57 file")
                return
        # Disable controls during extraction
        self.extract_button.config(state=tk.DISABLED)
        self.browse_input_button.config(state=tk.DISABLED)
        self.output_browse_button.config(state=tk.DISABLED)
        self.pc_checkbox.config(state=tk.DISABLED)
        self.batch_radio.config(state=tk.DISABLED)
        self.single_radio.config(state=tk.DISABLED)

        def extraction_thread():
            try:
                if mode == "batch":
                    # Batch processing
                    self.root.after(0, lambda: self.status_var.set(
                        "Processing batch..."))
                    files_processed, images_extracted = process_batch_e57_files(
                        input_path, output_path, extract_pc, self.max_workers_var.get())
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Batch Complete",
                        f"Processed {files_processed} files\n"
                        f"Extracted {images_extracted} total images\n"
                        f"Output saved to: {output_path}"
                    ))
                else:
                    # Single file processing
                    self.root.after(0, lambda: self.status_var.set(
                        "Processing single file..."))
                    # Create output folder if it doesn't exist
                    os.makedirs(output_path, exist_ok=True)
                    extracted_count = extract_images_from_e57(
                        input_path, output_path)
                    # Extract point cloud if requested
                    # Extract point cloud if requested
                    if extract_pc:
                        pc_success = extract_point_cloud_to_laz_streaming(
    input_path, output_path, max_workers=self.max_workers_var.get())
                        if not pc_success:
                            self.root.after(0, lambda: messagebox.showwarning(
                                "Warning",
                                "Point cloud extraction failed - check console for details"
                            ))
                    self.root.after(0, lambda: messagebox.showinfo(
                        "Extraction Complete",
                        f"Extracted {extracted_count} images\n"
                        f"Output saved to: {output_path}"
                    ))
            except Exception as e:
                print(f"Error during extraction: {str(e)}")
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", f"An error occurred during extraction:\n{str(e)}"
                ))
            finally:
                # Re-enable controls when done
                self.root.after(
                    0, lambda: self.extract_button.config(state=tk.NORMAL))
                self.root.after(
                    0, lambda: self.browse_input_button.config(state=tk.NORMAL))
                self.root.after(
                    0, lambda: self.output_browse_button.config(state=tk.NORMAL))
                self.root.after(
                    0, lambda: self.pc_checkbox.config(state=tk.NORMAL))
                self.root.after(
                    0, lambda: self.batch_radio.config(state=tk.NORMAL))
                self.root.after(
                    0, lambda: self.single_radio.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.status_var.set("Complete"))
                self.root.after(0, lambda: self.progress_bar.config(value=100))
        thread = threading.Thread(target=extraction_thread, daemon=True)
        thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = E57BatchImageExtractorApp(root)
    # Set application icon (optional)
    try:
        # Provide your own icon file
        root.iconbitmap(default='extractor_icon.ico')
    except:
        pass
    root.mainloop()
