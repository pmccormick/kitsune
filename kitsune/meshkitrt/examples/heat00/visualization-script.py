#!/usr/bin/env python3
"""
visualize_heat_transfer.py

This script creates visualizations of the heat transfer simulation results.
It reads CSV files from the heat_results directory and generates:
1. Heatmap visualizations of single time steps
2. An animation showing the evolution of temperature over time
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import Normalize

def load_result_files(directory="heat_results"):
    """Load all CSV result files in order"""
    files = glob.glob(f"{directory}/temperature_*.csv")
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return files

def create_single_frame_visualization(file_path, output_dir="heat_results/figs"):
    """Create a visualization of a single time step"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(file_path)
    step = int(file_path.split('_')[-1].split('.')[0])
    
    # Extract grid dimensions
    nx = data['i'].max() + 1
    ny = data['j'].max() + 1
    
    # Reshape data into a 2D grid
    temperature_grid = np.zeros((ny, nx))
    for _, row in data.iterrows():
        i, j = int(row['i']), int(row['j'])
        temperature_grid[j, i] = row['temperature']
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with the 'hot' colormap
    img = plt.imshow(temperature_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(img, label='Temperature (°C)')
    
    # Add axes labels
    plt.xlabel('X Cell Index')
    plt.ylabel('Y Cell Index')
    
    # Add title
    plt.title(f'Temperature Distribution at Step {step}')
    
    # Save figure
    output_file = f"{output_dir}/temperature_step_{step}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualization for step {step}: {output_file}")
    
    return temperature_grid

def create_animation(files, output_path="heat_results/temperature_animation.mp4"):
    """Create an animation of the temperature evolution"""
    if not files:
        print("No files found for animation")
        return
    
    # Load first file to get dimensions
    data = pd.read_csv(files[0])
    nx = data['i'].max() + 1
    ny = data['j'].max() + 1
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Initialize with zeros
    temperature_grid = np.zeros((ny, nx))
    img = ax.imshow(temperature_grid, cmap='hot', interpolation='nearest')
    plt.colorbar(img, ax=ax, label='Temperature (°C)')
    
    # Set title template
    title = ax.set_title('Temperature Distribution - Step 0')
    
    # Add axes labels
    ax.set_xlabel('X Cell Index')
    ax.set_ylabel('Y Cell Index')
    
    # Function to update the frame
    def update_frame(frame_idx):
        file_path = files[frame_idx]
        step = int(file_path.split('_')[-1].split('.')[0])
        
        # Load data
        data = pd.read_csv(file_path)
        
        # Update temperature grid
        temperature_grid = np.zeros((ny, nx))
        for _, row in data.iterrows():
            i, j = int(row['i']), int(row['j'])
            temperature_grid[j, i] = row['temperature']
        
        # Update image data
        img.set_array(temperature_grid)
        
        # Update title
        title.set_text(f'Temperature Distribution - Step {step}')
        
        return [img, title]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(files), 
        interval=200, blit=True
    )
    
    # Save animation
    anim.save(output_path, writer='ffmpeg', dpi=150)
    plt.close()
    
    print(f"Created animation: {output_path}")

def create_3d_visualization(file_path, output_dir="heat_results/figs"):
    """Create a 3D surface plot visualization of temperature"""
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(file_path)
    step = int(file_path.split('_')[-1].split('.')[0])
    
    # Extract grid dimensions
    nx = data['i'].max() + 1
    ny = data['j'].max() + 1
    
    # Create coordinate grids
    x = np.arange(0, nx)
    y = np.arange(0, ny)
    X, Y = np.meshgrid(x, y)
    
    # Create temperature grid
    Z = np.zeros((ny, nx))
    for _, row in data.iterrows():
        i, j = int(row['i']), int(row['j'])
        Z[j, i] = row['temperature']
    
    # Create 3D surface plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap='hot', linewidth=0, antialiased=False)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Temperature (°C)')
    
    # Set labels and title
    ax.set_xlabel('X Cell Index')
    ax.set_ylabel('Y Cell Index')
    ax.set_zlabel('Temperature (°C)')
    ax.set_title(f'3D Temperature Distribution at Step {step}')
    
    # Save figure
    output_file = f"{output_dir}/temperature_3d_step_{step}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created 3D visualization for step {step}: {output_file}")

def main():
    """Main function to create all visualizations"""
    print("Generating heat transfer visualizations...")
    
    # Load all result files
    files = load_result_files()
    
    if not files:
        print("No result files found. Run the simulation first.")
        return
        
    print(f"Found {len(files)} result files")
    
    # Create directory for visualizations
    viz_dir = "heat_results/figs"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create visualizations for each time step
    for file_path in files:
        create_single_frame_visualization(file_path, viz_dir)
        
        # For the last time step, also create a 3D visualization
        if file_path == files[-1]:
            create_3d_visualization(file_path, viz_dir)
    
    # Create animation
    try:
        create_animation(files)
    except Exception as e:
        print(f"Could not create animation: {e}")
        print("Note: Creating animations requires ffmpeg to be installed.")
    
    print("Visualization complete.")

if __name__ == "__main__":
    main()
