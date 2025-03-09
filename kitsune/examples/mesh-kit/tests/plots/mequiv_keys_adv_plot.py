#!/usr/bin/env python3
import sys
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_plots(input_dir, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files for each scenario
    scenarios = {}
    for file in glob.glob(f"{input_dir}/key_comparison_*_*.csv"):
        base = os.path.basename(file)
        parts = base.split('_')
        scenario = parts[2]  # Extract scenario name
        key_type = parts[3].split('.')[0]  # Extract key type
        
        if scenario not in scenarios:
            scenarios[scenario] = {}
        scenarios[scenario][key_type] = file
    
    # Process each scenario
    for scenario, files in scenarios.items():
        print(f"Processing scenario: {scenario}")
        
        # Load data for each key type
        data = {}
        for key_type, file in files.items():
            data[key_type] = pd.read_csv(file)
        
        # Create comparison plot
        plt.figure(figsize=(12, 7))
        sns.set_style("whitegrid")
        
        # Plot each key type with a different marker and color
        markers = ['o', 's', '^']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        for i, (key_type, df) in enumerate(data.items()):
            plt.semilogx(df['RelativeDelta'], df['EquivalenceRatio'], 
                        marker=markers[i % len(markers)], 
                        color=colors[i % len(colors)],
                        linestyle='-', 
                        linewidth=2,
                        label=f'{key_type} Key')
        
        plt.xlabel('Relative Delta', fontsize=12)
        plt.ylabel('Equivalence Ratio', fontsize=12)
        plt.title(f'Equivalence Key Comparison - {scenario}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{output_dir}/comparison_{scenario}.png", dpi=300)
        print(f"  Saved plot to {output_dir}/comparison_{scenario}.png")
        
        # Create performance comparison if data is available
        if 'HashTime_us' in data[list(data.keys())[0]]:
            plt.figure(figsize=(14, 6))
            
            # Set up subplot layout
            plt.subplot(1, 2, 1)
            
            # Extract hash times for each key type
            hash_times = {k: df['HashTime_us'].mean() for k, df in data.items() if 'HashTime_us' in df}
            names = list(hash_times.keys())
            values = list(hash_times.values())
            
            # Create bar chart
            plt.bar(names, values, color=colors[:len(names)])
            plt.ylabel('Hash Time (μs)', fontsize=12)
            plt.title('Hash Generation Performance', fontsize=14)
            
            # Set up subplot for equivalence time
            plt.subplot(1, 2, 2)
            
            # Extract equivalence times
            equiv_times = {k: df['EquivalenceTime_us'].mean() for k, df in data.items() if 'EquivalenceTime_us' in df}
            names = list(equiv_times.keys())
            values = list(equiv_times.values())
            
            # Create bar chart
            plt.bar(names, values, color=colors[:len(names)])
            plt.ylabel('Equivalence Check Time (μs)', fontsize=12)
            plt.title('Equivalence Check Performance', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_{scenario}.png", dpi=300)
            print(f"  Saved performance plot to {output_dir}/performance_{scenario}.png")
            
        # Also create hash collision plots if data is available
        if 'HashCollisionRate' in data[list(data.keys())[0]]:
            plt.figure(figsize=(10, 6))
            
            collision_rates = {k: df['HashCollisionRate'].mean() for k, df in data.items() if 'HashCollisionRate' in df}
            names = list(collision_rates.keys())
            values = list(collision_rates.values())
            
            plt.bar(names, values, color=colors[:len(names)])
            plt.ylabel('Hash Collision Rate', fontsize=12)
            plt.title(f'Hash Collision Rates - {scenario}', fontsize=14)
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/collisions_{scenario}.png", dpi=300)
            print(f"  Saved collision plot to {output_dir}/collisions_{scenario}.png")
        
        plt.close('all')  # Close all figures to free memory

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input_dir> <output_dir>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    create_plots(input_dir, output_dir)
    print("Plot generation complete.")

