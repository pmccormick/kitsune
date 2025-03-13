#!/usr/bin/env python3

import os
import glob

# Find all .txt files in the current directory and subdirectories
txt_files = glob.glob('**/*.txt', recursive=True)

# Rename each file
for file in txt_files:
    base = file[:-4]  # Remove the .txt extension
    new_name = base + '.cpp'
    
    os.rename(file, new_name)
    print(f"Converted: {file} -> {new_name}")

print("Conversion complete!")

