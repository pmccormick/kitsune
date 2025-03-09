#!/usr/bin/env python3
"""
Header Splitter - Splits large C++ header files into smaller, more manageable files.
This script specifically targets the AccessorIterators.h file but can be adapted for other files.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set

def extract_class_info(content: str) -> Dict[str, Tuple[int, int, Set[str]]]:
    """
    Extract information about each class in the file.
    Returns a dictionary mapping class names to (start_pos, end_pos, dependencies).
    """
    # Pattern to match class definitions
    class_pattern = re.compile(r'(\s*)(template\s*<.*?>)?\s*class\s+(\w+)(?:\s*:.*?)?\s*{', re.DOTALL)
    
    # Pattern to match the end of a class definition (closing brace followed by semicolon or just a closing brace)
    class_end_pattern = re.compile(r'};')
    
    # Find all class definitions
    class_info = {}
    stack = []  # To track nested classes
    
    for match in class_pattern.finditer(content):
        indent = match.group(1)
        is_template = match.group(2) is not None
        class_name = match.group(3)
        start_pos = match.start()
        
        # Find the matching closing brace
        brace_count = 1
        end_pos = start_pos + len(match.group(0))
        
        while brace_count > 0 and end_pos < len(content):
            if content[end_pos] == '{':
                brace_count += 1
            elif content[end_pos] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Check if followed by semicolon
                    if end_pos + 1 < len(content) and content[end_pos + 1] == ';':
                        end_pos += 1
            end_pos += 1
        
        # Extract dependencies - this is a simplified approach and might need refinement
        class_text = content[start_pos:end_pos]
        dependencies = set()
        for dep in re.findall(r'(\w+)::', class_text):
            if dep != class_name:  # Avoid self-references
                dependencies.add(dep)
        
        # Also look for template parameters that are class names
        for dep in re.findall(r'typename\s+(\w+)', class_text):
            if dep != 'T' and dep != 'LocationTag':  # Skip common template parameters
                dependencies.add(dep)
        
        # Add inheritance dependencies
        for dep in re.findall(r':\s*(?:public|protected|private)\s+(\w+)', class_text):
            dependencies.add(dep)
        
        class_info[class_name] = (start_pos, end_pos, dependencies)
    
    return class_info

def group_classes(class_info: Dict[str, Tuple[int, int, Set[str]]]) -> List[List[str]]:
    """
    Group classes into files based on their relationships and size.
    Returns a list of lists, where each inner list is a group of classes that should go into one file.
    """
    # Strategy: First group by template class, then by related functionality
    
    # Find template parent classes
    template_classes = set()
    for class_name in class_info:
        if "Iterator" in class_name or "Range" in class_name:
            parent_name = class_name.split("Iterator")[0] if "Iterator" in class_name else class_name.split("Range")[0]
            template_classes.add(parent_name)
    
    # Group by parent class
    groups = {}
    for class_name, (start_pos, end_pos, deps) in class_info.items():
        placed = False
        
        # Try to place with parent class
        for parent in template_classes:
            if parent in class_name:
                if parent not in groups:
                    groups[parent] = []
                groups[parent].append(class_name)
                placed = True
                break
        
        # If not placed with parent, create new group
        if not placed:
            groups[class_name] = [class_name]
    
    # Convert dict to list of lists
    result = list(groups.values())
    
    # Further split any groups that are too large (more than ~1000 lines)
    # This would require calculating the line count for each class
    
    return result

def generate_file_content(group: List[str], class_info: Dict[str, Tuple[int, int, Set[str]]], content: str, base_includes: str) -> str:
    """
    Generate the content for a new file containing the specified classes.
    """
    includes = base_includes + "\n"
    
    # Add additional includes based on dependencies
    all_deps = set()
    for class_name in group:
        all_deps.update(class_info[class_name][2])
    
    # Filter out dependencies that are in the current group
    external_deps = all_deps - set(group)
    
    # Add includes for external dependencies
    for dep in external_deps:
        for existing_group in all_groups:
            if dep in existing_group and existing_group != group:
                includes += f'#include "{get_filename_for_group(existing_group)}"\n'
    
    file_content = includes + "\n"
    
    # Add classes in the correct order
    group_with_positions = [(name, class_info[name][0]) for name in group]
    sorted_group = [name for name, _ in sorted(group_with_positions, key=lambda x: x[1])]
    
    for class_name in sorted_group:
        start_pos, end_pos, _ = class_info[class_name]
        class_text = content[start_pos:end_pos]
        file_content += class_text + "\n\n"
    
    return file_content

def get_filename_for_group(group: List[str]) -> str:
    """
    Generate a filename for a group of classes.
    """
    if not group:
        return "EmptyGroup.h"
    
    # Use the first class name in the group as the base filename
    base_name = group[0]
    
    # If it's a very specific name, try to generalize
    if "Iterator" in base_name:
        base_name = base_name.split("Iterator")[0] + "Iterators"
    elif "Range" in base_name:
        base_name = base_name.split("Range")[0] + "Ranges"
    
    # Special cases
    if "MeshAccessor" in base_name:
        return "MeshAccessorIterators.h"
    elif "RegionAccessor" in base_name:
        return "RegionAccessorIterators.h"
    elif "FieldAccessor" in base_name:
        return "FieldAccessorIterators.h"
    elif "CompoundAccessor" in base_name:
        return "CompoundAccessorIterators.h"
    
    return f"{base_name}.h"

def extract_header_comment_and_guards(content: str) -> Tuple[str, str, str]:
    """
    Extract the header comment and include guards from the original file.
    """
    # Extract header comment
    header_comment = ""
    header_match = re.search(r'/\*\*.*?\*/', content, re.DOTALL)
    if header_match:
        header_comment = header_match.group(0)
    
    # Extract include guards and everything between them
    guards_match = re.search(r'#ifndef\s+(\w+).*?#define\s+\1(.*?)#endif\s+//\s+\1', content, re.DOTALL)
    
    guard_name = ""
    includes = ""
    
    if guards_match:
        guard_name = guards_match.group(1)
        includes_section = guards_match.group(2)
        
        # Extract includes
        includes_match = re.search(r'#include.*', includes_section)
        if includes_match:
            includes_start = includes_match.start()
            includes_end = includes_section.find("\n\n", includes_start)
            if includes_end == -1:
                includes_end = len(includes_section)
            includes = includes_section[includes_start:includes_end]
    
    return header_comment, guard_name, includes

def create_new_header_files(content: str, output_dir: str):
    """
    Split the header file into multiple files and write them to disk.
    """
    # Extract header comment and include guards
    header_comment, original_guard_name, base_includes = extract_header_comment_and_guards(content)
    
    # Extract class information
    class_info = extract_class_info(content)
    
    # Group classes into files
    global all_groups
    all_groups = group_classes(class_info)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a master header file that includes all the others
    master_filename = "AccessorIterators.h"
    master_content = f"""{header_comment}

#ifndef {original_guard_name}
#define {original_guard_name}

{base_includes}

"""
    
    # Generate and write individual files
    for group in all_groups:
        filename = get_filename_for_group(group)
        guard_name = f"{filename.replace('.', '_').upper()}"
        
        file_content = f"""{header_comment}

#ifndef {guard_name}
#define {guard_name}

{base_includes}

"""
        
        # Add class content
        for class_name in group:
            start_pos, end_pos, _ = class_info[class_name]
            class_text = content[start_pos:end_pos]
            file_content += class_text + "\n\n"
        
        file_content += f"#endif // {guard_name}\n"
        
        # Write the file
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(file_content)
        
        # Add include to master header
        master_content += f'#include "{filename}"\n'
    
    # Finalize and write master header
    master_content += f"\n#endif // {original_guard_name}\n"
    with open(os.path.join(output_dir, master_filename), 'w') as f:
        f.write(master_content)
    
    print(f"Split into {len(all_groups)} files in {output_dir}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_headers.py <input_file> [output_directory]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "split_headers"
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    create_new_header_files(content, output_dir)

if __name__ == "__main__":
    main()

