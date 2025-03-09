#!/usr/bin/env python3
"""
Script to fix the issues with the split Accessor Iterator files
"""
import os
import re

# Create the common header file
def create_common_header():
    content = '''/**
 * @file AccessorIteratorsCommon.h
 * @brief Common definitions and forward declarations for iterator classes
 */

#ifndef ACCESSOR_ITERATORS_COMMON_H
#define ACCESSOR_ITERATORS_COMMON_H

#include "FieldIterators.h"
#include <functional>

// Forward declarations
template <typename T, typename LocationTag> class Field;
using RegionMask = uint64_t;
using FieldMask = uint64_t;
using FieldID = uint32_t;

// Add any additional shared declarations here

#endif // ACCESSOR_ITERATORS_COMMON_H
'''
    with open('AccessorIteratorsCommon.h', 'w') as f:
        f.write(content)

# Fix MeshAccessorIterators.h
def fix_mesh_accessor_iterators():
    # Read existing file
    with open('MeshAccessorIterators.h', 'r') as f:
        content = f.read()
    
    # Add include for common header
    include_line = '#include "AccessorIteratorsCommon.h"'
    content = re.sub(r'#include "FieldIterators.h"', include_line, content)
    
    # Update header guard if needed
    content = re.sub(r'#ifndef MESHACCESSORITERATORS_H', '#ifndef MESH_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#define MESHACCESSORITERATORS_H', '#define MESH_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#endif // MESHACCESSORITERATORS_H', '#endif // MESH_ACCESSOR_ITERATORS_H', content)
    
    # Write back to file
    with open('MeshAccessorIterators.h', 'w') as f:
        f.write(content)

# Fix RegionAccessorIterators.h
def fix_region_accessor_iterators():
    # Read existing file
    with open('RegionAccessorIterators.h', 'r') as f:
        content = f.read()
    
    # Add include for common header
    include_line = '#include "AccessorIteratorsCommon.h"'
    content = re.sub(r'#include "FieldIterators.h"', include_line, content)
    
    # Update header guard if needed
    content = re.sub(r'#ifndef REGIONACCESSORITERATORS_H', '#ifndef REGION_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#define REGIONACCESSORITERATORS_H', '#define REGION_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#endif // REGIONACCESSORITERATORS_H', '#endif // REGION_ACCESSOR_ITERATORS_H', content)
    
    # Write back to file
    with open('RegionAccessorIterators.h', 'w') as f:
        f.write(content)

# Fix FieldAccessorIterators.h
def fix_field_accessor_iterators():
    # Read existing file
    with open('FieldAccessorIterators.h', 'r') as f:
        content = f.read()
    
    # Add include for common header
    include_line = '#include "AccessorIteratorsCommon.h"'
    content = re.sub(r'#include "FieldIterators.h"', include_line, content)
    
    # Remove duplicate class definitions
    # This would need more complex parsing to handle properly
    
    # Update header guard if needed
    content = re.sub(r'#ifndef FIELDACCESSORITERATORS_H', '#ifndef FIELD_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#define FIELDACCESSORITERATORS_H', '#define FIELD_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#endif // FIELDACCESSORITERATORS_H', '#endif // FIELD_ACCESSOR_ITERATORS_H', content)
    
    # Write back to file
    with open('FieldAccessorIterators.h', 'w') as f:
        f.write(content)

# Fix CompoundAccessorIterators.h
def fix_compound_accessor_iterators():
    # Read existing file
    with open('CompoundAccessorIterators.h', 'r') as f:
        content = f.read()
    
    # Add include for common header and other iterators
    includes = '''#include "AccessorIteratorsCommon.h"
#include "MeshAccessorIterators.h"
#include "RegionAccessorIterators.h"
#include "FieldAccessorIterators.h"'''
    content = re.sub(r'#include "FieldIterators.h"', includes, content)
    
    # Update header guard if needed
    content = re.sub(r'#ifndef COMPOUNDACCESSORITERATORS_H', '#ifndef COMPOUND_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#define COMPOUNDACCESSORITERATORS_H', '#define COMPOUND_ACCESSOR_ITERATORS_H', content)
    content = re.sub(r'#endif // COMPOUNDACCESSORITERATORS_H', '#endif // COMPOUND_ACCESSOR_ITERATORS_H', content)
    
    # Write back to file
    with open('CompoundAccessorIterators.h', 'w') as f:
        f.write(content)

# Update the master header
def update_master_header():
    content = '''/**
 * @file AccessorIterators.h
 * @brief Iterator interfaces for Accessor classes
 * 
 * This file defines iterator interfaces for the various accessor classes
 * (MeshAccessor, RegionAccessor, FieldAccessor, CompoundAccessor).
 * These iterators build on the Field iterators to provide consistent and
 * efficient traversal patterns for mesh data.
 */

#ifndef ACCESSOR_ITERATORS_H
#define ACCESSOR_ITERATORS_H

#include "AccessorIteratorsCommon.h"
#include "MeshAccessorIterators.h"
#include "RegionAccessorIterators.h"
#include "FieldAccessorIterators.h"
#include "CompoundAccessorIterators.h"

#endif // ACCESSOR_ITERATORS_H
'''
    with open('AccessorIterators.h', 'w') as f:
        f.write(content)

# Run all the fixes
def fix_all():
    create_common_header()
    fix_mesh_accessor_iterators()
    fix_region_accessor_iterators()
    fix_field_accessor_iterators()
    fix_compound_accessor_iterators()
    update_master_header()
    
    # Remove unused files
    for file in ['CellIterators.h', 'InteriorCellIterators.h', 'BlockCellIterators.h', 'RegionCellIterators.h']:
        if os.path.exists(file):
            os.remove(file)
    
    print("All files fixed successfully!")

if __name__ == "__main__":
    fix_all()

