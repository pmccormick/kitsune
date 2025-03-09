/**
 * @file Region.cpp
 * @brief Implementation of the Region class
 * 
 * This file contains the implementation of the Region class methods,
 * focusing on efficient storage and operations for cell collections.
 * 
 * IMPLEMENTATION NOTES:
 * - Storage mode transitions require careful data conversion
 * - Performance optimization is achieved through specialized operations per mode
 * - Memory management is crucial for large meshes
 */

#include "Region.h"
#include "MeshBase.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

void Region::setStorageMode(StorageMode mode) {
    // If already in the requested mode, do nothing
    if (m_mode == mode) {
        return;
    }
    
    // Handle mode transitions with data conversion
    if (mode == StorageMode::BIT_VECTOR && m_mode == StorageMode::CELL_SET) {
        // Convert from cell set to bit vector
        ensureBitVector();
        m_mode = mode;
    }
    else if (mode == StorageMode::CELL_SET && m_mode == StorageMode::BIT_VECTOR) {
        // Convert from bit vector to cell set
        ensureCellSet();
        m_mode = mode;
    }
    else {
        // Simple mode change (e.g., to/from DYNAMIC)
        // No conversion needed, just change the mode
        m_mode = mode;
    }
}

bool Region::contains(const CellBase* cell) const {
    // Safety check - null cells are never in a region
    if (!cell) {
        return false;
    }
    
    // DYNAMIC mode always evaluates using the definition
    if (m_mode == StorageMode::DYNAMIC) {
        return m_definition->contains(cell);
    }
    
    // For other modes, check if the cell's index is in the region
    return containsIndex(cell->linearIndex());
}

bool Region::containsIndex(int linearIndex) const {
    // Check index is in valid range
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return false;
    }
    
    // Check based on storage mode
    switch (m_mode) {
        case StorageMode::CELL_SET:
            // Check if index is in the set
            return m_cellIndices.find(linearIndex) != m_cellIndices.end();
        
        case StorageMode::BIT_VECTOR:
            // Check if the corresponding bit is set
            // First verify bit vector exists and index is in range
            return m_bitVector && (static_cast<size_t>(linearIndex) < m_bitVector->size()) &&
                   (*m_bitVector)[linearIndex];
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we can't check by index alone
            // This shouldn't be called for dynamic mode, but return false to be safe
            return false;
        
        default:
            return false;
    }
}

void Region::addCell(const CellBase* cell) {
    // Safety check
    if (!cell) {
        return;
    }
    
    // Add the cell's linear index
    addCellIndex(cell->linearIndex());
}

void Region::addCellIndex(int linearIndex) {
    // Check index is in valid range
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return;
    }
    
    // Add based on storage mode
    switch (m_mode) {
        case StorageMode::CELL_SET:
            // Add index to the set
            m_cellIndices.insert(linearIndex);
            break;
        
        case StorageMode::BIT_VECTOR:
            // Ensure bit vector exists and is properly sized
            ensureBitVector();
            // Set the corresponding bit
            if (static_cast<size_t>(linearIndex) < m_bitVector->size()) {
                (*m_bitVector)[linearIndex] = true;
            }
            break;
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we need to switch to a storage-based mode
            m_mode = StorageMode::CELL_SET;
            m_cellIndices.insert(linearIndex);
            break;
    }
}

void Region::removeCell(const CellBase* cell) {
    // Safety check
    if (!cell) {
        return;
    }
    
    // Remove the cell's linear index
    removeCellIndex(cell->linearIndex());
}

void Region::removeCellIndex(int linearIndex) {
    // Check index is in valid range
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return;
    }
    
    // Remove based on storage mode
    switch (m_mode) {
        case StorageMode::CELL_SET:
            // Remove index from the set
            m_cellIndices.erase(linearIndex);
            break;
        
        case StorageMode::BIT_VECTOR:
            // Clear the corresponding bit
            if (m_bitVector && static_cast<size_t>(linearIndex) < m_bitVector->size()) {
                (*m_bitVector)[linearIndex] = false;
            }
            break;
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we need to switch to a storage-based mode
            // Initialize the storage with all cells that match the definition
            // Then remove this specific index
            m_mode = StorageMode::BIT_VECTOR;
            ensureBitVector();
            if (static_cast<size_t>(linearIndex) < m_bitVector->size()) {
                (*m_bitVector)[linearIndex] = false;
            }
            break;
    }
}

void Region::clear() {
    // Clear all data structures but keep the definition
    m_cellIndices.clear();
    if (m_bitVector) {
        m_bitVector->assign(m_bitVector->size(), false);
    }
}

size_t Region::size() const {
    // Calculate size based on storage mode
    switch (m_mode) {
        case StorageMode::CELL_SET:
            // Size is the number of indices in the set
            return m_cellIndices.size();
        
        case StorageMode::BIT_VECTOR:
            // Size is the number of set bits in the vector
            if (!m_bitVector) return 0;
            return std::count(m_bitVector->begin(), m_bitVector->end(), true);
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we would need to scan the entire mesh
            // Since we don't have access to the mesh here, return 0
            return 0;
        
        default:
            return 0;
    }
}

const std::unordered_set<int>& Region::getCellIndices() {
    // If in BIT_VECTOR mode, convert to CELL_SET first
    if (m_mode == StorageMode::BIT_VECTOR) {
        ensureCellSet();
    }
    else if (m_mode == StorageMode::DYNAMIC) {
        // For DYNAMIC mode, we can't get indices without a mesh reference
        // Return an empty set (this might not be ideal, but it's safe)
        m_cellIndices.clear();
    }
    
    return m_cellIndices;
}

const std::vector<bool>& Region::getBitVector() {
    // If in CELL_SET mode, convert to BIT_VECTOR first
    if (m_mode == StorageMode::CELL_SET) {
        ensureBitVector();
    }
    else if (m_mode == StorageMode::DYNAMIC) {
        // For DYNAMIC mode, we can't get a bit vector without a mesh reference
        // Create an empty bit vector (this might not be ideal, but it's safe)
        if (!m_bitVector) {
            m_bitVector = std::make_unique<std::vector<bool>>(m_meshSize, false);
        }
    }
    
    // Ensure we have a valid bit vector to return
    if (!m_bitVector) {
        m_bitVector = std::make_unique<std::vector<bool>>(m_meshSize, false);
    }
    
    return *m_bitVector;
}

Region Region::createUnion(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition for the union
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::UNION
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // Optimize based on storage modes for better performance
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Union of cell sets - merge the sets
            result.m_mode = StorageMode::CELL_SET;
            result.m_cellIndices = m_cellIndices;
            result.m_cellIndices.insert(other.m_cellIndices.begin(), other.m_cellIndices.end());
        } 
        else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Union of bit vectors - OR the vectors
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] || (*other.m_bitVector)[i];
            }
        }
        else {
            // DYNAMIC mode or mixed modes
            result.m_mode = StorageMode::DYNAMIC;
        }
    } 
    else {
        // Mixed storage modes - use DYNAMIC for now
        // A more optimized implementation could convert one storage format to match the other
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

Region Region::createIntersection(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition for the intersection
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::INTERSECTION
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // Optimize based on storage modes for better performance
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Intersection of cell sets - compute set intersection
            result.m_mode = StorageMode::CELL_SET;
            
            // Add indices that are in both sets
            for (int idx : m_cellIndices) {
                if (other.m_cellIndices.find(idx) != other.m_cellIndices.end()) {
                    result.m_cellIndices.insert(idx);
                }
            }
        } 
        else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Intersection of bit vectors - AND the vectors
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] && (*other.m_bitVector)[i];
            }
        }
        else {
            // DYNAMIC mode or mixed modes
            result.m_mode = StorageMode::DYNAMIC;
        }
    } 
    else {
        // Mixed storage modes - use DYNAMIC for now
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

Region Region::createDifference(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition for the difference
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::DIFFERENCE
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // Optimize based on storage modes for better performance
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Difference of cell sets - set difference
            result.m_mode = StorageMode::CELL_SET;
            
            // Add indices that are in this set but not in the other set
            for (int idx : m_cellIndices) {
                if (other.m_cellIndices.find(idx) == other.m_cellIndices.end()) {
                    result.m_cellIndices.insert(idx);
                }
            }
        } 
        else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Difference of bit vectors - A AND (NOT B)
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] && !(*other.m_bitVector)[i];
            }
        }
        else {
            // DYNAMIC mode or mixed modes
            result.m_mode = StorageMode::DYNAMIC;
        }
    } 
    else {
        // Mixed storage modes - use DYNAMIC for now
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

void Region::optimizeStorage(double threshold) {
    // If no mesh size is known, we can't optimize
    if (m_meshSize == 0) return;
    
    // Dynamic mode doesn't need optimization
    if (m_mode == StorageMode::DYNAMIC) {
        return;
    }
    
    // Calculate density (fraction of cells in the region)
    double density = static_cast<double>(size()) / m_meshSize;
    
    // Switch mode based on density
    if (density > threshold && m_mode == StorageMode::CELL_SET) {
        // Dense region - convert to bit vector for efficiency
        // Set the mode first, then ensure the bit vector is created and populated
        m_mode = StorageMode::BIT_VECTOR;
        ensureBitVector();
        // Clear the cell indices to save memory
        // Note: only do this if we're not in debug mode
        #ifndef DEBUG_REGION
        m_cellIndices.clear();
        #endif
    } 
    else if (density <= threshold && m_mode == StorageMode::BIT_VECTOR) {
        // Sparse region - convert to cell set for efficiency
        // Set the mode first, then ensure the cell set is created and populated
        m_mode = StorageMode::CELL_SET;
        ensureCellSet();
        // Clear the bit vector to save memory
        // Note: only do this if we're not in debug mode
        #ifndef DEBUG_REGION
        m_bitVector.reset();
        #endif
    }
}

void Region::ensureBitVector() {
    // If bit vector already exists, nothing to do
    if (m_bitVector) {
        return;
    }
    
    // Create a new bit vector of the appropriate size
    m_bitVector = std::make_unique<std::vector<bool>>(m_meshSize, false);
    
    // If we have cell indices, populate the bit vector
    for (int idx : m_cellIndices) {
        if (idx >= 0 && static_cast<size_t>(idx) < m_meshSize) {
            (*m_bitVector)[idx] = true;
        }
    }
}

void Region::ensureCellSet() {
    // If we don't have a bit vector, nothing to do
    if (!m_bitVector) {
        return;
    }
    
    // Clear any existing cell indices to start fresh
    m_cellIndices.clear();
    
    // Populate the cell set from the bit vector
    for (size_t i = 0; i < m_bitVector->size(); ++i) {
        if ((*m_bitVector)[i]) {
            m_cellIndices.insert(static_cast<int>(i));
        }
    }
}

