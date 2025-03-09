/**
 * @file Region.cpp
 * @brief Implementation of the Region class
 */

#include "Region.h"
#include "MeshBase.h"
#include <algorithm>
#include <stdexcept>

bool Region::contains(const CellBase* cell) const {
    if (!cell) return false;
    
    // If using dynamic mode, always check the definition
    if (m_mode == StorageMode::DYNAMIC) {
        return m_definition->contains(cell);
    }
    
    // Otherwise, check stored membership
    return containsIndex(cell->linearIndex());
}

bool Region::containsIndex(int linearIndex) const {
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return false;
    }
    
    switch (m_mode) {
        case StorageMode::CELL_SET:
            return m_cellIndices.find(linearIndex) != m_cellIndices.end();
        
        case StorageMode::BIT_VECTOR:
            return m_bitVector && (*m_bitVector)[linearIndex];
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we can't check by index directly
            // This shouldn't be called for dynamic mode
            return false;
        
        default:
            return false;
    }
}

void Region::addCell(const CellBase* cell) {
    if (!cell) return;
    
    addCellIndex(cell->linearIndex());
}

void Region::addCellIndex(int linearIndex) {
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return;
    }
    
    switch (m_mode) {
        case StorageMode::CELL_SET:
            m_cellIndices.insert(linearIndex);
            break;
        
        case StorageMode::BIT_VECTOR:
            ensureBitVector();
            (*m_bitVector)[linearIndex] = true;
            break;
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we can't add by index
            // Switch to cell set mode
            m_mode = StorageMode::CELL_SET;
            m_cellIndices.insert(linearIndex);
            break;
    }
}

void Region::removeCell(const CellBase* cell) {
    if (!cell) return;
    
    removeCellIndex(cell->linearIndex());
}

void Region::removeCellIndex(int linearIndex) {
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
        return;
    }
    
    switch (m_mode) {
        case StorageMode::CELL_SET:
            m_cellIndices.erase(linearIndex);
            break;
        
        case StorageMode::BIT_VECTOR:
            if (m_bitVector) {
                (*m_bitVector)[linearIndex] = false;
            }
            break;
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we can't remove by index
            // Convert to explicit storage first
            m_mode = StorageMode::BIT_VECTOR;
            ensureBitVector();
            (*m_bitVector)[linearIndex] = false;
            break;
    }
}

void Region::clear() {
    m_cellIndices.clear();
    if (m_bitVector) {
        m_bitVector->assign(m_bitVector->size(), false);
    }
}

size_t Region::size() const {
    switch (m_mode) {
        case StorageMode::CELL_SET:
            return m_cellIndices.size();
        
        case StorageMode::BIT_VECTOR:
            if (!m_bitVector) return 0;
            return std::count(m_bitVector->begin(), m_bitVector->end(), true);
        
        case StorageMode::DYNAMIC:
            // For dynamic mode, we don't know the size without scanning the mesh
            return 0;
        
        default:
            return 0;
    }
}

const std::unordered_set<int>& Region::getCellIndices() {
    if (m_mode == StorageMode::BIT_VECTOR) {
        // Convert bit vector to cell set
        ensureCellSet();
    }
    
    return m_cellIndices;
}

const std::vector<bool>& Region::getBitVector() {
    if (m_mode == StorageMode::CELL_SET) {
        // Convert cell set to bit vector
        ensureBitVector();
    }
    
    return *m_bitVector;
}

Region Region::createUnion(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::UNION
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // If both regions use the same storage mode, we can optimize
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Union of cell sets
            result.m_mode = StorageMode::CELL_SET;
            result.m_cellIndices = m_cellIndices;
            result.m_cellIndices.insert(other.m_cellIndices.begin(), other.m_cellIndices.end());
        } else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Union of bit vectors
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] || (*other.m_bitVector)[i];
            }
        }
    } else {
        // Mixed storage modes - use the definition for now
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

Region Region::createIntersection(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::INTERSECTION
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // If both regions use the same storage mode, we can optimize
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Intersection of cell sets
            result.m_mode = StorageMode::CELL_SET;
            for (int idx : m_cellIndices) {
                if (other.m_cellIndices.find(idx) != other.m_cellIndices.end()) {
                    result.m_cellIndices.insert(idx);
                }
            }
        } else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Intersection of bit vectors
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] && (*other.m_bitVector)[i];
            }
        }
    } else {
        // Mixed storage modes - use the definition for now
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

Region Region::createDifference(const Region& other, const std::string& name, RegionID newId) const {
    // Create a composite region definition
    auto compDef = std::make_shared<CompositeRegion>(
        name,
        m_definition,
        other.m_definition,
        CompositeRegion::Operation::DIFFERENCE
    );
    
    // Create new region with the composite definition
    Region result(newId, compDef, m_meshSize);
    
    // If both regions use the same storage mode, we can optimize
    if (m_mode == other.m_mode) {
        if (m_mode == StorageMode::CELL_SET) {
            // Difference of cell sets
            result.m_mode = StorageMode::CELL_SET;
            for (int idx : m_cellIndices) {
                if (other.m_cellIndices.find(idx) == other.m_cellIndices.end()) {
                    result.m_cellIndices.insert(idx);
                }
            }
        } else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector && other.m_bitVector) {
            // Difference of bit vectors
            result.m_mode = StorageMode::BIT_VECTOR;
            result.m_bitVector = std::make_unique<std::vector<bool>>(m_bitVector->size(), false);
            
            for (size_t i = 0; i < m_bitVector->size(); ++i) {
                (*result.m_bitVector)[i] = (*m_bitVector)[i] && !(*other.m_bitVector)[i];
            }
        }
    } else {
        // Mixed storage modes - use the definition for now
        result.m_mode = StorageMode::DYNAMIC;
    }
    
    return result;
}

void Region::optimizeStorage(double threshold) {
    // If no mesh size is known, we can't optimize
    if (m_meshSize == 0) return;
    
    if (m_mode == StorageMode::DYNAMIC) {
        // Don't optimize dynamic mode
        return;
    }
    
    // Calculate density (fraction of cells in the region)
    double density = static_cast<double>(size()) / m_meshSize;
    
    // Switch mode based on density
    if (density > threshold && m_mode == StorageMode::CELL_SET) {
        // Convert to bit vector for dense regions
        m_mode = StorageMode::BIT_VECTOR;
        ensureBitVector();
        m_cellIndices.clear();
    } else if (density <= threshold && m_mode == StorageMode::BIT_VECTOR) {
        // Convert to cell set for sparse regions
        m_mode = StorageMode::CELL_SET;
        ensureCellSet();
        m_bitVector.reset();
    }
}

void Region::ensureBitVector() {
    if (!m_bitVector) {
        m_bitVector = std::make_unique<std::vector<bool>>(m_meshSize, false);
        
        // Fill from cell indices if available
        for (int idx : m_cellIndices) {
            if (idx >= 0 && static_cast<size_t>(idx) < m_meshSize) {
                (*m_bitVector)[idx] = true;
            }
        }
    }
}

void Region::ensureCellSet() {
    if (m_bitVector) {
        // Fill cell indices from bit vector
        m_cellIndices.clear();
        
        for (size_t i = 0; i < m_bitVector->size(); ++i) {
            if ((*m_bitVector)[i]) {
                m_cellIndices.insert(static_cast<int>(i));
            }
        }
    }
}

