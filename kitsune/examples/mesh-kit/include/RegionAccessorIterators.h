/**
 * @file AccessorIterators.h
 * @brief Iterator interfaces for Accessor classes
 * 
 * This file defines iterator interfaces for the various accessor classes
 * (MeshAccessor, RegionAccessor, FieldAccessor, CompoundAccessor).
 * These iterators build on the Field iterators to provide consistent and
 * efficient traversal patterns for mesh data.
 */

#ifndef REGION_ACCESSOR_ITERATORS_H
#define REGION_ACCESSOR_ITERATORS_H

#include "AccessorIteratorsCommon.h"
#include <functional>


template <typename MeshType, typename CellType>
class RegionAccessorIterators {
public:
    /**
     * @brief Iterator for cells in a specific region
     */
    class RegionCellIterator {
    public:
        // STL iterator type traits
        using iterator_category = std::forward_iterator_tag;
        using value_type = CellType*;
        using difference_type = std::ptrdiff_t;
        using pointer = CellType**;
        using reference = CellType*&;
        
        /**
         * @brief Construct a new Region Cell Iterator
         * 
         * @param mesh Reference to the mesh
         * @param regionMask Region mask
         * @param i Initial i-index
         * @param j Initial j-index
         */
        RegionCellIterator(MeshType& mesh, RegionMask regionMask, int i = 0, int j = 0)
            : m_mesh(mesh), m_regionMask(regionMask), m_i(i), m_j(j) {
            // Find first cell in region if necessary
            if (i == 0 && j == 0) {
                findNextInRegion();
            }
        }
        
        // Core iterator operations
        value_type operator*() const { return m_mesh.getTypedCell(m_i, m_j); }
        
        RegionCellIterator& operator++() {
            ++m_i;
            if (m_i >= m_mesh.nx()) {
                m_i = 0;
                ++m_j;
            }
            
            findNextInRegion();
            return *this;
        }
        
        RegionCellIterator operator++(int) {
            RegionCellIterator tmp = *this;
            ++(*this);
            return tmp;
        }
        
        // Comparison
        bool operator==(const RegionCellIterator& other) const {
            return m_i == other.m_i && m_j == other.m_j;
        }
        
        bool operator!=(const RegionCellIterator& other) const {
            return !(*this == other);
        }
        
        // Current indices
        int i() const { return m_i; }
        int j() const { return m_j; }
        
    private:
        void findNextInRegion() {
            while (m_j < m_mesh.ny()) {
                CellType* cell = m_mesh.getTypedCell(m_i, m_j);
                if (cell && isCellInRegion(cell, m_regionMask)) {
                    return; // Found a cell in the region
                }
                
                ++m_i;
                if (m_i >= m_mesh.nx()) {
                    m_i = 0;
                    ++m_j;
                }
            }
        }
        
        bool isCellInRegion(const CellType* cell, RegionMask mask) const {
            // This is simplified and would need to match the actual implementation
            // in RegionAccessor::isCellInRegion
            int linearIndex = cell->linearIndex();
            linearIndex %= (sizeof(RegionMask) * 8);
            RegionMask cellBit = RegionMask(1) << linearIndex;
            return (mask & cellBit) != 0;
        }
        
        MeshType& m_mesh;
        RegionMask m_regionMask;
        int m_i, m_j;
    };
    
    /**
     * @brief Range class for cells in a region
     */
    class RegionCellRange {
    public:
        using iterator = RegionCellIterator;
        
        RegionCellRange(MeshType& mesh, RegionMask regionMask) 
            : m_mesh(mesh), m_regionMask(regionMask) {}
        
        iterator begin() { return iterator(m_mesh, m_regionMask); }
        iterator end() { return iterator(m_mesh, m_regionMask, 0, m_mesh.ny()); }
        
    private:
        MeshType& m_mesh;
        RegionMask m_regionMask;
    };
};

#endif // REGION_ACCESSOR_ITERATORS_H
