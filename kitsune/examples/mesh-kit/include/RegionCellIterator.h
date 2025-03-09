/**
 * @file RegionCellIterator.h
 * @brief Iterator for traversing cells in a region with concepts support
 * 
 * This file defines iterator classes for efficiently traversing cells
 * in a region with STL-compatible syntax, using C++20 concepts.
 */

#ifndef REGION_CELL_ITERATOR_H
#define REGION_CELL_ITERATOR_H

#include "Region.h"
#include "RegionConcepts.h"
#include "RegionTraversal.h"
#include <concepts>
#include <iterator>
#include <vector>
#include <memory>

/**
 * @brief Iterator for cells in a region
 * 
 * This iterator provides STL-compatible traversal of cells in a region.
 * 
 * @tparam MeshT The mesh type
 * @tparam CellT The cell type
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
class RegionCellIterator {
public:
    // STL iterator type traits
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellT*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellT**;
    using reference = CellT*&;
    
    /**
     * @brief Construct an iterator for a region
     * 
     * @param mesh Reference to the mesh
     * @param region Reference to the region
     * @param order Traversal order
     */
    RegionCellIterator(MeshT& mesh, const Region& region, 
                      RegionTraversalOrder order = RegionTraversalOrder::NATURAL)
        : m_mesh(mesh), 
          m_region(&region), 
          m_traversalStrategy(createTraversalStrategy<MeshT, CellT>(mesh, order)),
          m_currentIndex(0),
          m_isEnd(false) {
        
        // Get the first valid cell
        m_currentCell = m_traversalStrategy->getNextCell(*m_region, m_currentIndex);
        
        // Check if the region is empty
        if (!m_currentCell) {
            m_isEnd = true;
        }
    }
    
    /**
     * @brief Construct an end iterator
     * 
     * @param mesh Reference to the mesh
     */
    static RegionCellIterator endIterator(MeshT& mesh) {
        RegionCellIterator it(mesh);
        it.m_isEnd = true;
        return it;
    }
    
    /**
     * @brief Dereference operator
     * 
     * @return Pointer to current cell
     */
    value_type operator*() const {
        return m_currentCell;
    }
    
    /**
     * @brief Pre-increment operator
     * 
     * @return Reference to this iterator
     */
    RegionCellIterator& operator++() {
        if (m_isEnd) {
            return *this;
        }
        
        // Move to next cell
        m_currentIndex++;
        m_currentCell = m_traversalStrategy->getNextCell(*m_region, m_currentIndex);
        
        // Check if we've reached the end
        if (!m_currentCell) {
            m_isEnd = true;
        }
        
        return *this;
    }
    
    /**
     * @brief Post-increment operator
     * 
     * @return Iterator before increment
     */
    RegionCellIterator operator++(int) {
        RegionCellIterator tmp = *this;
        ++(*this);
        return tmp;
    }
    
    /**
     * @brief Equality comparison
     * 
     * @param other Other iterator
     * @return true if iterators are equal
     */
    bool operator==(const RegionCellIterator& other) const {
        if (m_isEnd && other.m_isEnd) return true;
        if (m_isEnd || other.m_isEnd) return false;
        
        return m_currentCell == other.m_currentCell;
    }
    
    /**
     * @brief Inequality comparison
     * 
     * @param other Other iterator
     * @return true if iterators are not equal
     */
    bool operator!=(const RegionCellIterator& other) const {
        return !(*this == other);
    }
    
    /**
     * @brief Get the current i-index
     * 
     * @return Current i-index
     */
    int i() const { 
        return m_currentCell ? m_currentCell->i() : -1; 
    }
    
    /**
     * @brief Get the current j-index
     * 
     * @return Current j-index
     */
    int j() const { 
        return m_currentCell ? m_currentCell->j() : -1; 
    }
    
private:
    // Private constructor for end iterator
    RegionCellIterator(MeshT& mesh) 
        : m_mesh(mesh), 
          m_region(nullptr),
          m_traversalStrategy(createTraversalStrategy<MeshT, CellT>(mesh, RegionTraversalOrder::NATURAL)),
          m_currentCell(nullptr),
          m_currentIndex(0),
          m_isEnd(true) {}
    
    MeshT& m_mesh;                                                 ///< Reference to the mesh
    const Region* m_region;                                        ///< Pointer to the region
    std::unique_ptr<TraversalStrategy<MeshT, CellT>> m_traversalStrategy; ///< Traversal strategy
    CellT* m_currentCell;                                          ///< Current cell pointer
    int m_currentIndex;                                           ///< Current index in traversal
    bool m_isEnd;                                                 ///< Whether this is an end iterator
};

/**
 * @brief Range-based for loop support for region cells
 * 
 * This class provides a convenient interface for using range-based
 * for loops with regions.
 * 
 * @tparam MeshT The mesh type
 * @tparam CellT The cell type
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
class RegionCellRange {
public:
    using iterator = RegionCellIterator<MeshT, CellT>;
    
    /**
     * @brief Construct a range for a region
     * 
     * @param mesh Reference to the mesh
     * @param region Reference to the region
     * @param order Traversal order
     */
    RegionCellRange(MeshT& mesh, const Region& region, 
                   RegionTraversalOrder order = RegionTraversalOrder::NATURAL)
        : m_mesh(mesh), m_region(region), m_order(order) {}
    
    /**
     * @brief Get an iterator to the first cell
     * 
     * @return Iterator to the first cell
     */
    iterator begin() { 
        return iterator(m_mesh, m_region, m_order); 
    }
    
    /**
     * @brief Get an end iterator
     * 
     * @return End iterator
     */
    iterator end() { 
        return iterator::endIterator(m_mesh); 
    }
    
private:
    MeshT& m_mesh;              ///< Reference to the mesh
    const Region& m_region;      ///< Reference to the region
    RegionTraversalOrder m_order; ///< Traversal order
};

#endif // REGION_CELL_ITERATOR_H
       //
