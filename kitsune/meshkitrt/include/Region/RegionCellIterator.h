/**
 * @file RegionCellIterator.h
 * @brief Iterator classes for traversing cells in a region
 * 
 * This file provides iterator classes that enable range-based for loop
 * support and standard algorithm compatibility for region-based operations.
 */

#ifndef REGION_CELL_ITERATOR_H
#define REGION_CELL_ITERATOR_H

#include "Region.h"
#include "RegionConcepts.h"
#include "RegionTraversal.h"
#include <memory>
#include <vector>
#include <iterator>

namespace mesh {

/**
 * @brief Iterator for cells in a region
 * 
 * This iterator provides STL-compatible iteration over cells in a region,
 * supporting different traversal strategies for optimized memory access.
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
class RegionCellIterator {
public:
    // Iterator type definitions for STL compatibility
    using iterator_category = std::forward_iterator_tag;
    using value_type = CellT*;
    using difference_type = std::ptrdiff_t;
    using pointer = CellT**;
    using reference = CellT*&;
    
    /**
     * @brief Construct a new Region Cell Iterator for begin position
     * 
     * @param mesh Reference to mesh
     * @param region Reference to region
     * @param order Traversal order to use
     */
    RegionCellIterator(
        MeshT& mesh,
        const Region& region,
        RegionTraversalOrder order = RegionTraversalOrder::NATURAL
    ) : m_mesh(mesh),
        m_region(region),
        m_traversalStrategy(createTraversalStrategy<MeshT, CellT>(mesh, order)),
        m_currentIndex(0),
        m_currentCell(nullptr)
    {
        // Get the first cell immediately
        advance();
    }
    
    /**
     * @brief Construct a new Region Cell Iterator for end position
     * 
     * @param mesh Reference to mesh
     * @param region Reference to region
     * @param endSentinel Special value indicating end iterator
     */
    RegionCellIterator(
        MeshT& mesh,
        const Region& region,
        int endSentinel
    ) : m_mesh(mesh),
        m_region(region),
        m_traversalStrategy(nullptr),
        m_currentIndex(endSentinel),
        m_currentCell(nullptr)
    {
        // End iterator doesn't need a traversal strategy or current cell
    }
    
    /**
     * @brief Dereference operator
     * 
     * @return Current cell pointer
     */
    CellT* operator*() const {
        return m_currentCell;
    }
    
    /**
     * @brief Pre-increment operator
     * 
     * @return Reference to this iterator after advancing
     */
    RegionCellIterator& operator++() {
        advance();
        return *this;
    }
    
    /**
     * @brief Post-increment operator
     * 
     * @return Copy of iterator before advancing
     */
    RegionCellIterator operator++(int) {
        RegionCellIterator tmp = *this;
        advance();
        return tmp;
    }
    
    /**
     * @brief Equality comparison operator
     * 
     * @param other Iterator to compare with
     * @return true if iterators point to the same position
     */
    bool operator==(const RegionCellIterator& other) const {
        // End iterator is a special case
        if (m_currentIndex == -1 || other.m_currentIndex == -1) {
            // If one is end, compare the current cell
            return (m_currentCell == nullptr && other.m_currentCell == nullptr);
        }
        
        // Otherwise compare indices
        return m_currentIndex == other.m_currentIndex;
    }
    
    /**
     * @brief Inequality comparison operator
     * 
     * @param other Iterator to compare with
     * @return true if iterators point to different positions
     */
    bool operator!=(const RegionCellIterator& other) const {
        return !(*this == other);
    }
    
private:
    MeshT& m_mesh;                                                  ///< Reference to mesh
    const Region& m_region;                                         ///< Reference to region
    std::unique_ptr<TraversalStrategy<MeshT, CellT>> m_traversalStrategy; ///< Traversal strategy
    int m_currentIndex;                                             ///< Current index in traversal
    CellT* m_currentCell;                                           ///< Current cell pointer
    
    /**
     * @brief Advance to the next valid cell
     */
    void advance() {
        if (!m_traversalStrategy) {
            // End iterator or invalid case
            m_currentCell = nullptr;
            return;
        }
        
        // Get the next cell from the traversal strategy
        m_currentCell = m_traversalStrategy->getNextCell(m_region, m_currentIndex);
        
        if (m_currentCell) {
            // Valid cell found, increment index
            ++m_currentIndex;
        }
        else {
            // No more cells, mark as end iterator
            m_currentIndex = -1;
        }
    }
};

/**
 * @brief Range class for enabling range-based for loop over region cells
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
class RegionCellRange {
public:
    /**
     * @brief Construct a new Region Cell Range
     * 
     * @param mesh Reference to mesh
     * @param region Reference to region
     * @param order Traversal order to use
     */
    RegionCellRange(
        MeshT& mesh,
        const Region& region,
        RegionTraversalOrder order = RegionTraversalOrder::NATURAL
    ) : m_mesh(mesh), m_region(region), m_order(order) {}
    
    /**
     * @brief Get begin iterator
     * 
     * @return Begin iterator
     */
    RegionCellIterator<MeshT, CellT> begin() const {
        return RegionCellIterator<MeshT, CellT>(m_mesh, m_region, m_order);
    }
    
    /**
     * @brief Get end iterator
     * 
     * @return End iterator
     */
    RegionCellIterator<MeshT, CellT> end() const {
        return RegionCellIterator<MeshT, CellT>(m_mesh, m_region, -1);
    }
    
private:
    MeshT& m_mesh;                     ///< Reference to mesh
    const Region& m_region;            ///< Reference to region
    RegionTraversalOrder m_order;      ///< Traversal order to use
};

/**
 * @brief Conversion function for BitArray to CellSet representation
 * 
 * This is useful for algorithms that need to process bit arrays as cell sets.
 * 
 * @tparam MeshT Type of mesh
 * @param mesh Reference to mesh
 * @param bitArray BitArray to convert
 * @return Vector of linear indices for set bits
 */
template <typename MeshT>
std::vector<int> bitArrayToCellIndices(
    const MeshT& mesh,
    const BitArray& bitArray
) {
    std::vector<int> indices;
    
    // Reserve space based on bit count (if available)
    if (bitArray.count() > 0) {
        indices.reserve(bitArray.count());
    }
    
    // Iterate through set bits and add their indices
    for (size_t idx = bitArray.findFirst(); idx < bitArray.size(); idx = bitArray.findNext(idx)) {
        indices.push_back(static_cast<int>(idx));
    }
    
    return indices;
}

/**
 * @brief Conversion function for CellSet to BitArray representation
 * 
 * This is useful for algorithms that need to process cell sets as bit arrays.
 * 
 * @tparam MeshT Type of mesh
 * @param mesh Reference to mesh
 * @param cellIndices Set of cell indices to convert
 * @return BitArray representation
 */
template <typename MeshT>
BitArray cellIndicesToBitArray(
    const MeshT& mesh,
    const std::unordered_set<int>& cellIndices
) {
    size_t meshSize = mesh.nx() * mesh.ny();
    BitArray bitArray(meshSize, false);
    
    // Set bits for each cell index
    for (int idx : cellIndices) {
        if (idx >= 0 && static_cast<size_t>(idx) < meshSize) {
            bitArray.set(idx, true);
        }
    }
    
    return bitArray;
}

} // namespace mesh

#endif // REGION_CELL_ITERATOR_H


