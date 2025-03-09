/**
 * @file RegionTraversal.h
 * @brief Define region traversal strategies using the concepts-based design
 * 
 * This file provides traversal strategies for region iteration,
 * allowing for different iteration orders (row-major, column-major, etc.)
 * and optimizing for different memory access patterns.
 */

#ifndef REGION_TRAVERSAL_H
#define REGION_TRAVERSAL_H

#include "RegionConcepts.h"
#include <vector>
#include <algorithm>

/**
 * @brief Base enum for standard traversal orders
 */
enum class RegionTraversalOrder {
    NATURAL,     ///< Natural order (as stored in region)
    ROW_MAJOR,   ///< Row-by-row traversal (good for row-major fields)
    COLUMN_MAJOR ///< Column-by-column traversal (good for column-major fields)
};

/**
 * @brief Base class for traversal strategies
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
class TraversalStrategy {
public:
    /**
     * @brief Construct a new Traversal Strategy
     * 
     * @param mesh Reference to the mesh
     */
    explicit TraversalStrategy(MeshT& mesh) : m_mesh(mesh) {}
    
    /**
     * @brief Sort a vector of cell indices based on the strategy
     * 
     * @param indices Vector of linear indices to sort
     */
    virtual void sortIndices(std::vector<int>& indices) const = 0;
    
    /**
     * @brief Sort a vector of cells based on the strategy
     * 
     * @param cells Vector of cell pointers to sort
     */
    virtual void sortCells(std::vector<CellT*>& cells) const = 0;
    
    /**
     * @brief Get the next cell in traversal order
     * 
     * @param region The region being traversed
     * @param currentIndex Current index in traversal
     * @return Pointer to the next cell, or nullptr if done
     */
    virtual CellT* getNextCell(const Region& region, int currentIndex) const = 0;
    
protected:
    MeshT& m_mesh;  ///< Reference to the mesh
};

/**
 * @brief Natural order traversal (as stored in region)
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
class NaturalOrderTraversal : public TraversalStrategy<MeshT, CellT> {
public:
    using TraversalStrategy<MeshT, CellT>::TraversalStrategy;
    
    void sortIndices(std::vector<int>& indices) const override {
        // No sorting needed for natural order
    }
    
    void sortCells(std::vector<CellT*>& cells) const override {
        // No sorting needed for natural order
    }
    
    CellT* getNextCell(const Region& region, int currentIndex) const override {
        // Implementation depends on the region's storage mode
        // This is a simplified version that works with the cell indices
        auto indices = region.getCellIndices();
        auto it = indices.begin();
        std::advance(it, currentIndex);
        
        if (it == indices.end()) {
            return nullptr;
        }
        
        int linearIndex = *it;
        int i = linearIndex % this->m_mesh.nx();
        int j = linearIndex / this->m_mesh.nx();
        
        return this->m_mesh.getTypedCell(i, j);
    }
};

/**
 * @brief Row-major traversal (good for row-major storage)
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
class RowMajorTraversal : public TraversalStrategy<MeshT, CellT> {
public:
    using TraversalStrategy<MeshT, CellT>::TraversalStrategy;
    
    void sortIndices(std::vector<int>& indices) const override {
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            int aJ = a / this->m_mesh.nx();
            int aI = a % this->m_mesh.nx();
            int bJ = b / this->m_mesh.nx();
            int bI = b % this->m_mesh.nx();
            
            if (aJ != bJ) return aJ < bJ;
            return aI < bI;
        });
    }
    
    void sortCells(std::vector<CellT*>& cells) const override {
        std::sort(cells.begin(), cells.end(), [](const CellT* a, const CellT* b) {
            if (a->j() != b->j()) return a->j() < b->j();
            return a->i() < b->i();
        });
    }
    
    CellT* getNextCell(const Region& region, int currentIndex) const override {
        // Similar to NaturalOrderTraversal but with row-major ordering
        auto indices = region.getCellIndices();
        std::vector<int> sortedIndices(indices.begin(), indices.end());
        sortIndices(sortedIndices);
        
        if (currentIndex >= static_cast<int>(sortedIndices.size())) {
            return nullptr;
        }
        
        int linearIndex = sortedIndices[currentIndex];
        int i = linearIndex % this->m_mesh.nx();
        int j = linearIndex / this->m_mesh.nx();
        
        return this->m_mesh.getTypedCell(i, j);
    }
};

/**
 * @brief Column-major traversal (good for column-major storage)
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
class ColumnMajorTraversal : public TraversalStrategy<MeshT, CellT> {
public:
    using TraversalStrategy<MeshT, CellT>::TraversalStrategy;
    
    void sortIndices(std::vector<int>& indices) const override {
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            int aI = a % this->m_mesh.nx();
            int aJ = a / this->m_mesh.nx();
            int bI = b % this->m_mesh.nx();
            int bJ = b / this->m_mesh.nx();
            
            if (aI != bI) return aI < bI;
            return aJ < bJ;
        });
    }
    
    void sortCells(std::vector<CellT*>& cells) const override {
        std::sort(cells.begin(), cells.end(), [](const CellT* a, const CellT* b) {
            if (a->i() != b->i()) return a->i() < b->i();
            return a->j() < b->j();
        });
    }
    
    CellT* getNextCell(const Region& region, int currentIndex) const override {
        // Similar implementation but with column-major ordering
        auto indices = region.getCellIndices();
        std::vector<int> sortedIndices(indices.begin(), indices.end());
        sortIndices(sortedIndices);
        
        if (currentIndex >= static_cast<int>(sortedIndices.size())) {
            return nullptr;
        }
        
        int linearIndex = sortedIndices[currentIndex];
        int i = linearIndex % this->m_mesh.nx();
        int j = linearIndex / this->m_mesh.nx();
        
        return this->m_mesh.getTypedCell(i, j);
    }
};

/**
 * @brief Factory function to create traversal strategy based on order
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @param mesh Reference to the mesh
 * @param order Desired traversal order
 * @return Unique pointer to traversal strategy
 */
template <typename MeshT, typename CellT>
requires Mesh<MeshT, CellT>
std::unique_ptr<TraversalStrategy<MeshT, CellT>> createTraversalStrategy(
    MeshT& mesh,
    RegionTraversalOrder order
) {
    switch (order) {
        case RegionTraversalOrder::NATURAL:
            return std::make_unique<NaturalOrderTraversal<MeshT, CellT>>(mesh);
            
        case RegionTraversalOrder::ROW_MAJOR:
            return std::make_unique<RowMajorTraversal<MeshT, CellT>>(mesh);
            
        case RegionTraversalOrder::COLUMN_MAJOR:
            return std::make_unique<ColumnMajorTraversal<MeshT, CellT>>(mesh);
            
        default:
            return std::make_unique<NaturalOrderTraversal<MeshT, CellT>>(mesh);
    }
}

#endif // REGION_TRAVERSAL_H


