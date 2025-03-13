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
#include "Region.h"
#include "RegionOperations.h"
#include <vector>
#include <algorithm>
#include <execution>

namespace mesh {

/**
 * @brief Base enum for standard traversal orders
 */
enum class RegionTraversalOrder {
    NATURAL,     ///< Natural order (as stored in region)
    ROW_MAJOR,   ///< Row-by-row traversal (good for row-major fields)
    COLUMN_MAJOR, ///< Column-by-column traversal (good for column-major fields)
    BLOCKED,     ///< Blocked traversal for cache optimization
    Z_ORDER      ///< Z-order curve for better cache coherence
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
     * @brief Virtual destructor
     */
    virtual ~TraversalStrategy() = default;
    
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
requires MeshConcept<MeshT, CellT>
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
        // For efficiency, get the indices from the region
        const auto& indices = region.getCellIndices();
        
        // Get a const_iterator to the beginning of the set
        auto it = indices.begin();
        
        // Advance the iterator by currentIndex positions
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
requires MeshConcept<MeshT, CellT>
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
        // For row major, we need to get all indices, sort them, and then pick the right one
        const auto& cellIndices = region.getCellIndices();
        std::vector<int> sortedIndices(cellIndices.begin(), cellIndices.end());
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
requires MeshConcept<MeshT, CellT>
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
        // For column major, we need to get all indices, sort them, and then pick the right one
        const auto& cellIndices = region.getCellIndices();
        std::vector<int> sortedIndices(cellIndices.begin(), cellIndices.end());
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
 * @brief Blocked traversal for better cache utilization
 * 
 * This traversal strategy divides the domain into blocks and traverses
 * cells within each block before moving to the next, improving cache locality.
 * The block size is typically chosen to fit within L1 or L2 cache for optimal
 * performance with stencil operations and other algorithms with 2D spatial locality.
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
class BlockedTraversal : public TraversalStrategy<MeshT, CellT> {
public:
    /**
     * @brief Construct a new Blocked Traversal
     * 
     * @param mesh Reference to the mesh
     * @param blockSizeX Block size in x-direction (default optimized for common L1 cache sizes)
     * @param blockSizeY Block size in y-direction (default optimized for common L1 cache sizes)
     */
    BlockedTraversal(MeshT& mesh, int blockSizeX = 16, int blockSizeY = 16)
        : TraversalStrategy<MeshT, CellT>(mesh), m_blockSizeX(blockSizeX), m_blockSizeY(blockSizeY) {}
    
    void sortIndices(std::vector<int>& indices) const override {
        // Convert indices to i,j coordinates
        std::vector<std::tuple<int, int, int>> coords;  // (original index, i, j)
        coords.reserve(indices.size());
        
        for (int idx : indices) {
            int i = idx % this->m_mesh.nx();
            int j = idx / this->m_mesh.nx();
            coords.emplace_back(idx, i, j);
        }
        
        // Sort by block first, then by row within block, then by column
        std::sort(coords.begin(), coords.end(), [this](const auto& a, const auto& b) {
            int blockA_j = std::get<2>(a) / m_blockSizeY;
            int blockA_i = std::get<1>(a) / m_blockSizeX;
            int blockB_j = std::get<2>(b) / m_blockSizeY;
            int blockB_i = std::get<1>(b) / m_blockSizeX;
            
            // Compare blocks in row-major order
            if (blockA_j != blockB_j) return blockA_j < blockB_j;
            if (blockA_i != blockB_i) return blockA_i < blockB_i;
            
            // Within the same block, use row-major order
            if (std::get<2>(a) != std::get<2>(b)) return std::get<2>(a) < std::get<2>(b);
            return std::get<1>(a) < std::get<1>(b);
        });
        
        // Extract the sorted indices
        for (size_t i = 0; i < coords.size(); ++i) {
            indices[i] = std::get<0>(coords[i]);
        }
    }
    
    void sortCells(std::vector<CellT*>& cells) const override {
        std::sort(cells.begin(), cells.end(), [this](const CellT* a, const CellT* b) {
            int blockA_j = a->j() / m_blockSizeY;
            int blockA_i = a->i() / m_blockSizeX;
            int blockB_j = b->j() / m_blockSizeY;
            int blockB_i = b->i() / m_blockSizeX;
            
            // Compare blocks in row-major order
            if (blockA_j != blockB_j) return blockA_j < blockB_j;
            if (blockA_i != blockB_i) return blockA_i < blockB_i;
            
            // Within the same block, use row-major order
            if (a->j() != b->j()) return a->j() < b->j();
            return a->i() < b->i();
        });
    }
    
    CellT* getNextCell(const Region& region, int currentIndex) const override {
        // For blocked traversal, we need to get all indices, sort them, and then pick the right one
        const auto& cellIndices = region.getCellIndices();
        std::vector<int> sortedIndices(cellIndices.begin(), cellIndices.end());
        sortIndices(sortedIndices);
        
        if (currentIndex >= static_cast<int>(sortedIndices.size())) {
            return nullptr;
        }
        
        int linearIndex = sortedIndices[currentIndex];
        int i = linearIndex % this->m_mesh.nx();
        int j = linearIndex / this->m_mesh.nx();
        
        return this->m_mesh.getTypedCell(i, j);
    }
    
private:
    int m_blockSizeX;  ///< Block size in x-direction
    int m_blockSizeY;  ///< Block size in y-direction
};

/**
 * @brief Z-Order curve traversal for improved cache coherence
 * 
 * This traversal follows a Z-order curve (Morton code) which preserves spatial
 * locality in multiple dimensions. This recursive "Z" pattern provides better
 * cache behavior for many algorithms by ensuring that nearby points in 2D space
 * remain close in memory, reducing cache misses for operations with 2D locality.
 * 
 * Z-order curves are particularly effective for:
 * - Multi-dimensional neighbor queries
 * - Hierarchical algorithms (multi-grid, AMR)
 * - Cache-oblivious algorithms
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
class ZOrderTraversal : public TraversalStrategy<MeshT, CellT> {
public:
    using TraversalStrategy<MeshT, CellT>::TraversalStrategy;
    
    void sortIndices(std::vector<int>& indices) const override {
        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            int aI = a % this->m_mesh.nx();
            int aJ = a / this->m_mesh.nx();
            int bI = b % this->m_mesh.nx();
            int bJ = b / this->m_mesh.nx();
            
            uint32_t mortonA = encodeMorton2D(aI, aJ);
            uint32_t mortonB = encodeMorton2D(bI, bJ);
            
            return mortonA < mortonB;
        });
    }
    
    void sortCells(std::vector<CellT*>& cells) const override {
        std::sort(cells.begin(), cells.end(), [this](const CellT* a, const CellT* b) {
            uint32_t mortonA = encodeMorton2D(a->i(), a->j());
            uint32_t mortonB = encodeMorton2D(b->i(), b->j());
            
            return mortonA < mortonB;
        });
    }
    
    CellT* getNextCell(const Region& region, int currentIndex) const override {
        // For Z-order, we need to get all indices, sort them, and then pick the right one
        const auto& cellIndices = region.getCellIndices();
        std::vector<int> sortedIndices(cellIndices.begin(), cellIndices.end());
        sortIndices(sortedIndices);
        
        if (currentIndex >= static_cast<int>(sortedIndices.size())) {
            return nullptr;
        }
        
        int linearIndex = sortedIndices[currentIndex];
        int i = linearIndex % this->m_mesh.nx();
        int j = linearIndex / this->m_mesh.nx();
        
        return this->m_mesh.getTypedCell(i, j);
    }
    
private:
    /**
     * @brief Interleave bits to create a Morton code (Z-order)
     * 
     * This function interleaves the bits of the x and y coordinates to create a
     * Morton code, which maps 2D coordinates to 1D while preserving spatial locality.
     * The resulting ordering follows a Z-shaped pattern at all scales.
     * 
     * @param x X-coordinate (must be <= 2^16-1)
     * @param y Y-coordinate (must be <= 2^16-1)
     * @return Morton code as a 32-bit unsigned int
     */
    static uint32_t encodeMorton2D(uint16_t x, uint16_t y) {
        return (expandBits(y) << 1) + expandBits(x);
    }
    
    /**
     * @brief Expand a 16-bit integer to 32 bits by inserting zeros between bits
     * 
     * This helper function expands a 16-bit integer into 32 bits by inserting zeros
     * between each of the original bits. This is a key step in creating Morton codes.
     * 
     * Example: 
     * Input:  0000000000000101 (5 in binary)
     * Output: 00000000000000000000000000010001 (interleaved with zeros)
     * 
     * @param v Value to expand (must be <= 2^16-1)
     * @return Expanded value with zeros interleaved between original bits
     */
    static uint32_t expandBits(uint16_t v) {
        uint32_t x = static_cast<uint32_t>(v);
        x = (x | (x << 8)) & 0x00FF00FF;  // Interleave bytes: 0000000000000101 -> 0000000000010001
        x = (x | (x << 4)) & 0x0F0F0F0F;  // Interleave 4-bit chunks
        x = (x | (x << 2)) & 0x33333333;  // Interleave 2-bit chunks
        x = (x | (x << 1)) & 0x55555555;  // Interleave individual bits
        return x;
    }
};

/**
 * @brief Factory function to create traversal strategy based on order
 * 
 * This function creates the appropriate traversal strategy based on the
 * specified order, allowing client code to easily switch between different
 * traversal patterns without knowing the implementation details.
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @param mesh Reference to the mesh
 * @param order Desired traversal order
 * @param blockSizeX Block size in x-direction (for blocked traversal)
 * @param blockSizeY Block size in y-direction (for blocked traversal)
 * @return Unique pointer to traversal strategy
 */
template <typename MeshT, typename CellT>
requires MeshConcept<MeshT, CellT>
std::unique_ptr<TraversalStrategy<MeshT, CellT>> createTraversalStrategy(
    MeshT& mesh,
    RegionTraversalOrder order,
    int blockSizeX = 16,
    int blockSizeY = 16
) {
    switch (order) {
        case RegionTraversalOrder::NATURAL:
            return std::make_unique<NaturalOrderTraversal<MeshT, CellT>>(mesh);
            
        case RegionTraversalOrder::ROW_MAJOR:
            return std::make_unique<RowMajorTraversal<MeshT, CellT>>(mesh);
            
        case RegionTraversalOrder::COLUMN_MAJOR:
            return std::make_unique<ColumnMajorTraversal<MeshT, CellT>>(mesh);
            
        case RegionTraversalOrder::BLOCKED:
            return std::make_unique<BlockedTraversal<MeshT, CellT>>(mesh, blockSizeX, blockSizeY);
            
        case RegionTraversalOrder::Z_ORDER:
            return std::make_unique<ZOrderTraversal<MeshT, CellT>>(mesh);
            
        default:
            // Default to natural order if an unknown order is specified
            return std::make_unique<NaturalOrderTraversal<MeshT, CellT>>(mesh);
    }
}

/**
 * @brief Apply a function to each cell in a region with specific traversal order
 * 
 * This utility function provides a high-level interface for iterating over a region
 * with a specified traversal order, making it easy to optimize memory access patterns
 * for different algorithms without modifying the algorithm itself.
 * 
 * Example usage:
 * ```cpp
 * // Process cells in a region using Z-order traversal for better cache locality
 * forEachCellInRegionOrdered<MyMesh, MyCell>(mesh, region, [](MyCell* cell) {
 *     // Process cell here...
 * }, RegionTraversalOrder::Z_ORDER);
 * ```
 * 
 * @tparam MeshT Type of mesh
 * @tparam CellT Type of cell
 * @tparam FuncT Type of function to apply
 * @param mesh Mesh containing the cells
 * @param region Region to iterate over
 * @param func Function to apply to each cell
 * @param order Traversal order to use
 */
template <typename MeshT, typename CellT, typename FuncT>
requires std::invocable<FuncT, CellT*>
void forEachCellInRegionOrdered(
    MeshT& mesh,
    const Region& region,
    FuncT func,
    RegionTraversalOrder order = RegionTraversalOrder::NATURAL
) {
    // Create appropriate traversal strategy
    auto strategy = createTraversalStrategy<MeshT, CellT>(mesh, order);
    
    // Get all cells in the region
    std::vector<CellT*> cells;
    cells.reserve(region.size());  // Reserve space for efficiency
    
    // Get cells using existing function from RegionOperations.h
    forEachCellInRegion<MeshT, CellT>(mesh, region, [&cells](CellT* cell) {
        cells.push_back(cell);
    });
    
    // Sort cells according to the strategy
    strategy->sortCells(cells);
    
    // Apply function to each cell
    for (CellT* cell : cells) {
        if (cell) {
            func(cell);
        }
    }
}

} // namespace mesh

#endif // REGION_TRAVERSAL_H

