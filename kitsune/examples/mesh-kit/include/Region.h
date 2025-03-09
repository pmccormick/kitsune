/**
 * @file Region.h
 * @brief Defines the Region class for grouping cells in a mesh
 * 
 * This file implements a flexible region system that can represent
 * arbitrary collections of cells using different storage strategies
 * based on region characteristics (sparse or dense).
 * 
 * IMPLEMENTATION NOTES:
 * - Regions use a hybrid approach with multiple storage modes for efficiency
 * - Storage modes can be switched dynamically based on region density
 * - Special care is needed when transitioning between storage modes
 * - The Region class delegates membership criteria to a RegionDefinition
 */

#ifndef REGION_H
#define REGION_H

#include "CellBase.h"
#include <climits>
#include <string>
#include <unordered_set>
#include <vector>
#include <memory>
#include <functional>

// Forward declarations
class MeshBase;
template <typename T> class Mesh;

/**
 * @brief Region identifier type
 */
using RegionID = uint32_t;

/**
 * @brief Abstract base class for region definitions
 * 
 * RegionDefinition provides a polymorphic interface for defining
 * the membership criteria of a region (geometric, predicates, etc.)
 */
class RegionDefinition {
public:
    virtual ~RegionDefinition() = default;
    
    /**
     * @brief Check if a cell belongs to this region
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell is in the region
     */
    virtual bool contains(const CellBase* cell) const = 0;
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief Get a bounding box for the region (if applicable)
     * 
     * @return Pair of min/max indices that bound the region
     */
    virtual std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const {
      // Create pairs explicitly with maximum extents for safety
      std::pair<int, int> minPair(0, 0);
      std::pair<int, int> maxPair(INT_MAX, INT_MAX);
      return std::make_pair(minPair, maxPair);
    }
};

/**
 * @brief Predicate-based region definition
 * 
 * Defines a region using a function that determines membership
 */
class PredicateRegion : public RegionDefinition {
public:
    /**
     * @brief Construct a new Predicate Region
     * 
     * @param name Region name
     * @param predicate Function that determines if a cell is in the region
     */
    PredicateRegion(std::string name, std::function<bool(const CellBase*)> predicate)
        : m_name(std::move(name)), m_predicate(std::move(predicate)) {}
    
    /**
     * @brief Check if a cell belongs to this region using the predicate
     * 
     * @param cell Pointer to the cell to check
     * @return true if the predicate returns true for this cell
     */
    bool contains(const CellBase* cell) const override {
        return m_predicate(cell);
    }
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override {
        return m_name;
    }
    
private:
    std::string m_name;                             ///< Name of the region
    std::function<bool(const CellBase*)> m_predicate; ///< Function determining membership
};

/**
 * @brief Rectangular region definition
 * 
 * Defines a region as a rectangular area in the mesh
 */
class RectangularRegion : public RegionDefinition {
public:
    /**
     * @brief Construct a new Rectangular Region
     * 
     * @param name Region name
     * @param minI Minimum i-index (inclusive)
     * @param minJ Minimum j-index (inclusive)
     * @param maxI Maximum i-index (inclusive)
     * @param maxJ Maximum j-index (inclusive)
     */
    RectangularRegion(std::string name, int minI, int minJ, int maxI, int maxJ)
        : m_name(std::move(name)), m_minI(minI), m_minJ(minJ), m_maxI(maxI), m_maxJ(maxJ) {}
    
    /**
     * @brief Check if a cell belongs to this region by testing if it's within bounds
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell is within the defined rectangle
     */
    bool contains(const CellBase* cell) const override {
        int i = cell->i();
        int j = cell->j();
        return i >= m_minI && i <= m_maxI && j >= m_minJ && j <= m_maxJ;
    }
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override {
        return m_name;
    }
    
    /**
     * @brief Get the bounding box for the rectangular region
     * 
     * @return Pair of min/max indices that exactly define the rectangle
     */
    std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const override {
        return {{m_minI, m_minJ}, {m_maxI, m_maxJ}};
    }
    
private:
    std::string m_name;  ///< Name of the region
    int m_minI, m_minJ;  ///< Lower bounds (inclusive)
    int m_maxI, m_maxJ;  ///< Upper bounds (inclusive)
};

/**
 * @brief Composite region definition using set operations
 * 
 * Combines two region definitions using set operations (union, intersection, difference)
 */
class CompositeRegion : public RegionDefinition {
public:
    /**
     * @brief Set operation types
     */
    enum class Operation {
        UNION,        ///< A ∪ B (cells in either region)
        INTERSECTION, ///< A ∩ B (cells in both regions)
        DIFFERENCE    ///< A - B (cells in A but not in B)
    };
    
    /**
     * @brief Construct a new Composite Region
     * 
     * @param name Region name
     * @param regionA First region definition
     * @param regionB Second region definition
     * @param operation Set operation to apply
     */
    CompositeRegion(std::string name, 
                   std::shared_ptr<RegionDefinition> regionA,
                   std::shared_ptr<RegionDefinition> regionB,
                   Operation operation)
        : m_name(std::move(name)), 
          m_regionA(std::move(regionA)),
          m_regionB(std::move(regionB)),
          m_operation(operation) {}
    
    /**
     * @brief Check if a cell belongs to this region using the composite operation
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell satisfies the composite criteria
     */
    bool contains(const CellBase* cell) const override {
        bool inA = m_regionA->contains(cell);
        
        switch (m_operation) {
            case Operation::UNION:
                return inA || m_regionB->contains(cell);
            
            case Operation::INTERSECTION:
                return inA && m_regionB->contains(cell);
            
            case Operation::DIFFERENCE:
                return inA && !m_regionB->contains(cell);
                
            default:
                return false;
        }
    }
    
    /**
     * @brief Get the name of the region definition
     * 
     * @return Region definition name
     */
    std::string getName() const override {
        return m_name;
    }
    
    /**
     * @brief Get the bounding box for the composite region
     * 
     * For UNION: combines the bounds of both regions
     * For INTERSECTION/DIFFERENCE: uses the bounds of the first region
     * 
     * @return Pair of min/max indices that bound the region
     */
    std::pair<std::pair<int, int>, std::pair<int, int>> getBounds() const override {
        auto boundsA = m_regionA->getBounds();
        auto boundsB = m_regionB->getBounds();
        
        // For union, we take the combined bounds
        if (m_operation == Operation::UNION) {
            return {
                {std::min(boundsA.first.first, boundsB.first.first),
                 std::min(boundsA.first.second, boundsB.first.second)},
                {std::max(boundsA.second.first, boundsB.second.first),
                 std::max(boundsA.second.second, boundsB.second.second)}
            };
        }
        
        // For intersection and difference, we can use A's bounds
        return boundsA;
    }
    
private:
    std::string m_name;                           ///< Name of the region
    std::shared_ptr<RegionDefinition> m_regionA;  ///< First region definition
    std::shared_ptr<RegionDefinition> m_regionB;  ///< Second region definition
    Operation m_operation;                        ///< Set operation to apply
};

/**
 * @brief Region class representing a collection of cells
 * 
 * The Region class implements a hybrid approach for storing cell collections,
 * automatically selecting between cell sets and bit vectors based on region density.
 */
class Region {
public:
    /**
     * @brief Storage mode for the region
     */
    enum class StorageMode {
        CELL_SET,   ///< Store explicit cell indices (for sparse regions)
        BIT_VECTOR, ///< Store a bit per cell (for dense regions)
        DYNAMIC     ///< Recalculate on-the-fly (for frequently changing regions)
    };
    
    /**
     * @brief Construct a new Region with a definition
     * 
     * @param id Region identifier
     * @param definition Region definition object
     * @param meshSize Total number of cells in the mesh
     */
    Region(RegionID id, std::shared_ptr<RegionDefinition> definition, size_t meshSize = 0)
        : m_id(id), 
          m_definition(std::move(definition)),
          m_mode(StorageMode::CELL_SET),
          m_meshSize(meshSize) {}
    
    /**
     * @brief Virtual destructor
     */
    virtual ~Region() = default;
    
    /**
     * @brief Get the region identifier
     * 
     * @return Region ID
     */
    RegionID getId() const { return m_id; }
    
    /**
     * @brief Get the region name
     * 
     * @return Region name
     */
    std::string getName() const { return m_definition->getName(); }
    
    /**
     * @brief Get the region definition
     * 
     * @return Pointer to the region definition
     */
    std::shared_ptr<const RegionDefinition> getDefinition() const { return m_definition; }
    
    /**
     * @brief Get the current storage mode
     * 
     * @return Storage mode
     */
    StorageMode getStorageMode() const { return m_mode; }
    
    /**
     * @brief Set the storage mode explicitly
     * 
     * IMPORTANT: This will ensure data is properly converted between formats.
     * When changing modes, the internal storage structures will be updated to
     * maintain consistency.
     * 
     * @param mode Storage mode to use
     */
    void setStorageMode(StorageMode mode);
    
    /**
     * @brief Check if a cell is in the region
     * 
     * For DYNAMIC mode: Evaluates using the definition
     * For CELL_SET mode: Checks if the cell's linear index is in the set
     * For BIT_VECTOR mode: Checks the bit at the cell's linear index
     * 
     * @param cell Pointer to the cell
     * @return true if the cell is in the region
     */
    bool contains(const CellBase* cell) const;
    
    /**
     * @brief Check if a cell index is in the region
     * 
     * For CELL_SET mode: Checks if the index is in the set
     * For BIT_VECTOR mode: Checks the bit at the specified index
     * For DYNAMIC mode: Always returns false (can't check by index alone)
     * 
     * @param linearIndex Linear index of the cell
     * @return true if the cell is in the region
     */
    bool containsIndex(int linearIndex) const;
    
    /**
     * @brief Build or rebuild the region storage from a mesh
     * 
     * This method evaluates the region definition against all cells in the mesh
     * (or within the bounds, if available) and populates the internal storage.
     * 
     * @tparam CellType Type of cell in the mesh
     * @param mesh Mesh to build the region from
     */
    template <typename CellType>
    void buildFromMesh(const Mesh<CellType>& mesh);
    
    /**
     * @brief Add a cell to the region
     * 
     * @param cell Pointer to the cell to add
     */
    void addCell(const CellBase* cell);
    
    /**
     * @brief Add a cell index to the region
     * 
     * For CELL_SET mode: Adds the index to the set
     * For BIT_VECTOR mode: Sets the bit at the specified index
     * For DYNAMIC mode: Switches to CELL_SET mode and adds the index
     * 
     * @param linearIndex Linear index of the cell to add
     */
    void addCellIndex(int linearIndex);
    
    /**
     * @brief Remove a cell from the region
     * 
     * @param cell Pointer to the cell to remove
     */
    void removeCell(const CellBase* cell);
    
    /**
     * @brief Remove a cell index from the region
     * 
     * For CELL_SET mode: Removes the index from the set
     * For BIT_VECTOR mode: Clears the bit at the specified index
     * For DYNAMIC mode: Switches to appropriate mode and removes the index
     * 
     * @param linearIndex Linear index of the cell to remove
     */
    void removeCellIndex(int linearIndex);
    
    /**
     * @brief Clear the region (remove all cells)
     * 
     * Clears all internal storage structures but keeps the region definition.
     */
    void clear();
    
    /**
     * @brief Get the number of cells in the region
     * 
     * @return Cell count
     */
    size_t size() const;
    
    /**
     * @brief Check if the region is empty
     * 
     * @return true if the region contains no cells
     */
    bool isEmpty() const { return size() == 0; }
    
    /**
     * @brief Get the cell indices in the region
     * 
     * If using BIT_VECTOR mode, this will create a set of cell indices 
     * from the bit vector. This operation can be expensive for large 
     * dense regions.
     * 
     * @return Set of linear indices for cells in the region
     */
    const std::unordered_set<int>& getCellIndices();
    
    /**
     * @brief Get the bit vector for the region
     * 
     * If using CELL_SET mode, this will create a bit vector from the
     * cell indices set. This ensures the bit vector representation is available
     * when needed.
     * 
     * @return Reference to bit vector (one bit per cell)
     */
    const std::vector<bool>& getBitVector();
    
    /**
     * @brief Create a region representing the union of this region and another
     * 
     * @param other Other region to union with
     * @param name Name for the new region
     * @param newId ID for the new region
     * @return New region representing the union
     */
    Region createUnion(const Region& other, const std::string& name, RegionID newId) const;
    
    /**
     * @brief Create a region representing the intersection of this region and another
     * 
     * @param other Other region to intersect with
     * @param name Name for the new region
     * @param newId ID for the new region
     * @return New region representing the intersection
     */
    Region createIntersection(const Region& other, const std::string& name, RegionID newId) const;
    
    /**
     * @brief Create a region representing the difference of this region and another
     * 
     * @param other Other region to subtract
     * @param name Name for the new region
     * @param newId ID for the new region
     * @return New region representing the difference
     */
    Region createDifference(const Region& other, const std::string& name, RegionID newId) const;
    
    /**
     * @brief Optimize the storage mode based on region density
     * 
     * This method automatically selects the most efficient storage mode
     * based on the density of the region (ratio of cells in region to total cells).
     * 
     * @param threshold Threshold ratio (0-1) for switching between storage modes
     */
    void optimizeStorage(double threshold = 0.1);
    
protected:
    RegionID m_id;                            ///< Region identifier
    std::shared_ptr<RegionDefinition> m_definition; ///< Region definition
    StorageMode m_mode;                      ///< Current storage mode
    size_t m_meshSize;                       ///< Total number of cells in the mesh
    
    // Storage options
    std::unordered_set<int> m_cellIndices;    ///< Cell indices (for sparse regions)
    std::unique_ptr<std::vector<bool>> m_bitVector; ///< Bit vector (for dense regions)
    
    /**
     * @brief Ensure bit vector storage is initialized
     * 
     * If using CELL_SET mode, this creates a bit vector and populates it
     * from the cell indices. The bit vector is guaranteed to be valid
     * after this call, regardless of the previous state.
     */
    void ensureBitVector();
    
    /**
     * @brief Ensure cell set storage is initialized
     * 
     * If using BIT_VECTOR mode, this creates a cell index set and populates it
     * from the bit vector. The cell index set is guaranteed to be valid
     * after this call, regardless of the previous state.
     */
    void ensureCellSet();
};

// Template Implementation
template <typename CellType>
void Region::buildFromMesh(const Mesh<CellType>& mesh) {
    // Clear existing data
    clear();
    
    // Store mesh size for bounds checking
    m_meshSize = mesh.nx() * mesh.ny();
    
    // If using dynamic mode, just store the definition and return
    if (m_mode == StorageMode::DYNAMIC) {
        return;
    }
    
    // Get bounds from definition to optimize the scan
    auto bounds = m_definition->getBounds();
    
    // Limit bounds to actual mesh size
    int minI = std::max(bounds.first.first, 0);
    int minJ = std::max(bounds.first.second, 0);
    int maxI = std::min(bounds.second.first, mesh.nx() - 1);
    int maxJ = std::min(bounds.second.second, mesh.ny() - 1);
    
    // Only create bit vector if we're using BIT_VECTOR mode
    if (m_mode == StorageMode::BIT_VECTOR) {
        m_bitVector = std::make_unique<std::vector<bool>>(m_meshSize, false);
    }
    
    // Loop through cells in the bounds and check against definition
    for (int j = minJ; j <= maxJ; ++j) {
        for (int i = minI; i <= maxI; ++i) {
            // Important: Use getTypedCell to get the correct cell type
            const CellType* cell = mesh.getTypedCell(i, j);
            
            // Check if the cell exists and is in the region
            if (cell && m_definition->contains(cell)) {
                // Track the cell according to the current storage mode
                if (m_mode == StorageMode::CELL_SET) {
                    m_cellIndices.insert(cell->linearIndex());
                } else if (m_mode == StorageMode::BIT_VECTOR && m_bitVector) {
                    // Make sure the bit vector is initialized
                    (*m_bitVector)[cell->linearIndex()] = true;
                }
            }
        }
    }
    
    // Optimize storage if needed
    optimizeStorage();
}

#endif // REGION_H
