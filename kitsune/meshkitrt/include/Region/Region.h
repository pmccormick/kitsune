/**
 * @file Region.h
 * @brief High-performance, thread-safe region representation for mesh-based simulations.
 * 
 * This file defines the Region class, which represents a collection of cells within a mesh.
 * The Region class employs an adaptive storage strategy that automatically switches between
 * different internal representations based on the region characteristics and operations.
 * 
 * Key features:
 * - Adaptive storage system that optimizes for both memory usage and performance
 * - Support for dense and sparse regions through different storage modes
 * - Automatic optimization that selects the best storage mode based on region density
 * - Thread-safe implementation for concurrent access and modification
 * - Flexible region definition through various geometric and predicate-based approaches
 * - Support for set operations (union, intersection, difference) between regions
 * 
 * The class's design separates the region definition (shape/membership criteria) from
 * the storage representation, allowing for flexible and efficient region manipulation.
 */

#ifndef REGION_H
#define REGION_H

#include "Mesh.h"
#include "Cell.h"
#include "BitArray.h"
#include "RegionDefinition.h"
#include <unordered_set>
#include <memory>
#include <functional>
#include <string>
#include <atomic>
#include <mutex>

namespace mesh {

  // Type alias for region identifiers
  using RegionID = uint32_t;

  /**
   * @brief Enumeration of the different storage modes for regions.
   * 
   * Each mode represents a different internal representation strategy with
   * different performance characteristics and memory usage patterns:
   * 
   * - DYNAMIC: No explicit storage; membership evaluated on demand through the definition
   * - CELL_SET: Explicit storage of member cell indices in an unordered_set (sparse regions)
   * - BIT_ARRAY: Compressed bit-based representation (dense regions)
   */
  enum class RegionStorageMode {
    DYNAMIC,    ///< On-demand evaluation using the region definition
    CELL_SET,   ///< Explicit storage as a set of cell indices (for sparse regions)
    BIT_ARRAY   ///< Compressed bit array representation (for dense regions)
  };

  /**
   * @brief High-performance, adaptive representation of a collection of cells within a mesh.
   * 
   * The Region class represents a subset of cells in a mesh using an adaptive storage strategy
   * that automatically switches between different internal representations based on region
   * characteristics. This approach optimizes both memory usage and access performance.
   * 
   * Thread Safety:
   * This implementation is thread-safe for concurrent access and modification through the
   * use of mutexes and atomic operations. Methods are documented with their specific
   * thread-safety guarantees.
   */
  class Region {
  public:
    /**
     * @brief Construct a new Region object.
     * 
     * Initializes a region with the specified ID, definition, and mesh binding.
     * Based on the region definition and mesh characteristics, the constructor
     * may automatically materialize the region storage or keep it in DYNAMIC mode.
     * 
     * Thread Safety: Thread-safe for different regions; not thread-safe for same region.
     * 
     * @param id Unique identifier for the region
     * @param definition Shared pointer to the region definition that determines membership
     * @param mesh Pointer to the mesh containing the cells
     * @throws std::invalid_argument If mesh is nullptr
     */
    Region(RegionID id, std::shared_ptr<RegionDefinition> definition, Mesh* mesh);

    /**
     * @brief Copy constructor.
     * 
     * Creates a new region that is a copy of another. For efficiency, only the active
     * storage mode's data structure is deeply copied; the other is left empty.
     * 
     * Thread Safety: Thread-safe with respect to other operations on the source region.
     * 
     * @param other Region to copy
     */
    Region(const Region& other) noexcept;

    /**
     * @brief Move constructor.
     * 
     * Creates a new region by taking ownership of another region's resources.
     * The moved-from region is left in a valid but unspecified state.
     * 
     * Thread Safety: Not thread-safe. The source region must not be accessed concurrently.
     * 
     * @param other Region to move from
     */
    Region(Region&& other) noexcept;

    /**
     * @brief Copy assignment operator.
     * 
     * Replaces the contents with a copy of another region. For efficiency, only the
     * active storage mode's data structure is deeply copied; the other is left empty.
     * 
     * Thread Safety: Thread-safe with respect to other operations on the source region.
     *                Not thread-safe for the destination region.
     * 
     * @param other Region to copy
     * @return Reference to this region
     */
    Region& operator=(const Region& other) noexcept;

    /**
     * @brief Move assignment operator.
     * 
     * Replaces the contents by taking ownership of another region's resources.
     * The moved-from region is left in a valid but unspecified state.
     * 
     * Thread Safety: Not thread-safe. Neither region must be accessed concurrently.
     * 
     * @param other Region to move from
     * @return Reference to this region
     */
    Region& operator=(Region&& other) noexcept;

    /**
     * @brief Check if the region contains a specific cell.
     * 
     * Determines whether the given cell is a member of this region based on the
     * current storage mode. In DYNAMIC mode, it evaluates the region definition;
     * in explicit storage modes, it checks the corresponding data structure.
     * 
     * Thread Safety: Thread-safe for concurrent reads. May cause storage materialization.
     * 
     * @param cell Pointer to the cell to check
     * @return true if the cell is in the region, false otherwise
     */
    bool contains(const Cell* cell) const;

    /**
     * @brief Check if the region contains a cell with the given linear index.
     * 
     * Similar to contains(Cell*) but accepts a linearized cell index instead of
     * a cell pointer. This can be more efficient when the caller already has the
     * linear index.
     * 
     * Thread Safety: Thread-safe for concurrent reads. May cause storage materialization.
     * 
     * @param linearIndex Linear index of the cell to check
     * @return true if the cell with this index is in the region, false otherwise
     */
    bool containsIndex(int linearIndex) const;

    /**
     * @brief Add a cell to the region.
     * 
     * Adds the specified cell to the region by adding its linear index to the
     * internal storage. If in DYNAMIC mode, this will trigger a transition to
     * an explicit storage mode.
     * 
     * Thread Safety: Thread-safe for concurrent modifications.
     * 
     * @param cell Pointer to the cell to add
     * @throws std::invalid_argument If cell is nullptr or not associated with a mesh
     */
    void addCell(const Cell* cell);

    /**
     * @brief Add a cell to the region by its linear index.
     * 
     * Adds a cell to the region using its linear index. This is more efficient than
     * addCell() when the caller already has the linear index.
     * 
     * Thread Safety: Thread-safe for concurrent modifications.
     * 
     * @param linearIndex Linear index of the cell to add
     * @throws std::out_of_range If linearIndex is negative or beyond mesh bounds
     */
    void addCellIndex(int linearIndex);

    /**
     * @brief Remove a cell from the region.
     * 
     * Removes the specified cell from the region. If in DYNAMIC mode, this will
     * trigger a transition to an explicit storage mode.
     * 
     * Thread Safety: Thread-safe for concurrent modifications.
     * 
     * @param cell Pointer to the cell to remove
     * @throws std::invalid_argument If cell is nullptr or not associated with a mesh
     */
    void removeCell(const Cell* cell);

    /**
     * @brief Remove a cell from the region by its linear index.
     * 
     * Removes a cell from the region using its linear index. This is more efficient than
     * removeCell() when the caller already has the linear index.
     * 
     * Thread Safety: Thread-safe for concurrent modifications.
     * 
     * @param linearIndex Linear index of the cell to remove
     * @throws std::out_of_range If linearIndex is negative or beyond mesh bounds
     */
    void removeCellIndex(int linearIndex);

    /**
     * @brief Remove all cells from the region.
     * 
     * Clears all cell memberships while maintaining the region definition and mesh binding.
     * 
     * Thread Safety: Thread-safe for concurrent modifications.
     */
    void clear();

    /**
     * @brief Get the number of cells in the region.
     * 
     * Calculates and returns the current number of cells in the region.
     * For DYNAMIC mode, this will evaluate the definition for all cells.
     * 
     * Thread Safety: Thread-safe for concurrent reads. May cause storage materialization.
     * 
     * @return The number of cells in the region
     */
    size_t size() const;

    /**
     * @brief Get the unique identifier of the region.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return RegionID The region's unique identifier
     */
    RegionID getId() const { return m_id; }

    /**
     * @brief Get the region definition.
     * 
     * Returns the shared pointer to the region definition that determines cell membership.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return Shared pointer to the region definition
     */
    std::shared_ptr<RegionDefinition> getDefinition() const { return m_definition; }

    /**
     * @brief Get the current storage mode of the region.
     * 
     * Returns the current storage mode (DYNAMIC, CELL_SET, or BIT_ARRAY).
     * 
     * Thread Safety: Thread-safe for reads, but the mode may change due to concurrent operations.
     * 
     * @return Current storage mode
     */
    RegionStorageMode getStorageMode() const { 
      std::lock_guard<std::mutex> lock(m_modificationMutex);
      return m_mode; 
    }

    /**
     * @brief Explicitly set the storage mode of the region.
     * 
     * Changes the internal representation to the specified mode, performing necessary
     * data conversions. Note that this bypasses the automatic optimization system.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @param mode The desired storage mode
     * @throws std::invalid_argument If mode is DYNAMIC (cannot convert to DYNAMIC)
     */
    void setStorageMode(RegionStorageMode mode);

    /**
     * @brief Get the mesh associated with this region.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return Pointer to the associated mesh
     */
    Mesh* mesh() const { return m_mesh; }

    // Add this method to the Region class (in Region.h and implement in Region.cpp)
    Cell* getCellPtr(uint32_t i, uint32_t j) const {
      if (!m_mesh || i >= m_mesh->nx() || j >= m_mesh->ny()) {
	return nullptr;
      }
  
      // Create a persistent Cell object and return its address
      // Note: This assumes the Cell will live long enough for the operation
      static thread_local Cell cellStorage;
      cellStorage = m_mesh->getCell(i, j);
      return &cellStorage;
    }

    /**
     * @brief Get the total number of cells in the associated mesh.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return Total number of cells in the mesh
     */
    size_t getMeshSize() const { return m_meshSize; }

    /**
     * @brief Get a read-only reference to the cell indices storage.
     * 
     * Returns a const reference to the set of cell indices. If the current storage
     * mode is not CELL_SET, it will convert to that representation first.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @return Const reference to the set of cell indices
     */
    const std::unordered_set<int>& getCellIndices() const;

    /**
     * @brief Get a writable reference to the cell indices storage.
     * 
     * Returns a non-const reference to the set of cell indices, switching to
     * CELL_SET mode if necessary. This allows direct modification of the storage.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @return Reference to the set of cell indices
     * @throws std::logic_error If the conversion to CELL_SET mode fails
     */
    std::unordered_set<int>& getCellIndicesForWrite();

    /**
     * @brief Get a read-only reference to the bit array storage.
     * 
     * Returns a const reference to the bit array. If the current storage
     * mode is not BIT_ARRAY, it will convert to that representation first.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @return Const reference to the bit array
     */
    const BitArray& getBitArray() const;

    /**
     * @brief Get a writable reference to the bit array storage.
     * 
     * Returns a non-const reference to the bit array, switching to
     * BIT_ARRAY mode if necessary. This allows direct modification of the storage.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @return Reference to the bit array
     * @throws std::logic_error If the conversion to BIT_ARRAY mode fails
     */
    BitArray& getBitArrayForWrite();

    /**
     * @brief Replace the bit array storage with a new one.
     * 
     * Sets the region's bit array to the provided one and switches to BIT_ARRAY mode.
     * This is more efficient than copying bit-by-bit when you already have a BitArray.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @param bitArray BitArray to use (moved, not copied)
     * @throws std::invalid_argument If the bit array size doesn't match the mesh size
     */
    void setBitArray(BitArray bitArray);

    /**
     * @brief Create a copy of this region with a specific storage mode.
     * 
     * Returns a new region with the same membership but using the specified storage mode.
     * 
     * Thread Safety: Thread-safe for concurrent reads.
     * 
     * @param mode Desired storage mode for the new region
     * @return A new Region object with the specified storage mode
     * @throws std::invalid_argument If mode is DYNAMIC (cannot convert to DYNAMIC)
     */
    Region toStorageMode(RegionStorageMode mode) const;

    /**
     * @brief Optimize the storage representation based on region characteristics.
     * 
     * Analyzes the region density and current storage mode to determine if a different
     * mode would be more efficient. If a better mode is found, converts to that mode.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @param threshold Density threshold for mode selection (default is 0.1, or 10%)
     */
    void optimizeStorage(double threshold = 0.1);

    /**
     * @brief Force immediate optimization regardless of operation count.
     * 
     * Temporarily enables optimization (if disabled) and immediately performs storage
     * optimization, ignoring the normal operation count threshold.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     */
    void forceOptimization();

    /**
     * @brief Check if automatic optimization is enabled.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return true if automatic optimization is enabled, false otherwise
     */
    bool isOptimizationEnabled() const { return m_optimizationEnabled; }

    /**
     * @brief Enable or disable automatic optimization.
     * 
     * When disabled, the region will not automatically optimize its storage mode
     * after operations, which can be useful for batch operations.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @param enable Whether to enable automatic optimization
     */
    void setOptimizationEnabled(bool enable) { 
      std::lock_guard<std::mutex> lock(m_modificationMutex);
      m_optimizationEnabled = enable; 
    }

    /**
     * @brief Set a custom optimization strategy.
     * 
     * Replaces the default optimization decision functions with custom ones,
     * allowing tailored optimization behavior for different workloads.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @param shouldOptimize Function to decide when optimization should occur
     * @param selectMode Function to select the optimal storage mode
     * @throws std::invalid_argument If either function pointer is null
     */
    void setOptimizationStrategy(
        std::function<bool(const Region&, size_t)> shouldOptimize,
        std::function<RegionStorageMode(const Region&, RegionStorageMode, double)> selectMode);

    /**
     * @brief Set the default general-purpose optimization strategy.
     * 
     * Restores the default optimization behavior, which balances memory usage
     * and performance for general-purpose workloads.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     */
    void setDefaultOptimizationStrategy();

    /**
     * @brief Set an optimization strategy optimized for simulation workloads.
     * 
     * Sets an optimization strategy that prioritizes stability during simulations
     * by reducing the frequency of mode transitions and favoring BIT_ARRAY for
     * efficient set operations.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     */
    void setSimulationOptimizationStrategy();

    /**
     * @brief Rebind the region to a different mesh.
     * 
     * Updates the mesh pointer and adjusts the internal storage as needed.
     * If the new mesh has different dimensions, the storage will be rebuilt.
     * 
     * Thread Safety: Thread-safe for concurrent operations.
     * 
     * @param newMesh Pointer to the new mesh
     * @throws std::invalid_argument If newMesh is nullptr
     */
    void rebindToMesh(Mesh* newMesh);

  private:
    // Core region data
    RegionID m_id;                                    ///< Unique identifier for the region
    std::shared_ptr<RegionDefinition> m_definition;   ///< Function defining region membership
    RegionStorageMode m_mode;                         ///< Current storage mode
    Mesh* m_mesh;                                     ///< Pointer to the associated mesh
    size_t m_meshSize;                                ///< Total number of cells in the mesh

    // Storage structures - only one is actively used based on m_mode
    mutable std::unordered_set<int> m_cellIndices;    ///< Cell indices for CELL_SET mode
    mutable BitArray m_bitArray;                      ///< Bit flags for BIT_ARRAY mode

    // Thread safety
    mutable std::mutex m_modificationMutex;           ///< Mutex for protecting state changes
    mutable std::atomic<size_t> m_operationsSinceOptimization;  ///< Operation counter for optimization
    mutable std::atomic<double> m_lastOptimizationRatio;        ///< Last calculated region density
    std::atomic<bool> m_optimizationEnabled;                    ///< Whether auto-optimization is enabled

    // Optimization strategy function objects
    std::function<bool(const Region&, size_t)> m_shouldOptimizeFunc;          ///< When to optimize
    std::function<RegionStorageMode(const Region&, RegionStorageMode, double)> m_selectModeFunc;  ///< How to select mode

    /**
     * @brief Track an operation for optimization purposes.
     * 
     * Increments the operation counter and potentially triggers optimization
     * if the threshold is reached. This is called after modifications.
     * 
     * Thread Safety: Thread-safe.
     */
    void trackOperation() const;

    /**
     * @brief Consider whether to optimize the storage based on recent operations.
     * 
     * Evaluates whether optimization should occur based on the operation count
     * and optimization strategy. If needed, calls optimizeStorage().
     * 
     * Thread Safety: Thread-safe.
     */
    void considerOptimization() const;

    /**
     * @brief Ensure the bit array is properly initialized and populated.
     * 
     * If the bit array doesn't match the mesh size, recreates it and populates
     * it from either cell indices or by evaluating the definition.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @throws std::logic_error If mesh size is unknown (m_meshSize is 0)
     */
    void ensureBitArray() const;

    /**
     * @brief Ensure the cell set is properly initialized and populated.
     * 
     * If in BIT_ARRAY mode, populates the cell set from the bit array.
     * If in DYNAMIC mode, evaluates the definition for all cells.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @throws std::logic_error If the region is in an inconsistent state
     */
    void ensureCellSet() const;

    /**
     * @brief Materialize storage from DYNAMIC mode to an explicit mode.
     * 
     * Converts from DYNAMIC mode to either CELL_SET or BIT_ARRAY mode
     * by evaluating the region definition for all relevant cells.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @throws std::logic_error If not in DYNAMIC mode or mesh is nullptr
     */
    void materializeStorage();

    /**
     * @brief Determine if the region should be materialized immediately.
     * 
     * Based on region characteristics (e.g., rectangular regions, small meshes),
     * decides whether to immediately materialize storage during construction.
     * 
     * Thread Safety: Thread-safe.
     * 
     * @return true if immediate materialization is recommended, false otherwise
     */
    bool shouldMaterializeImmediately() const;

    /**
     * @brief Reset the region to an empty state after move operations.
     * 
     * Resets all internal state to create a consistent empty region.
     * Used in move constructor and move assignment to leave the
     * moved-from object in a valid but unspecified state.
     * 
     * Thread Safety: Not thread-safe. The region must not be accessed concurrently.
     */
    void resetToEmptyState();
  };

  /**
   * @brief Create a region representing the union of two regions.
   * 
   * Combines two regions, producing a new region that contains all cells that
   * are in either input region. The implementation automatically selects the
   * most efficient storage mode for the result based on the input regions.
   * 
   * Thread Safety: Thread-safe with respect to the input regions.
   * 
   * @param regionA First input region
   * @param regionB Second input region
   * @param name Optional name for the resulting region
   * @return A new region representing the union
   * @throws std::invalid_argument If the input regions use different meshes
   */
  Region createUnionRegion(const Region& regionA, const Region& regionB, const std::string& name = "Union");

  /**
   * @brief Create a region representing the intersection of two regions.
   * 
   * Combines two regions, producing a new region that contains only cells that
   * are in both input regions. The implementation automatically selects the
   * most efficient storage mode for the result based on the input regions.
   * 
   * Thread Safety: Thread-safe with respect to the input regions.
   * 
   * @param regionA First input region
   * @param regionB Second input region
   * @param name Optional name for the resulting region
   * @return A new region representing the intersection
   * @throws std::invalid_argument If the input regions use different meshes
   */
  Region createIntersectionRegion(const Region& regionA, const Region& regionB, const std::string& name = "Intersection");

  /**
   * @brief Create a region representing the set difference of two regions (A - B).
   * 
   * Produces a new region that contains cells that are in the first region but not
   * in the second region. The implementation automatically selects the most efficient
   * storage mode for the result based on the input regions.
   * 
   * Thread Safety: Thread-safe with respect to the input regions.
   * 
   * @param regionA First input region (minuend)
   * @param regionB Second input region (subtrahend)
   * @param name Optional name for the resulting region
   * @return A new region representing the difference
   * @throws std::invalid_argument If the input regions use different meshes
   */
  Region createDifferenceRegion(const Region& regionA, const Region& regionB, const std::string& name = "Difference");

  /**
   * @brief Create a region by filtering cells in a mesh using a predicate.
   * 
   * Constructs a region containing only cells that satisfy the given predicate function.
   * 
   * Thread Safety: Thread-safe with respect to the mesh.
   * 
   * @param mesh Mesh to filter
   * @param predicate Function that determines if a cell should be included
   * @param name Optional name for the resulting region
   * @return A new region containing the filtered cells
   */
  Region filterMesh(
      const Mesh& mesh,
      std::function<bool(const Cell*)> predicate,
      const std::string& name = "FilteredRegion");

  /**
   * @brief Check if a cell is in a region.
   * 
   * Utility function to determine if a cell belongs to a region.
   * This is equivalent to region.contains(cell) but more explicit.
   * 
   * Thread Safety: Thread-safe for concurrent reads of the region.
   * 
   * @param cell Cell to check
   * @param region Region to check against
   * @return true if the cell is in the region, false otherwise
   */
  bool isCellInRegion(const Cell* cell, const Region& region);

} // namespace mesh

#endif // REGION_H
