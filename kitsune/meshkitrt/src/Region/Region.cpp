/**
 * @file Region.cpp
 * @brief Implementation of the Region class
 *
 * This file contains the implementation of the Region class, including
 * constructors, assignment operators, and core functionality. The Region
 * class provides an adaptive representation of cell collections in a mesh,
 * automatically optimizing storage based on region characteristics.
 */

#include "Region.h"
#include "RegionUtils.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <functional>
#include <cmath>

namespace mesh {

  // Constructor implementation
  Region::Region(RegionID id, std::shared_ptr<RegionDefinition> definition, Mesh* mesh)
    : m_id(id),
      m_definition(std::move(definition)),
      m_mode(RegionStorageMode::DYNAMIC),
      m_mesh(mesh),
      m_operationsSinceOptimization(0),
      m_lastOptimizationRatio(0.0),
      m_optimizationEnabled(true),
      m_bitArray(0, false)  // Initialize with empty BitArray
  {
    // Validate mesh - a null mesh would make cell operations impossible
    if (!mesh) {
      throw std::invalid_argument("Cannot create Region: mesh cannot be null");
    }

    // Store mesh size for bounds checking and density calculations
    // This is cached to avoid repeated calculations
    m_meshSize = mesh->size();

    // Set default optimization strategy functions
    // These define when optimization occurs and which storage mode is selected
    setDefaultOptimizationStrategy();

    // Check if we should materialize immediately based on region characteristics
    // This is an optimization for certain types of regions where immediate
    // materialization is more efficient than starting in DYNAMIC mode
    if (shouldMaterializeImmediately()) {
      // Thread safety: Since this is the constructor, we don't need
      // mutex protection here as the object isn't accessible yet
      materializeStorage();
    }
  }

  // Copy constructor implementation - optimized to only copy active storage structures
  Region::Region(const Region& other) noexcept
    : m_id(other.m_id),
      m_definition(other.m_definition), 
      m_mode(other.m_mode),
      m_mesh(other.m_mesh),
      m_meshSize(other.m_meshSize),
      m_operationsSinceOptimization(other.m_operationsSinceOptimization.load()),
      m_lastOptimizationRatio(other.m_lastOptimizationRatio.load()),
      m_optimizationEnabled(other.m_optimizationEnabled.load()),
      m_shouldOptimizeFunc(other.m_shouldOptimizeFunc),
      m_selectModeFunc(other.m_selectModeFunc),
      // Initialize BitArray correctly based on mode
      m_bitArray(other.m_mode == RegionStorageMode::BIT_ARRAY ? 
                other.m_bitArray : BitArray(0, false))
  {
    // Lock the source region to ensure a consistent copy
    std::lock_guard<std::mutex> otherLock(other.m_modificationMutex);
    
    // Only copy the active storage structure based on mode to reduce memory usage
    // This is a key optimization - we avoid copying unused representations
    if (m_mode == RegionStorageMode::CELL_SET) {
      // Deep copy of cell indices set
      m_cellIndices = other.m_cellIndices;
      // BitArray is already initialized in the initializer list
    }
    else if (m_mode == RegionStorageMode::BIT_ARRAY) {
      // BitArray is already copied in the initializer list
      // Don't copy cell indices to save memory
      m_cellIndices.clear();
    }
    else {
      // For DYNAMIC mode, we don't need to copy either storage structure
      // since membership is determined by the definition
      m_cellIndices.clear();
      // BitArray is already initialized in the initializer list
    }
  }

  // Move constructor implementation with complete state transfer
  Region::Region(Region&& other) noexcept
    : m_id(other.m_id),
      m_definition(std::move(other.m_definition)),
      m_mode(other.m_mode),
      m_meshSize(other.m_meshSize),
      m_mesh(other.m_mesh),
      m_cellIndices(std::move(other.m_cellIndices)),
      m_bitArray(std::move(other.m_bitArray)),
      m_operationsSinceOptimization(other.m_operationsSinceOptimization.load()),
      m_lastOptimizationRatio(other.m_lastOptimizationRatio.load()),
      m_optimizationEnabled(other.m_optimizationEnabled.load()),
      m_shouldOptimizeFunc(std::move(other.m_shouldOptimizeFunc)),
      m_selectModeFunc(std::move(other.m_selectModeFunc))
  {
    // Lock the source region to ensure a consistent move
    // This is necessary even for move operations to prevent concurrent
    // modifications during the move
    std::lock_guard<std::mutex> otherLock(other.m_modificationMutex);
    
    // Reset the moved-from object to a consistent empty state
    // This ensures the moved-from object is in a valid but unspecified state
    // after the move, as required by C++ move semantics
    other.resetToEmptyState();
  }

  // Copy assignment operator implementation - optimized to only copy active storage
  Region& Region::operator=(const Region& other) noexcept {
    if (this != &other) {
      // Lock both regions to prevent concurrent modifications
      // Use std::lock to prevent potential deadlocks when locking multiple mutexes
      std::lock(m_modificationMutex, other.m_modificationMutex);
      
      // Use lock_guard with adopt_lock to manage the already-acquired locks
      std::lock_guard<std::mutex> thisLock(m_modificationMutex, std::adopt_lock);
      std::lock_guard<std::mutex> otherLock(other.m_modificationMutex, std::adopt_lock);
      
      // Copy basic properties
      m_id = other.m_id;
      m_definition = other.m_definition;
      m_mode = other.m_mode;
      m_meshSize = other.m_meshSize;
      m_mesh = other.m_mesh;

      // Only copy the active storage structure based on mode
      // This optimization reduces memory usage by avoiding unnecessary copies
      if (m_mode == RegionStorageMode::CELL_SET) {
        m_cellIndices = other.m_cellIndices;
        // Clear bit array to save memory
        m_bitArray = BitArray(0, false);
      }
      else if (m_mode == RegionStorageMode::BIT_ARRAY) {
        m_bitArray = other.m_bitArray;
        // Clear cell indices to save memory
        m_cellIndices.clear();
      }
      else {
        // For DYNAMIC mode, don't need either storage structure
        m_cellIndices.clear();
        m_bitArray = BitArray(0, false);
      }

      // Copy optimization data
      m_operationsSinceOptimization.store(other.m_operationsSinceOptimization.load());
      m_lastOptimizationRatio.store(other.m_lastOptimizationRatio.load());
      m_optimizationEnabled.store(other.m_optimizationEnabled.load());
      m_shouldOptimizeFunc = other.m_shouldOptimizeFunc;
      m_selectModeFunc = other.m_selectModeFunc;
    }
    return *this;
  }

  // Move assignment operator implementation with complete state transfer
  Region& Region::operator=(Region&& other) noexcept {
    if (this != &other) {
      // Lock both regions to prevent concurrent modifications
      std::lock(m_modificationMutex, other.m_modificationMutex);
      
      // Use lock_guard with adopt_lock to manage the already-acquired locks
      std::lock_guard<std::mutex> thisLock(m_modificationMutex, std::adopt_lock);
      std::lock_guard<std::mutex> lockOther(other.m_modificationMutex, std::adopt_lock);
      
      // Move basic properties
      m_id = other.m_id;
      m_definition = std::move(other.m_definition);
      m_mode = other.m_mode;
      m_meshSize = other.m_meshSize;
      m_mesh = other.m_mesh;
      
      // Move storage structures (only one will be active based on mode)
      m_cellIndices = std::move(other.m_cellIndices);
      m_bitArray = std::move(other.m_bitArray);

      // Move optimization data
      m_operationsSinceOptimization.store(other.m_operationsSinceOptimization.load());
      m_lastOptimizationRatio.store(other.m_lastOptimizationRatio.load());
      m_optimizationEnabled.store(other.m_optimizationEnabled.load());
      m_shouldOptimizeFunc = std::move(other.m_shouldOptimizeFunc);
      m_selectModeFunc = std::move(other.m_selectModeFunc);

      // Reset the moved-from object to a consistent empty state
      other.resetToEmptyState();
    }
    return *this;
  }

  // Implementation of resetToEmptyState for consistent moved-from object state
  void Region::resetToEmptyState() {
    // Reset basic properties to a consistent empty state
    // This is important for moved-from objects to maintain valid state
    m_mode = RegionStorageMode::DYNAMIC;
    m_meshSize = 0;
    m_mesh = nullptr;

    // Clear storage structures to free memory
    m_cellIndices.clear();
    m_bitArray = BitArray(0, false);

    // Reset optimization tracking
    m_operationsSinceOptimization.store(0);
    m_lastOptimizationRatio.store(0.0);

    // Clear strategy functions (note: this breaks the object until a new strategy is set)
    // The caller must ensure a new strategy is set if the object will be reused
    m_shouldOptimizeFunc = nullptr;
    m_selectModeFunc = nullptr;
  }

  // Cell membership test implementation - determines if a cell belongs to the region
  bool Region::contains(const Cell* cell) const {
    // Thread safety: This method only performs reads, but may trigger materialization
    
    // Safety check - null cells are never in a region
    // This validation is important to prevent null pointer dereferences
    if (!cell) {
      return false;
    }

    // Check if cell has a valid mesh
    // We can't determine membership for cells without mesh context
    if (!cell->mesh()) {
      return false;
    }

    // DYNAMIC mode always evaluates using the definition
    // This avoids the overhead of storage materialization when possible
    if (m_mode == RegionStorageMode::DYNAMIC) {
      return m_definition->contains(cell);
    }

    // Get cell's linear index for storage-based lookup
    int linearIndex;
    try {
      linearIndex = cell->linearIndex();
    } catch (const std::exception& e) {
      // If linearIndex throws, the cell is invalid
      return false;
    }

    // For other modes, check if the cell's index is in the region
    // Delegate to the index-based implementation for consistent behavior
    return containsIndex(linearIndex);
  }

  // Index-based membership test implementation - core lookup function
  bool Region::containsIndex(int linearIndex) const {
    // Thread safety: This method only performs reads, but may trigger materialization
    
    // Check index is in valid range
    // This is important for safety and prevents out-of-bounds access
    if (linearIndex < 0 || (m_meshSize > 0 && static_cast<size_t>(linearIndex) >= m_meshSize)) {
      return false;
    }

    // Check based on storage mode
    // Each mode has its own efficient lookup mechanism
    switch (m_mode) {
    case RegionStorageMode::CELL_SET:
      // Thread safety: unordered_set is not thread-safe for concurrent modifications,
      // but concurrent reads are safe. This method doesn't modify the set.
      // Check if index is in the set - O(1) average time complexity
      return m_cellIndices.find(linearIndex) != m_cellIndices.end();

    case RegionStorageMode::BIT_ARRAY:
      // Thread safety: BitArray's get() method is thread-safe
      // First verify bit array is correctly sized to prevent inconsistent state
      {
        // Lock to check and potentially materialize the bit array
        std::lock_guard<std::mutex> lock(m_modificationMutex);
        if (m_bitArray.size() != m_meshSize) {
          ensureBitArray();
        }
      }
      // Check if the corresponding bit is set - O(1) time complexity
      return m_bitArray.get(linearIndex);

    case RegionStorageMode::DYNAMIC:
      {
        // For dynamic mode, convert index to cell and check definition
        // This requires the mesh to map linear index back to coordinates
        if (!m_mesh) {
          throw std::logic_error("Cannot check index: no mesh bound to region");
        }

        // Convert linear index to i,j coordinates
        auto [i, j] = m_mesh->toIndices(linearIndex);
        Cell* cell = getCellPtr(i, j);

        if (!cell) {
          return false;
        }
        // Evaluate the definition for the cell
        return m_definition->contains(cell);
      }

    default:
      return false;
    }
  }

  // Implementation to add a cell to the region
  void Region::addCell(const Cell* cell) {
    // Thread safety: Protected with mutex since this modifies internal storage
    
    // Validate the cell
    if (!cell) {
      throw std::invalid_argument("Cannot add null cell to region");
    }
    
    if (!cell->mesh() || cell->mesh() != m_mesh) {
      throw std::invalid_argument("Cell must belong to the same mesh as the region");
    }
    
    // Get linear index
    int linearIndex = cell->linearIndex();
    
    // Delegate to index-based implementation
    addCellIndex(linearIndex);
  }

  // Implementation to add a cell index to the region
  void Region::addCellIndex(int linearIndex) {
    // Thread safety: Protected with mutex since this modifies internal storage
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Validate index
    if (linearIndex < 0 || static_cast<size_t>(linearIndex) >= m_meshSize) {
      throw std::out_of_range("Cell index out of range");
    }
    
    // If in DYNAMIC mode, we need to materialize first
    if (m_mode == RegionStorageMode::DYNAMIC) {
      materializeStorage();
    }
    
    // Now add the cell index based on current storage mode
    if (m_mode == RegionStorageMode::CELL_SET) {
      m_cellIndices.insert(linearIndex);
    }
    else if (m_mode == RegionStorageMode::BIT_ARRAY) {
      // Ensure bit array is properly initialized
      if (m_bitArray.size() != m_meshSize) {
        ensureBitArray();
      }
      
      m_bitArray.set(linearIndex, true);
    }
    
    // Track the operation for potential optimization
    trackOperation();
  }

  // Implementation to remove a cell from the region
  void Region::removeCell(const Cell* cell) {
    // Thread safety: Protected with mutex since this modifies internal storage
    
    // Validate the cell
    if (!cell) {
      throw std::invalid_argument("Cannot remove null cell from region");
    }
    
    if (!cell->mesh() || cell->mesh() != m_mesh) {
      throw std::invalid_argument("Cell must belong to the same mesh as the region");
    }
    
    // Get linear index
    int linearIndex = cell->linearIndex();
    
    // Delegate to index-based implementation
    removeCellIndex(linearIndex);
  }

  // Implementation to remove a cell index from the region
  void Region::removeCellIndex(int linearIndex) {
    // Thread safety: Protected with mutex since this modifies internal storage
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Validate index
    if (linearIndex < 0 || static_cast<size_t>(linearIndex) >= m_meshSize) {
      throw std::out_of_range("Cell index out of range");
    }
    
    // If in DYNAMIC mode, we need to materialize first
    if (m_mode == RegionStorageMode::DYNAMIC) {
      materializeStorage();
    }
    
    // Now remove the cell index based on current storage mode
    if (m_mode == RegionStorageMode::CELL_SET) {
      m_cellIndices.erase(linearIndex);
    }
    else if (m_mode == RegionStorageMode::BIT_ARRAY) {
      // Ensure bit array is properly initialized
      if (m_bitArray.size() != m_meshSize) {
        ensureBitArray();
      }
      
      m_bitArray.set(linearIndex, false);
    }
    
    // Track the operation for potential optimization
    trackOperation();
  }

  // Implementation to remove all cells from the region
  void Region::clear() {
    // Thread safety: Protected with mutex since it modifies internal storage
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Clear all data structures but keep the definition and mesh
    // This efficiently resets the region to an empty state
    m_cellIndices.clear();

    // Reset the bit array if it's initialized
    if (m_bitArray.size() > 0) {
      m_bitArray.clear();
    }

    // Reset optimization tracking
    m_operationsSinceOptimization.store(0);
    m_lastOptimizationRatio.store(0.0);
  }

  // Count the number of cells in the region
  size_t Region::size() const {
    // Thread safety: Protected with mutex since it may modify internal state
    // during materialization, though logically this is a const operation
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Calculate size based on storage mode
    // Each mode has its own efficient way to determine the count
    switch (m_mode) {
    case RegionStorageMode::CELL_SET:
      // Size is the number of indices in the set - O(1) time complexity
      return m_cellIndices.size();

    case RegionStorageMode::BIT_ARRAY:
      // Size is the number of set bits in the bit array
      // This takes advantage of BitArray's cached count feature for efficiency
      if (m_bitArray.size() != m_meshSize) {
        ensureBitArray();
      }
      return m_bitArray.count();

    case RegionStorageMode::DYNAMIC: {
      // For dynamic mode, count cells that match the definition
      // This is potentially expensive as we must check every cell
      if (!m_mesh) {
        throw std::logic_error("Cannot determine size: no mesh bound to region");
      }

      size_t count = 0;
      for (uint32_t j = 0; j < m_mesh->ny(); ++j) {
        for (uint32_t i = 0; i < m_mesh->nx(); ++i) {
          Cell* cell = getCellPtr(i, j);
          // Count cells that match the definition
          if (cell && m_definition->contains(cell)) {
            ++count;
          }
        }
      }
      return count;
    }

    default:
      return 0;
    }
  }

  // Explicitly set the storage mode
  void Region::setStorageMode(RegionStorageMode mode) {
    // Thread safety: Protected with mutex since this may modify internal storage
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // No change needed if already in this mode
    if (m_mode == mode) {
      return;
    }
    
    // Can't convert to DYNAMIC mode - it's a starting state only
    if (mode == RegionStorageMode::DYNAMIC) {
      throw std::invalid_argument("Cannot convert to DYNAMIC storage mode");
    }
    
    // If current mode is DYNAMIC, materialize first
    if (m_mode == RegionStorageMode::DYNAMIC) {
      materializeStorage();
    }
    
    // Now convert between explicit storage modes if needed
    if (m_mode != mode) {
      if (mode == RegionStorageMode::CELL_SET) {
        // Convert from BIT_ARRAY to CELL_SET
        ensureCellSet();
        // Clear bit array to save memory
        m_bitArray = BitArray(0, false);
      }
      else if (mode == RegionStorageMode::BIT_ARRAY) {
        // Convert from CELL_SET to BIT_ARRAY
        ensureBitArray();
        // Clear cell indices to save memory
        m_cellIndices.clear();
      }
    }
    
    // Update the mode
    m_mode = mode;
  }

  // Get a read-only reference to the cell indices
  const std::unordered_set<int>& Region::getCellIndices() const {
    // Thread safety: Protected with mutex since this may modify internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // If not in CELL_SET mode, ensure the cell set is populated
    if (m_mode != RegionStorageMode::CELL_SET) {
      ensureCellSet();
    }
    
    return m_cellIndices;
  }

  // Get a writable reference to the cell indices
  std::unordered_set<int>& Region::getCellIndicesForWrite() {
    // Thread safety: Protected with mutex since this modifies internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // If not in CELL_SET mode, switch to it
    if (m_mode != RegionStorageMode::CELL_SET) {
      ensureCellSet();
      m_mode = RegionStorageMode::CELL_SET;
    }
    
    return m_cellIndices;
  }

  // Get a read-only reference to the bit array
  const BitArray& Region::getBitArray() const {
    // Thread safety: Protected with mutex since this may modify internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // If not in BIT_ARRAY mode, ensure the bit array is populated
    if (m_mode != RegionStorageMode::BIT_ARRAY) {
      ensureBitArray();
    }
    
    return m_bitArray;
  }

  // Get a writable reference to the bit array
  BitArray& Region::getBitArrayForWrite() {
    // Thread safety: Protected with mutex since this modifies internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // If not in BIT_ARRAY mode, switch to it
    if (m_mode != RegionStorageMode::BIT_ARRAY) {
      ensureBitArray();
      m_mode = RegionStorageMode::BIT_ARRAY;
    }
    
    return m_bitArray;
  }

  // Replace the bit array with a new one
  void Region::setBitArray(BitArray bitArray) {
    // Thread safety: Protected with mutex since this modifies internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Verify size matches
    if (bitArray.size() != m_meshSize) {
      throw std::invalid_argument("BitArray size does not match mesh size");
    }
    
    // Replace the bit array and update mode
    m_bitArray = std::move(bitArray);
    m_mode = RegionStorageMode::BIT_ARRAY;
    
    // Clear cell indices to save memory
    m_cellIndices.clear();
  }

  // Create a copy with specific storage mode
  Region Region::toStorageMode(RegionStorageMode mode) const {
    // Thread safety: This is a const method that returns a new object
    
    // Can't convert to DYNAMIC mode
    if (mode == RegionStorageMode::DYNAMIC) {
      throw std::invalid_argument("Cannot convert to DYNAMIC storage mode");
    }
    
    // Create a copy
    Region result(*this);
    
    // Set the desired storage mode
    result.setStorageMode(mode);
    
    return result;
  }

  // Optimize storage based on region characteristics
  void Region::optimizeStorage(double threshold) {
    // Thread safety: Protected with mutex since this modifies internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Skip if already in DYNAMIC mode
    if (m_mode == RegionStorageMode::DYNAMIC) {
      return;
    }
    
    // Calculate density (ratio of member cells to total cells)
    double density = static_cast<double>(size()) / m_meshSize;
    
    // Store for future optimization decisions
    m_lastOptimizationRatio.store(density, std::memory_order_relaxed);
    
    // Use the strategy function to select optimal mode
    RegionStorageMode optimalMode = m_selectModeFunc(*this, m_mode, threshold);
    
    // Switch to the optimal mode if different from current
    if (optimalMode != m_mode) {
      setStorageMode(optimalMode);
    }
    
    // Reset operation counter
    m_operationsSinceOptimization.store(0, std::memory_order_relaxed);
  }

  // Force immediate optimization
  void Region::forceOptimization() {
    // Thread safety: Protected with mutex through optimizeStorage()
    
    // Save current optimization state
    bool wasEnabled = m_optimizationEnabled.load();
    
    // Temporarily enable optimization if disabled
    if (!wasEnabled) {
      m_optimizationEnabled.store(true);
    }
    
    // Perform optimization
    optimizeStorage();
    
    // Restore previous state
    if (!wasEnabled) {
      m_optimizationEnabled.store(false);
    }
  }

  // Set custom optimization strategy
  void Region::setOptimizationStrategy(
      std::function<bool(const Region&, size_t)> shouldOptimize,
      std::function<RegionStorageMode(const Region&, RegionStorageMode, double)> selectMode) {
    // Thread safety: Protected with mutex since this modifies strategy functions
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Validate functions
    if (!shouldOptimize || !selectMode) {
      throw std::invalid_argument("Optimization strategy functions cannot be null");
    }
    
    // Set strategy functions
    m_shouldOptimizeFunc = std::move(shouldOptimize);
    m_selectModeFunc = std::move(selectMode);
  }

  // Set default general-purpose optimization strategy
  void Region::setDefaultOptimizationStrategy() {
    // Thread safety: Protected with mutex through setOptimizationStrategy()
    
    // Create default strategy functions
    auto shouldOptimize = [](const Region& region, size_t operationCount) -> bool {
      // Don't optimize too frequently
      if (operationCount < 100) return false;
      
      // Always optimize after a significant number of operations
      if (operationCount >= 1000) return true;
      
      // Calculate density
      double density = 0.0;
      try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
      } catch (const std::exception&) {
        // If we can't calculate density, don't optimize
        return false;
      }
      
      // Optimize more frequently when near the decision boundary
      bool nearThreshold = std::abs(density - 0.1) < 0.02;
      
      return (nearThreshold && operationCount >= 500);
    };
    
    auto selectMode = [](const Region& region, RegionStorageMode currentMode, double threshold) -> RegionStorageMode {
      // Calculate density
      double density = 0.0;
      try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
      } catch (const std::exception&) {
        // If we can't calculate density, stay in current mode
        return currentMode;
      }
      
      // Use hysteresis to prevent oscillation
      if (currentMode == RegionStorageMode::BIT_ARRAY) {
        // Stay in BIT_ARRAY mode unless density is significantly below threshold
        return (density > threshold * 0.7) ? 
          RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
      } else {
        // Stay in CELL_SET mode unless density is significantly above threshold
        return (density > threshold * 1.3) ? 
          RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
      }
    };
    
    // Set the strategy
    setOptimizationStrategy(shouldOptimize, selectMode);
  }

  // Set optimization strategy tuned for simulation workloads
  void Region::setSimulationOptimizationStrategy() {
    // Thread safety: Protected with mutex through setOptimizationStrategy()
    
    // Create simulation-optimized strategy functions
    auto shouldOptimize = [](const Region& region, size_t operationCount) -> bool {
      // For simulations, optimize less frequently
      if (operationCount < 5000) return false;
      
      // For very large regions, be even more conservative
      if (region.getMeshSize() > 1000000) {
        return operationCount > 10000;
      }
      
      return true;
    };
    
    auto selectMode = [](const Region& region, RegionStorageMode currentMode, double threshold) -> RegionStorageMode {
      // Calculate density
      double density = 0.0;
      try {
        density = static_cast<double>(region.size()) / region.getMeshSize();
      } catch (const std::exception&) {
        // If we can't calculate density, stay in current mode
        return currentMode;
      }
      
      // For simulations, prefer BIT_ARRAY for improved set operation performance
      // Use a lower threshold (0.05 instead of 0.1)
      double simThreshold = threshold > 0 ? threshold : 0.05; // Default to 0.05 if no threshold specified
      
      // With stronger hysteresis to avoid mode switching during simulation
      if (currentMode == RegionStorageMode::BIT_ARRAY) {
        // Stay in BIT_ARRAY mode unless density is very low
        return (density > simThreshold * 0.4) ? 
          RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
      } else {
        // Only switch to BIT_ARRAY if density is definitively high
        return (density > simThreshold * 2.0) ? 
          RegionStorageMode::BIT_ARRAY : RegionStorageMode::CELL_SET;
      }
    };
    
    // Set the strategy
    setOptimizationStrategy(shouldOptimize, selectMode);
  }

  // Rebind the region to a different mesh
  void Region::rebindToMesh(Mesh* newMesh) {
    // Thread safety: Protected with mutex since this modifies internal state
    std::lock_guard<std::mutex> lock(m_modificationMutex);
    
    // Validate new mesh
    if (!newMesh) {
      throw std::invalid_argument("Cannot rebind: mesh cannot be null");
    }
    
    // Calculate new mesh size
    size_t newMeshSize = newMesh->size();
    
    // If mesh size changed, we need to rebuild storage
    if (newMeshSize != m_meshSize) {
      // For BIT_ARRAY mode, resize the bit array
      if (m_mode == RegionStorageMode::BIT_ARRAY) {
        m_bitArray.resize(newMeshSize, false);
      }
      
      // For CELL_SET mode, we need to filter out invalid indices
      // This is more complex as it requires remapping indices
      // For simplicity, we'll reset to DYNAMIC mode
      if (m_mode == RegionStorageMode::CELL_SET) {
        m_cellIndices.clear();
        m_mode = RegionStorageMode::DYNAMIC;
      }
    }
    
    // Update mesh pointer and size
    m_mesh = newMesh;
    m_meshSize = newMeshSize;
    
    // Reset optimization tracking
    m_operationsSinceOptimization.store(0);
    m_lastOptimizationRatio.store(0.0);
  }

  // Track modifications for optimization purposes
  void Region::trackOperation() const {
    // Thread safety: Uses atomic operations for thread-safe counter updates
    if (m_optimizationEnabled.load(std::memory_order_relaxed)) {
      // Increment operation counter
      size_t currentCount = m_operationsSinceOptimization.fetch_add(1, std::memory_order_relaxed) + 1;
      
      // Check if we've reached the optimization threshold
      considerOptimization();
    }
  }
  
  // Evaluate whether optimization should occur
  void Region::considerOptimization() const {
    // Thread safety: Uses double-checked locking for efficiency and correctness
    
    // Quick check outside the lock for performance
    if (!m_optimizationEnabled.load(std::memory_order_relaxed)) return;

    // Use strategy to determine if optimization is needed
    // This avoids locking unless optimization is actually needed
    bool shouldOptimize = false;
    size_t opCount = m_operationsSinceOptimization.load(std::memory_order_relaxed);

    try {
      shouldOptimize = m_shouldOptimizeFunc(*this, opCount);
    } catch (const std::exception& e) {
      // If the decision function throws, don't optimize and log warning
      std::cerr << "Warning: Failed to evaluate optimization strategy: "
                << e.what() << std::endl;
      return;
    }

    if (shouldOptimize) {
      // Double-checked locking pattern: check again after acquiring lock
      // This avoids race conditions while minimizing lock contention
      std::lock_guard<std::mutex> lock(m_modificationMutex);
      opCount = m_operationsSinceOptimization.load(std::memory_order_relaxed);
      
      if (m_shouldOptimizeFunc(*this, opCount)) {
        try {
          // Const cast needed because we need to modify internal state
          // This is safe because the method is called from const methods
          // that are allowed to modify mutable members
          const_cast<Region*>(this)->optimizeStorage();

          // Reset counter and update ratio
          m_operationsSinceOptimization.store(0, std::memory_order_relaxed);
          m_lastOptimizationRatio.store(static_cast<double>(size()) / m_meshSize, 
                                       std::memory_order_relaxed);
        } catch (const std::exception& e) {
          // If optimization fails, log warning but continue
          std::cerr << "Warning: Failed to optimize region storage: "
                    << e.what() << std::endl;
        }
      }
    }
  }

  // Ensure the bit array is properly initialized
  void Region::ensureBitArray() const {
    // Thread safety: This should be called while holding m_modificationMutex
    
    // If the bit array is already correctly sized, nothing to do
    if (m_bitArray.size() == m_meshSize) {
      return;
    }
    
    // Create a new bit array of the correct size
    BitArray newBitArray(m_meshSize, false);
    
    // If we're in CELL_SET mode, populate the bit array from cell indices
    if (m_mode == RegionStorageMode::CELL_SET) {
      for (int idx : m_cellIndices) {
        if (idx >= 0 && static_cast<size_t>(idx) < m_meshSize) {
          newBitArray.set(idx, true);
        }
      }
    }
    // If we're in DYNAMIC mode, evaluate the definition for all cells
    else if (m_mode == RegionStorageMode::DYNAMIC) {
      if (!m_mesh) {
        throw std::logic_error("Cannot build bit array: no mesh bound to region");
      }
      
      for (uint32_t j = 0; j < m_mesh->ny(); ++j) {
        for (uint32_t i = 0; i < m_mesh->nx(); ++i) {
          Cell* cell = getCellPtr(i, j);
          if (cell && m_definition->contains(cell)) {
            int idx = m_mesh->linearIndex(i, j);
            newBitArray.set(idx, true);
          }
        }
      }
    }
    
    // Replace the bit array
    m_bitArray = std::move(newBitArray);
  }
  
  // Ensure the cell set is properly initialized
  void Region::ensureCellSet() const {
    // Thread safety: This should be called while holding m_modificationMutex
    
    // If we're in BIT_ARRAY mode, populate cell set from bit array
    if (m_mode == RegionStorageMode::BIT_ARRAY) {
      // Clear existing cell indices
      m_cellIndices.clear();
      
      // Reserve space for efficiency
      size_t setBits = m_bitArray.count();
      m_cellIndices.reserve(setBits);
      
      // Add indices for each set bit
      for (size_t idx = m_bitArray.findFirst(); 
           idx < m_bitArray.size(); 
           idx = m_bitArray.findNext(idx)) {
        m_cellIndices.insert(static_cast<int>(idx));
      }
    }
    // If we're in DYNAMIC mode, evaluate the definition for all cells
    else if (m_mode == RegionStorageMode::DYNAMIC) {
      if (!m_mesh) {
        throw std::logic_error("Cannot build cell set: no mesh bound to region");
      }
      
      // Clear existing cell indices
      m_cellIndices.clear();
      
      // Add indices for cells that match the definition
      for (uint32_t j = 0; j < m_mesh->ny(); ++j) {
        for (uint32_t i = 0; i < m_mesh->nx(); ++i) {
          Cell* cell = getCellPtr(i, j);
          if (cell && m_definition->contains(cell)) {
            int idx = m_mesh->linearIndex(i, j);
            m_cellIndices.insert(idx);
          }
        }
      }
    }
  }
  
  // Materialize storage from DYNAMIC mode to explicit storage
  void Region::materializeStorage() {
    // Thread safety: This should be called while holding m_modificationMutex
    
    // Only applicable in DYNAMIC mode
    if (m_mode != RegionStorageMode::DYNAMIC) {
      return;
    }
    
    // We need a valid mesh to materialize
    if (!m_mesh) {
      throw std::logic_error("Cannot materialize storage: no mesh bound to region");
    }
    
    // Use the definition bounds to limit the cells we check
    auto bounds = m_definition->getBounds();
    
    // Clamp bounds to mesh dimensions
    int minI = std::max(bounds.first.first, 0);
    int minJ = std::max(bounds.first.second, 0);
    int maxI = std::min(bounds.second.first, static_cast<int>(m_mesh->nx()) - 1);
    int maxJ = std::min(bounds.second.second, static_cast<int>(m_mesh->ny()) - 1);
    
    // Count cells that match the definition
    size_t count = 0;
    for (int j = minJ; j <= maxJ; ++j) {
      for (int i = minI; i <= maxI; ++i) {
        Cell* cell = getCellPtr(i, j);
        if (cell && m_definition->contains(cell)) {
          ++count;
        }
      }
    }
    
    // Calculate density to determine optimal storage mode
    double density = static_cast<double>(count) / m_meshSize;
    
    // Choose storage mode based on density
    if (density > 0.1) {
      // Dense region - use bit array
      // Initialize bit array and select storage mode
      m_bitArray = BitArray(m_meshSize, false);
      m_mode = RegionStorageMode::BIT_ARRAY;
      
      // Populate bit array
      for (int j = minJ; j <= maxJ; ++j) {
        for (int i = minI; i <= maxI; ++i) {
          Cell* cell = getCellPtr(i, j);
          if (cell && m_definition->contains(cell)) {
            int idx = m_mesh->linearIndex(i, j);
            m_bitArray.set(idx, true);
          }
        }
      }
      
      // Clear cell indices to save memory
      m_cellIndices.clear();
    }
    else {
      // Sparse region - use cell set
      // Initialize cell set and select storage mode
      m_cellIndices.clear();
      m_cellIndices.reserve(count);
      m_mode = RegionStorageMode::CELL_SET;
      
      // Populate cell set
      for (int j = minJ; j <= maxJ; ++j) {
        for (int i = minI; i <= maxI; ++i) {
          Cell* cell = getCellPtr(i, j);
          if (cell && m_definition->contains(cell)) {
            int idx = m_mesh->linearIndex(i, j);
            m_cellIndices.insert(idx);
          }
        }
      }
      
      // Initialize bit array to empty to save memory
      m_bitArray = BitArray(0, false);
    }
  }
  
  // Determine if a region should be materialized immediately upon construction
  bool Region::shouldMaterializeImmediately() const {
    // Use heuristics to decide if immediate materialization is beneficial
    
    // For rectangular regions, materialize immediately (efficient to evaluate)
    // Rectangle bounds are trivially computed, making materialization cheap
    if (dynamic_cast<const RectangularRegion*>(m_definition.get())) {
      return true;
    }

    // For small meshes, always materialize
    // The overhead of dynamic evaluation isn't worth it for small meshes
    if (m_meshSize < 10000) {
      return true;
    }

    // Otherwise, stay in DYNAMIC mode until explicitly materialized
    // This defers the potentially expensive materialization until needed
    return false;
  }

} // namespace mesh

