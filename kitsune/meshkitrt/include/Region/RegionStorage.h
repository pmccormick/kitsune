/**
 * @file RegionStorage.h
 * @brief Defines storage-related functionality for the Region class
 * 
 * This file contains declarations related to the storage mechanisms used by
 * the Region class, including storage mode enums, optimization function types,
 * and operations to convert between different storage representations.
 */

#ifndef REGION_STORAGE_H
#define REGION_STORAGE_H

#include "BitArray.h"
#include <functional>
#include <unordered_set>

namespace mesh {

// Forward declarations
class Region;
class RegionDefinition;

/**
 * @brief Type alias for optimization decision function
 * 
 * Function that determines whether a region should be optimized based on
 * operation count and current state.
 */
using ShouldOptimizeFunc = std::function<bool(const Region&, size_t)>;

/**
 * @brief Type alias for mode selection function
 * 
 * Function that selects the optimal storage mode based on region characteristics.
 */
using SelectModeFunc = std::function<RegionStorageMode(const Region&, RegionStorageMode, double)>;

/**
 * @brief Convert a bit array to a set of cell indices
 * 
 * Utility function that extracts the indices of set bits in a bit array.
 * 
 * @param bitArray Source bit array
 * @return Set of indices corresponding to set bits
 */
std::unordered_set<int> bitArrayToCellIndices(const BitArray& bitArray);

/**
 * @brief Convert a set of cell indices to a bit array
 * 
 * Utility function that creates a bit array with bits set for each index.
 * 
 * @param cellIndices Source cell indices
 * @param meshSize Total size of the mesh (for bit array sizing)
 * @return Bit array representation
 */
BitArray cellIndicesToBitArray(const std::unordered_set<int>& cellIndices, size_t meshSize);

/**
 * @brief Create default optimization strategy functions
 * 
 * Creates a pair of functions implementing the default optimization strategy:
 * - Optimize after 100+ operations if near the threshold
 * - Optimize after 1000+ operations unconditionally
 * - Use a threshold of ~10% with hysteresis
 * 
 * @return Pair of optimization strategy functions (should optimize, select mode)
 */
std::pair<ShouldOptimizeFunc, SelectModeFunc> createDefaultOptimizationStrategy();

/**
 * @brief Create simulation-optimized strategy functions
 * 
 * Creates a pair of functions implementing a strategy tuned for simulation workloads:
 * - Optimize less frequently
 * - Prefer BIT_ARRAY mode for improved set operation performance
 * - Use a lower threshold with strong hysteresis
 * 
 * @return Pair of optimization strategy functions (should optimize, select mode)
 */
std::pair<ShouldOptimizeFunc, SelectModeFunc> createSimulationOptimizationStrategy();

} // namespace 
#endif // REGION_STORAGE_H

