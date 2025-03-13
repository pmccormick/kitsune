/**
 * @file RegionUtils.h
 * @brief Utility functions for the Region system
 * 
 * This file provides utility functions used throughout the Region system,
 * including ID generation, conversion helpers, and other common operations.
 */

#ifndef REGION_UTILS_H
#define REGION_UTILS_H

#include "Region.h"

#include <limits>
#include <cstddef> 
#include <cstdint>
#include <string>


namespace mesh {

// Forward declaration for RegionID
using RegionID = uint32_t;

/**
 * @brief Generate a robust composite ID for regions created from set operations
 * 
 * This function creates a unique ID for composite regions by combining the IDs of
 * the component regions with the operation type. It uses the FNV-1a hash algorithm
 * to minimize collisions while ensuring the result fits within the RegionID range.
 *
 * This approach is more robust than simple arithmetic operations (addition, 
 * multiplication, subtraction) which are prone to collisions and overflow issues.
 * 
 * @param idA ID of the first region
 * @param idB ID of the second region
 * @param operationCode Code indicating the operation type (1=union, 2=intersection, 3=difference)
 * @return A unique Region ID for the composite region
 */
RegionID generateCompositeID(RegionID idA, RegionID idB, size_t operationCode);

/**
 * @brief Create a region from set operation on two existing regions
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param operation Operation code (1=union, 2=intersection, 3=difference)
 * @param name Optional name for the region
 * @return New region resulting from the set operation
 */
class Region;
Region createRegionFromSetOperation(const Region& regionA, const Region& regionB, 
                                    size_t operation, const std::string& name);

/**
 * @brief Create a union of two regions
 * 
 * Creates a new region representing all cells that are in either input region.
 * The resulting region will have a storage mode optimized for the characteristics
 * of the union operation.
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the union
 */
Region createUnionRegion(const Region& regionA, const Region& regionB, 
                         const std::string& name);

/**
 * @brief Create an intersection of two regions
 * 
 * Creates a new region representing cells that are in both input regions.
 * The resulting region will have a storage mode optimized for the characteristics
 * of the intersection operation.
 * 
 * @param regionA First region
 * @param regionB Second region
 * @param name Optional name for the region
 * @return Region representing the intersection
 */
Region createIntersectionRegion(const Region& regionA, const Region& regionB, 
                                const std::string& name);

/**
 * @brief Create a difference of two regions (A - B)
 * 
 * Creates a new region representing cells that are in the first region but not
 * in the second region. The resulting region will have a storage mode optimized 
 * for the characteristics of the difference operation.
 * 
 * @param regionA First region (minuend)
 * @param regionB Second region (subtrahend)
 * @param name Optional name for the region
 * @return Region representing the difference
 */
Region createDifferenceRegion(const Region& regionA, const Region& regionB, 
                              const std::string& name);

/**
 * @brief Apply a function to each cell in a region
 * 
 * @tparam MeshT Type of mesh
 * @tparam FuncT Type of function to apply
 * @param mesh Mesh containing the cells
 * @param region Region to iterate over
 * @param func Function to apply to each cell
 */
template <typename MeshT, typename FuncT>
void forEachCellInRegion(MeshT& mesh, const Region& region, FuncT func);

/**
 * @brief Get all cells in a region
 * 
 * @tparam MeshT Type of mesh
 * @param mesh Mesh containing the cells
 * @param region Region to get cells from
 * @return Vector of pointers to cells in the region
 */
template <typename MeshT>
std::vector<typename MeshT::cell_type*> getCellsInRegion(MeshT& mesh, const Region& region);

/**
 * @brief Check if a cell is in a region
 * 
 * @param cell Cell to check
 * @param region Region to check
 * @return true if the cell is in the region
 */
bool isCellInRegion(const Cell* cell, const Region& region);

// Template method implementations will be in RegionUtils.hpp

} // namespace 
#endif // REGION_UTILS_H

