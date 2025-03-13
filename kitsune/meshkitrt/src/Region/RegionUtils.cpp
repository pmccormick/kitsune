/**
 * @file RegionUtils.cpp
 * @brief Implementation of utility functions for the Region system
 *
 * This file implements the utility functions declared in RegionUtils.h,
 * providing common operations used throughout the Region system.
 */

#include "RegionUtils.h"
#include "Region.h"
#include "RegionDefinition.h"
#include <limits>

namespace mesh {

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
  RegionID generateCompositeID(RegionID idA, RegionID idB, size_t operationCode) {
    // Combine IDs with a hash function to minimize collisions
    std::size_t h1 = static_cast<std::size_t>(idA);
    std::size_t h2 = static_cast<std::size_t>(idB);

    // FNV-1a hash variant to combine the IDs and operation
    constexpr std::size_t FNV_prime = 1099511628211ULL;
    constexpr std::size_t FNV_offset = 14695981039346656037ULL;

    std::size_t hash = FNV_offset;

    // Mix in first ID
    hash ^= h1;
    hash *= FNV_prime;

    // Mix in second ID
    hash ^= h2;
    hash *= FNV_prime;

    // Mix in operation code to differentiate union/intersection/difference
    hash ^= operationCode;
    hash *= FNV_prime;

    // Ensure we stay within RegionID range
    // (assuming RegionID is uint32_t as defined in the header)
    return static_cast<RegionID>(hash % std::numeric_limits<RegionID>::max());
  }

  /**
   * @brief Check if a cell is in a region
   *
   * @param cell Cell to check
   * @param region Region to check
   * @return true if the cell is in the region
   */
  bool isCellInRegion(const Cell* cell, const Region& region) {
    return region.contains(cell);
  }

  /**
   * @brief Create a region containing cells that match a predicate
   *
   * @tparam MeshT Type of mesh
   * @tparam CellT Type of cell
   * @tparam PredT Type of predicate function
   * @param mesh Mesh to create region from
   * @param predicate Function determining if a cell is in the region
   * @param name Optional name for the region
   * @return Region object
   */
  Region filterMesh(
		    const Mesh& mesh,
		    std::function<bool(const Cell*)> predicate,
		    const std::string& name)
  {
    // Create a region definition
    auto definition = std::make_shared<PredicateRegion>(name, predicate);

    // Create a stable ID based on name and current time
    std::hash<std::string> hasher;
    RegionID id = static_cast<RegionID>(hasher(name) % 1000000);

    // Create region with mesh binding
    Region result(id, definition, const_cast<Mesh*>(&mesh));

    return result;
  }

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
  Region createUnionRegion(const Region& regionA, const Region& regionB, const std::string& name) {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
      throw std::invalid_argument("Cannot create union of regions bound to different meshes");
    }

    // Create a composite region definition for the union
    auto compDef = std::make_shared<CompositeRegion>(
						     name,
						     regionA.getDefinition(),
						     regionB.getDefinition(),
						     CompositeRegion::Operation::UNION
						     );

    // Create a stable ID using the robust composite ID generator
    // Use operation code 1 for union
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 1);

    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());

    // Optimal storage selection based on storage modes of inputs
    // If both regions use the same storage mode, use that mode
    if (regionA.getStorageMode() == regionB.getStorageMode()) {
      result.setStorageMode(regionA.getStorageMode());

      if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
	// For cell sets, perform set union
	auto& resultIndices = result.getCellIndicesForWrite();
	const auto& indicesA = regionA.getCellIndices();
	const auto& indicesB = regionB.getCellIndices();

	// Reserve space for efficiency
	resultIndices.reserve(indicesA.size() + indicesB.size());

	// Insert all indices from both regions
	resultIndices.insert(indicesA.begin(), indicesA.end());
	resultIndices.insert(indicesB.begin(), indicesB.end());
      }
      else if (regionA.getStorageMode() == RegionStorageMode::BIT_ARRAY) {
	// For bit arrays, perform bitwise OR
	BitArray resultBits = regionA.getBitArray();
	resultBits.bitwiseOr(regionB.getBitArray());
	result.setBitArray(std::move(resultBits));
      }
    }
    else {
      // Different storage modes - choose based on estimated density
      double densityA = static_cast<double>(regionA.size()) / regionA.getMeshSize();
      double densityB = static_cast<double>(regionB.size()) / regionB.getMeshSize();

      // Estimate density of union: P(A∪B) = P(A) + P(B) - P(A∩B)
      // Assuming independence: P(A∩B) ≈ P(A)P(B)
      double estimatedDensity = densityA + densityB - (densityA * densityB);

      if (estimatedDensity > 0.1) {
	// Dense result - use bit array
	BitArray resultBits = regionA.getBitArray();
	resultBits.bitwiseOr(regionB.getBitArray());
	result.setBitArray(std::move(resultBits));
      }
      else {
	// Sparse result - use cell set
	auto& resultIndices = result.getCellIndicesForWrite();
	const auto& indicesA = regionA.getCellIndices();
	const auto& indicesB = regionB.getCellIndices();

	// Reserve space for efficiency
	resultIndices.reserve(indicesA.size() + indicesB.size());

	// Insert all indices from both regions
	resultIndices.insert(indicesA.begin(), indicesA.end());
	resultIndices.insert(indicesB.begin(), indicesB.end());
      }
    }

    return result;
  }

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
  Region createIntersectionRegion(const Region& regionA, const Region& regionB, const std::string& name) {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
      throw std::invalid_argument("Cannot create intersection of regions bound to different meshes");
    }

    // Create a composite region definition for the intersection
    auto compDef = std::make_shared<CompositeRegion>(
						     name,
						     regionA.getDefinition(),
						     regionB.getDefinition(),
						     CompositeRegion::Operation::INTERSECTION
						     );

    // Create a stable ID using the robust composite ID generator
    // Use operation code 2 for intersection
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 2);

    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());

    // For intersection, it's usually most efficient to use the same storage mode
    // as the smaller of the two regions, since the result can't be larger than that

    // Get sizes to determine which region is smaller
    size_t sizeA = regionA.size();
    size_t sizeB = regionB.size();

    if (sizeA < sizeB) {
      // Region A is smaller, process its elements
      if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
	// Use cell set for the result
	result.setStorageMode(RegionStorageMode::CELL_SET);
	auto& resultIndices = result.getCellIndicesForWrite();
	const auto& indicesA = regionA.getCellIndices();

	// Reserve space for efficiency (worst case: all of smaller region)
	resultIndices.reserve(indicesA.size());

	// Add indices that are in both regions
	for (int idx : indicesA) {
	  if (regionB.containsIndex(idx)) {
	    resultIndices.insert(idx);
	  }
	}
      }
      else {
	// Use bit array for the result
	BitArray resultBits = regionA.getBitArray();
	resultBits.bitwiseAnd(regionB.getBitArray());
	result.setBitArray(std::move(resultBits));
      }
    }
    else {
      // Region B is smaller (or they're equal), process its elements
      if (regionB.getStorageMode() == RegionStorageMode::CELL_SET) {
	// Use cell set for the result
	result.setStorageMode(RegionStorageMode::CELL_SET);
	auto& resultIndices = result.getCellIndicesForWrite();
	const auto& indicesB = regionB.getCellIndices();

	// Reserve space for efficiency (worst case: all of smaller region)
	resultIndices.reserve(indicesB.size());

	// Add indices that are in both regions
	for (int idx : indicesB) {
	  if (regionA.containsIndex(idx)) {
	    resultIndices.insert(idx);
	  }
	}
      }
      else {
	// Use bit array for the result
	BitArray resultBits = regionB.getBitArray();
	resultBits.bitwiseAnd(regionA.getBitArray());
	result.setBitArray(std::move(resultBits));
      }
    }

    // Automatically optimize storage based on result density
    result.optimizeStorage();

    return result;
  }

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
  Region createDifferenceRegion(const Region& regionA, const Region& regionB, const std::string& name) {
    // Validate that both regions use the same mesh
    if (regionA.mesh() != regionB.mesh()) {
      throw std::invalid_argument("Cannot create difference of regions bound to different meshes");
    }

    // Create a composite region definition for the difference
    auto compDef = std::make_shared<CompositeRegion>(
						     name,
						     regionA.getDefinition(),
						     regionB.getDefinition(),
						     CompositeRegion::Operation::DIFFERENCE
						     );

    // Create a stable ID using the robust composite ID generator
    // Use operation code 3 for difference
    RegionID id = generateCompositeID(regionA.getId(), regionB.getId(), 3);

    // Create new region with the composite definition and the same mesh
    Region result(id, compDef, regionA.mesh());

    // For difference, the result can't be larger than region A, so we
    // can use the same storage mode
    if (regionA.getStorageMode() == RegionStorageMode::CELL_SET) {
      // Use cell set for the result
      result.setStorageMode(RegionStorageMode::CELL_SET);
      auto& resultIndices = result.getCellIndicesForWrite();
      const auto& indicesA = regionA.getCellIndices();

      // Reserve space for efficiency (worst case: all of first region)
      resultIndices.reserve(indicesA.size());

      // Add indices that are in A but not in B
      for (int idx : indicesA) {
	if (!regionB.containsIndex(idx)) {
	  resultIndices.insert(idx);
	}
      }
    }
    else {
      // Use bit array for the result
      BitArray resultBits = regionA.getBitArray();
      resultBits.bitwiseAndNot(regionB.getBitArray());
      result.setBitArray(std::move(resultBits));
    }

    // Automatically optimize storage based on result density
    result.optimizeStorage();

    return result;
  }

  /**
   * @brief Create a region from set operation on two existing regions
   *
   * @param regionA First region
   * @param regionB Second region
   * @param operation Operation code (1=union, 2=intersection, 3=difference)
   * @param name Optional name for the region
   * @return New region resulting from the set operation
   */
  Region createRegionFromSetOperation(const Region& regionA, const Region& regionB, size_t operation, const std::string& name) {
    switch (operation) {
    case 1:
      return createUnionRegion(regionA, regionB, name);
    case 2:
      return createIntersectionRegion(regionA, regionB, name);
    case 3:
      return createDifferenceRegion(regionA, regionB, name);
    default:
      throw std::invalid_argument("Invalid operation code: " + std::to_string(operation));
    }
  }

} // namespace mesh
