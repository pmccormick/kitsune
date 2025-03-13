/**
 * @file RegionDefinition.cpp
 * @brief Implementation of the CompositeRegion class methods
 */

#include "RegionDefinition.h"

namespace mesh {

  /**
   * @brief Check if a cell belongs to this region using the composite operation
   *
   * Evaluates the set operation between the two component regions.
   *
   * @param cell Pointer to the cell to check
   * @return true if the cell satisfies the composite criteria, false otherwise
   */
  bool CompositeRegion::contains(const mesh::Cell* cell) const {
    if (!cell) return false;

    // Evaluate based on the operation type
    switch (m_operation) {
    case Operation::UNION:
      return m_regionA->contains(cell) || m_regionB->contains(cell);

    case Operation::INTERSECTION:
      return m_regionA->contains(cell) && m_regionB->contains(cell);

    case Operation::DIFFERENCE:
      return m_regionA->contains(cell) && !m_regionB->contains(cell);

    default:
      return false;
    }
  }

  /**
   * @brief Get the name of the region definition
   *
   * @return Region definition name
   */
  std::string CompositeRegion::getName() const {
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
  std::pair<std::pair<int, int>, std::pair<int, int>> CompositeRegion::getBounds() const {
    // Get bounds from both component regions
    auto boundsA = m_regionA->getBounds();
    auto boundsB = m_regionB->getBounds();

    // For UNION: expand bounds to cover both regions
    if (m_operation == Operation::UNION) {
      return {
	{std::min(boundsA.first.first, boundsB.first.first),
	 std::min(boundsA.first.second, boundsB.first.second)},
	{std::max(boundsA.second.first, boundsB.second.first),
	 std::max(boundsA.second.second, boundsB.second.second)}
      };
    }

    // For INTERSECTION and DIFFERENCE: just use the bounds of the first region
    // as these operations can't produce results larger than the first region
    return boundsA;
  }

  /**
   * @brief Get the operation type for this composite region
   *
   * @return Operation type
   */
  CompositeRegion::Operation CompositeRegion::getOperation() const {
    return m_operation;
  }

} // namespace mesh
