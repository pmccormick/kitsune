/**
 * @file RegionUtils.hpp
 * @brief Template implementations for RegionUtils.h
 *
 * This file contains the template implementations for the functions declared
 * in RegionUtils.h. It is included at the end of RegionUtils.h.
 */

#ifndef REGION_UTILS_HPP
#define REGION_UTILS_HPP

#include "Region.h"
#include "Cell.h"
#include <type_traits>
#include <concepts>

namespace mesh {

  // Implementation of forEachCellInRegion
  template <typename MeshT, typename FuncT>
  requires std::invocable<FuncT, typename MeshT::cell_type*>
  void forEachCellInRegion(MeshT& mesh, const Region& region, FuncT func) {
    // Choose the most efficient iteration strategy based on storage mode
    if (region.getStorageMode() == RegionStorageMode::CELL_SET) {
      // Iterate through explicit cell indices
      const auto& indices = region.getCellIndices();
      for (int idx : indices) {
	// Convert linear index to 2D coordinates
	auto [i, j] = mesh.toIndices(idx);

	// Get the cell and call the function
	auto* cell = mesh.getTypedCell(i, j);
	if (cell) {
	  func(cell);
	}
      }
    }
    else if (region.getStorageMode() == RegionStorageMode::BIT_ARRAY) {
      // Iterate through bit array using efficient findFirst/findNext
      const BitArray& bitArray = region.getBitArray();

      for (size_t idx = bitArray.findFirst(); idx < bitArray.size(); idx = bitArray.findNext(idx)) {
	// Convert linear index to 2D coordinates
	auto [i, j] = mesh.toIndices(static_cast<int>(idx));

	// Get the cell and call the function
	auto* cell = mesh.getTypedCell(i, j);
	if (cell) {
	  func(cell);
	}
      }
    }
    else if (region.getStorageMode() == RegionStorageMode::DYNAMIC) {
      // For dynamic regions, iterate through all cells in the mesh
      for (int j = 0; j < mesh.ny(); ++j) {
	for (int i = 0; i < mesh.nx(); ++i) {
	  auto* cell = mesh.getTypedCell(i, j);
	  if (cell && region.contains(cell)) {
	    func(cell);
	  }
	}
      }
    }
  }

  // Implementation of getCellsInRegion
  template <typename MeshT>
  std::vector<typename MeshT::cell_type*> getCellsInRegion(MeshT& mesh, const Region& region) {
    std::vector<typename MeshT::cell_type*> cells;

    // Reserve space based on region size for efficiency
    cells.reserve(region.size());

    // Use forEachCellInRegion to fill the vector
    forEachCellInRegion(mesh, region, [&cells](typename MeshT::cell_type* cell) {
      cells.push_back(cell);
    });

    return cells;
  }

  // Implementation of isCellInRegion is inline in RegionUtils.cpp

} // namespace mesh

#endif // REGION_UTILS_HPP
