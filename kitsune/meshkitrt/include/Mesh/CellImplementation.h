/**
 * @file CellImplementation.h
 * @brief Implementation of Cell methods that depend on complete Mesh definition
 *
 * This file contains the implementations of Cell methods that require the Mesh
 * class to be completely defined. It resolves circular dependencies between
 * Cell and Mesh by separating the implementation from the declaration.
 *
 * DEVELOPER NOTES:
 * - This file should be included after both Mesh.h and Cell.h
 * - All methods are inlined for performance
 * - This approach maintains high performance while breaking circular dependencies
 *
 * @see Cell.h, Mesh.h
 */

#ifndef MESH_CELL_IMPLEMENTATION_H
#define MESH_CELL_IMPLEMENTATION_H

#include "Mesh.h"
#include "Cell.h"

namespace mesh {

  // Implementation of methods that depend on the complete Mesh definition

  /**
   * @brief Converts the 2D cell indices to a 1D linear index.
   *
   * This method relies on the Mesh's linearIndex() function. 
   * The linear index is used for efficient storage and lookup in grid-based data structures.
   *
   * @return int Linear index corresponding to the cell
   * @throws std::logic_error if the cell is invalid
   */
  [[clang::always_inline]] int Cell::linearIndex() const {
    if (!isValid()) {
      throw std::logic_error("Cannot compute linear index: invalid cell");
    }
    return m_mesh->linearIndex(m_i, m_j);
  }

  /**
   * @brief Determines whether the cell is at the boundary of the mesh.
   *
   * A cell is on the boundary if any of its indices are at 0 or at the maximum value.
   * Boundary detection is important for applying boundary conditions in simulations.
   *
   * @return true if the cell is a boundary cell; false otherwise
   * @throws std::logic_error if the cell is invalid
   */
  [[clang::always_inline]] bool Cell::isBoundary() const {
    if (!isValid()) {
      throw std::logic_error("Cannot determine boundary status: invalid cell");
    }
    return (m_i == 0 || m_j == 0 || m_i == m_mesh->nx() - 1 || m_j == m_mesh->ny() - 1);
  }

  /**
   * @brief Returns a neighboring cell in the specified direction.
   *
   * The direction is specified using bit flags:
   *   - NORTH (0x01): Cell above
   *   - EAST (0x02): Cell to the right
   *   - SOUTH (0x04): Cell below
   *   - WEST (0x08): Cell to the left
   *
   * Direction flags can be combined to get diagonal neighbors:
   *   - NORTH | EAST: Northeast diagonal
   *   - NORTH | WEST: Northwest diagonal
   *   - SOUTH | EAST: Southeast diagonal
   *   - SOUTH | WEST: Southwest diagonal
   *
   * If the neighbor would fall outside the mesh, an invalid cell is returned.
   * This method is essential for stencil operations and neighborhood-based algorithms.
   *
   * @param direction Bit flag indicating the desired direction
   * @return Cell Neighbor cell view
   * @throws std::logic_error if the current cell is invalid
   */
  [[clang::always_inline]] Cell Cell::neighbor(uint8_t direction) const {
    if (!isValid()) {
      throw std::logic_error("Cannot get neighbor: invalid cell");
    }

    auto [di, dj] = getDirectionOffset(direction);
    int ni = m_i + di;
    int nj = m_j + dj;

    // Check for out-of-bound indices
    if (ni < 0 || ni >= m_mesh->nx() || nj < 0 || nj >= m_mesh->ny()) {
      return Cell(nullptr, -1, -1);  // Return invalid cell
    }

    return Cell(m_mesh, ni, nj);
  }

  /**
   * @brief Checks whether the cell is valid.
   *
   * A valid cell has a non-null mesh pointer and indices within the mesh bounds.
   * Many cell operations check validity before proceeding.
   *
   * @return true if valid, false otherwise
   */
  [[clang::always_inline]] bool Cell::isValid() const {
    return m_mesh != nullptr && m_i >= 0 && m_j >= 0 &&
           m_i < m_mesh->nx() && m_j < m_mesh->ny();
  }

  /**
   * @brief Implementation of the Mesh::getCell method.
   *
   * Creates a lightweight Cell view for the requested position.
   * This factory method is how cells are typically accessed from a mesh.
   *
   * @param i Index in x-direction (column)
   * @param j Index in y-direction (row)
   * @return Cell View of the cell at (i,j)
   */
  //[[clang::always_inline]] Cell Mesh::getCell(uint32_t i, uint32_t j) const {
  //  return Cell(const_cast<Mesh*>(this), static_cast<int>(i), static_cast<int>(j));
  //}

} // namespace mesh

#endif // MESH_CELL_IMPLEMENTATION_H
