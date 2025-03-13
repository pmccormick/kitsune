/**
 * @file CellImpl.cpp
 * @brief Implementation of Cell methods that depend on complete Mesh definition
 *
 * This file contains the implementations of Cell methods that require full knowledge
 * of the Mesh class definition. By separating these implementations from the Cell.h header,
 * we resolve circular dependencies between Cell and Mesh classes.
 *
 * @see Cell.h, Mesh.h
 */

#include "Mesh.h"
#include "Cell.h"

namespace mesh {

/**
 * @brief Converts the 2D cell indices to a 1D linear index.
 *
 * This method relies on the Mesh's linearIndex() function to convert
 * the cell's (i,j) coordinates to a linear index for storage operations.
 *
 * @return int Linear index corresponding to the cell.
 * @throws std::logic_error if the cell is invalid.
 */
int Cell::linearIndex() const {
  if (!isValid()) {
    throw std::logic_error("Cannot compute linear index: invalid cell");
  }
  
  // Cell indices are guaranteed to be non-negative by isValid() check
  uint32_t ui = static_cast<uint32_t>(m_i);
  uint32_t uj = static_cast<uint32_t>(m_j);
  
  try {
    return static_cast<int>(m_mesh->linearIndex(ui, uj));
  } catch (const std::exception& e) {
    // Translate any mesh exceptions for better context
    throw std::logic_error(std::string("Error computing linear index: ") + e.what());
  }
}

/**
 * @brief Determines whether the cell is at the boundary of the mesh.
 *
 * A cell is on the boundary if any of its indices are at 0 or at the maximum value.
 * This is used for applying boundary conditions in simulations.
 *
 * @return true if the cell is a boundary cell; false otherwise.
 * @throws std::logic_error if the cell is invalid.
 */
bool Cell::isBoundary() const {
  if (!isValid()) {
    throw std::logic_error("Cannot determine boundary status: invalid cell");
  }
  
  // We already verified that m_i and m_j are >= 0 in isValid()
  return (m_i == 0 || m_j == 0 || 
          m_i == static_cast<int>(m_mesh->nx() - 1) || 
          m_j == static_cast<int>(m_mesh->ny() - 1));
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
 *
 * @param direction Bit flag indicating the desired direction.
 * @return Cell Neighbor cell view, or an invalid cell if out of bounds.
 * @throws std::logic_error if the current cell is invalid.
 */
Cell Cell::neighbor(uint8_t direction) const {
  if (!isValid()) {
    throw std::logic_error("Cannot get neighbor: invalid cell");
  }

  // Get direction offsets
  auto [di, dj] = getDirectionOffset(direction);
  
  // Calculate new indices, being careful with integer arithmetic
  int ni = m_i + di;
  int nj = m_j + dj;

  // Check for out-of-bound indices
  if (ni < 0 || ni >= static_cast<int>(m_mesh->nx()) || 
      nj < 0 || nj >= static_cast<int>(m_mesh->ny())) {
    // Return an invalid cell - safer than throwing an exception
    // since boundary conditions often check for neighbors
    return Cell();
  }

  // Create a new cell with the neighbor coordinates
  return Cell(m_mesh, ni, nj);
}

/**
 * @brief Checks whether the cell is valid.
 *
 * A valid cell has a non-null mesh pointer and indices within the mesh bounds.
 * Many cell operations check validity before proceeding.
 *
 * @return true if valid, false otherwise.
 */
bool Cell::isValid() const {
  // A cell is valid if:
  // 1. It has a non-null mesh pointer
  if (m_mesh == nullptr) {
    return false;
  }
  
  // 2. Its indices are non-negative
  if (m_i < 0 || m_j < 0) {
    return false;
  }
  
  // 3. Its indices are within the mesh bounds
  if (static_cast<uint32_t>(m_i) >= m_mesh->nx() || 
      static_cast<uint32_t>(m_j) >= m_mesh->ny()) {
    return false;
  }
  
  return true;
}

} // namespace mesh
  //
