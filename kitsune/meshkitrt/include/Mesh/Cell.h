/**
 * @file Cell.h
 * @brief Lightweight Cell class for high-performance mesh-based simulations.
 *
 * This file defines the lightweight Cell view that is used in conjunction
 * with the concept-based iterators. It retains the design principles:
 *
 * 1. Lightweight View: Contains only a mesh pointer and grid indices.
 * 2. Logical Grid Only: All physical calculations are delegated to Mesh.
 * 3. Performance First: Core methods are designed for maximum compiler optimization.
 *
 * ADDITIONAL DEVELOPER NOTES:
 * - Do not add physical coordinate calculations to this class; these belong in Mesh.
 * - This class is intended for use in performance-critical loops and range-based iteration.
 *
 * @see Mesh.h, CellIterators.h
 */

#ifndef MESH_CELL_H
#define MESH_CELL_H

#include <array>
#include <cassert>
#include <cstdint>
#include <utility>
#include <stdexcept>

// Forward declarations to avoid circular dependencies
namespace mesh {
  class Mesh;

  /**
   * @brief Direction enum for cell navigation
   */
  enum Direction : uint8_t {
    NORTH = 0x01, ///< Cell above
    EAST  = 0x02, ///< Cell to the right
    SOUTH = 0x04, ///< Cell below
    WEST  = 0x08  ///< Cell to the left
  };

  /**
   * @brief Represents a lightweight view of a cell in a mesh.
   *
   * The Cell class provides minimal storage (only a mesh pointer and indices)
   * and basic navigation functions. It is designed to be created on-demand during iteration.
   * 
   * Key design features:
   * - Extremely lightweight (only 3 fields: mesh pointer and 2 indices)
   * - Created on-demand by Mesh factory methods
   * - No data storage (data is managed separately by Field classes)
   * - Thread-safe for reading (no internal mutable state)
   */
  class Cell {
  public:
    /**
     * @brief Default constructor creates an invalid cell.
     *
     * An invalid cell has a nullptr mesh pointer and negative indices.
     * This is useful for representing "no cell" in algorithms.
     */
    Cell() : m_mesh(nullptr), m_i(-1), m_j(-1) {}

    /**
     * @brief Get the column index (i) of the cell.
     * @return int Column index (horizontal position).
     */
    int i() const { return m_i; }

    /**
     * @brief Get the row index (j) of the cell.
     * @return int Row index (vertical position).
     */
    int j() const { return m_j; }

    /**
     * @brief Get a pointer to the owning mesh.
     * @return Mesh* Pointer to the mesh.
     */
    Mesh* mesh() const { return m_mesh; }

    /**
     * @brief Returns the cell indices as a pair (i, j).
     * @return std::pair<int, int> Pair containing the cell indices (column, row).
     */
    std::pair<int, int> indices() const { return {m_i, m_j}; }

    /**
     * @brief Converts the 2D cell indices to a 1D linear index.
     *
     * This method relies on the Mesh's linearIndex() function. It throws
     * a std::logic_error if the cell is invalid.
     *
     * @return int Linear index corresponding to the cell.
     * @throws std::logic_error if the cell is invalid.
     */
    int linearIndex() const;

    /**
     * @brief Determines whether the cell is at the boundary of the mesh.
     *
     * A cell is on the boundary if any of its indices are at 0 or at the maximum value.
     *
     * @return true if the cell is a boundary cell; false otherwise.
     * @throws std::logic_error if the cell is invalid.
     */
    bool isBoundary() const;

    /**
     * @brief Returns a neighboring cell in the specified direction.
     *
     * The direction is specified using bit flags:
     *   - NORTH, EAST, SOUTH, WEST, and their combinations.
     * If the neighbor would fall outside the mesh, an invalid cell is returned.
     *
     * @param direction Bit flag indicating the desired direction.
     * @return Cell Neighbor cell view.
     * @throws std::logic_error if the current cell is invalid.
     */
    Cell neighbor(uint8_t direction) const;

    /**
     * @brief Equality operator compares two cells for the same mesh and indices.
     */
    bool operator==(const Cell& other) const {
      return m_mesh == other.m_mesh && m_i == other.m_i && m_j == other.m_j;
    }

    /**
     * @brief Inequality operator.
     */
    bool operator!=(const Cell& other) const {
      return !(*this == other);
    }

    /**
     * @brief Checks whether the cell is valid.
     *
     * A valid cell has a non-null mesh pointer and indices within the mesh bounds.
     *
     * @return true if valid, false otherwise.
     */
    bool isValid() const;

    /**
     * @brief Utility: Returns the (di,dj) offset corresponding to a direction flag.
     *
     * @param direction Direction bit flag.
     * @return std::pair<int, int> Pair containing the offset.
     */
    static std::pair<int, int> getDirectionOffset(uint8_t direction) {
      int di = 0;
      int dj = 0;
      // Add offsets for each direction flag.
      if (direction & NORTH) dj += 1;
      if (direction & SOUTH) dj -= 1;
      if (direction & EAST)  di += 1;
      if (direction & WEST)  di -= 1;
      return {di, dj};
    }

  private:
    // Private constructor; only Mesh and iterator classes can create a valid cell.
    Cell(Mesh* mesh, int i, int j)
      : m_mesh(mesh), m_i(i), m_j(j) {}

    Mesh* m_mesh; ///< Pointer to the mesh containing this cell.
    int m_i;      ///< Column index.
    int m_j;      ///< Row index.

    // Grant friend access to Mesh and iterator classes for constructing cells.
    friend class Mesh;
    friend class BasicCellIterator;
    friend class InteriorCellIterator;
    friend class BoundaryCellIterator;
    friend class RectangleCellIterator;
  };

} // namespace mesh

#endif // MESH_CELL_H
       //
