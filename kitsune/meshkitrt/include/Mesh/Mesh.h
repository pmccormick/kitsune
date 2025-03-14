/**
 * @file Mesh.h
 * @brief Base class for mesh containers with lightweight cell views
 * -------------------------------------------------------------
 *
 * DESIGN PRINCIPLES:
 *
 * 1. FACTORY, NOT CONTAINER: Mesh acts as a factory for Cell views,
 *    not a container of Cell objects. Cells are created on-demand.
 *
 * 2. RANGE-BASED ITERATION: Provides methods that return CellRange objects
 *    for efficient, modern range-based iteration.
 *
 * 3. MINIMAL INTERFACE: Defines only the core interfaces needed by
 *    Cell views and derived mesh classes.
 *
 * 4. PHYSICAL/LOGICAL SEPARATION: Separates logical grid operations from
 *    physical coordinate calculations.
 *
 * 5. CODE GENERATION FRIENDLY: Designed to work with generated specialized
 *    implementations rather than using virtual methods.
 *
 * MESH LAYOUT AND TERMINOLOGY:
 *
 *    ^ j (rows)
 *    |
 *    |     (0,ny-1) --- (nx-1,ny-1)
 *    |        |             |
 *    |        |             |
 *    |     (0,0) ----- (nx-1,0)
 *    |
 *    +----------------------> i (columns)
 *
 * Where:
 *   - Origin (0,0) is at the bottom-left
 *   - i increases to the right (columns)
 *   - j increases upward (rows)
 *   - Each grid cell is identified by its (i,j) indices
 *
 * @note For junior developers:
 * - Cells are NOT stored in the mesh; they are created on-demand as views.
 * - Use range-based for loops with cells(), interiorCells(), etc.
 * - When extending this class, focus on specializing the field storage.
 */

#ifndef MESH_H
#define MESH_H

#include <utility>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>

// Include Cell.h to get Cell class declaration.
// The Cell class only has forward declaration of Mesh, so we can include it here.
#include "Cell.h"

namespace mesh {
  // Forward declarations for types defined elsewhere
  class CellRange;

  // Type aliases for IDs
  using FieldID = uint32_t;

  /**
   * @brief Abstract base class for mesh containers
   *
   * Provides common functionality and interface for all mesh types.
   * Acts as a factory for Cell views rather than storing Cell objects.
   * 
   * The Mesh class represents a 2D structured grid of cells, providing:
   * - Grid dimension information (nx, ny)
   * - Cell factory methods
   * - Index conversion utilities
   * - Iterators for traversing cells in various patterns
   */
  class Mesh {
  public:
    /**
     * @brief Construct a new Mesh
     *
     * Creates a mesh with the specified dimensions. The mesh represents
     * a 2D grid of cells with nx columns and ny rows.
     *
     * @param nx Number of cells in x-direction (columns)
     * @param ny Number of cells in y-direction (rows)
     * @throws std::invalid_argument If dimensions are not positive
     */
    Mesh(uint32_t nx, uint32_t ny)
      : m_nx(nx), m_ny(ny)
    {
      // Validate parameters
      if (nx == 0 || ny == 0) {
        throw std::invalid_argument("Mesh dimensions must be positive");
      }
    }

    /**
     * @brief Virtual destructor
     * 
     * Virtual destructor ensures proper cleanup for derived classes.
     */
    virtual ~Mesh() = default;

    /**
     * @brief Get the number of cells in the x-direction
     *
     * @return uint32_t Number of cells in x-direction (columns)
     */
    uint32_t nx() const { return m_nx; }

    /**
     * @brief Get the number of cells in the y-direction
     *
     * @return uint32_t Number of cells in y-direction (rows)
     */
    uint32_t ny() const { return m_ny; }

    /**
     * @brief Get the total number of cells in the mesh
     *
     * @return uint32_t Total number of cells (nx * ny)
     */
    uint32_t size() const { return m_nx * m_ny; }

    /**
     * @brief Convert 2D indices to linear index
     *
     * Maps (i,j) indices to a linear index for array access.
     * For a row-major layout, this is typically i + j*nx.
     *
     * @param i Index in x-direction (column)
     * @param j Index in y-direction (row)
     * @return uint32_t Linear index
     * @throws std::out_of_range If indices are outside mesh bounds
     */
    uint32_t linearIndex(uint32_t i, uint32_t j) const {
      if (i >= m_nx || j >= m_ny) {
        throw std::out_of_range("Cell indices out of bounds");
      }
      return i + j * m_nx;
    }

    /**
     * @brief Convert linear index to 2D indices
     *
     * Converts a linear index back to (i,j) coordinates. This is the
     * inverse of the linearIndex function.
     *
     * @param linearIdx Linear index
     * @return std::pair<uint32_t, uint32_t> (i,j) indices (column, row)
     * @throws std::out_of_range If linearIdx is outside mesh bounds
     */
    std::pair<uint32_t, uint32_t> toIndices(uint32_t linearIdx) const {
      if (linearIdx >= size()) {
        throw std::out_of_range("Linear index out of bounds");
      }
      uint32_t j = linearIdx / m_nx;
      uint32_t i = linearIdx % m_nx;
      return {i, j};
    }

    /**
     * @brief Check if indices are within mesh bounds
     *
     * Validates whether the given indices fall within the mesh dimensions.
     *
     * @param i Index in x-direction (column)
     * @param j Index in y-direction (row)
     * @return true If indices are valid
     * @return false If indices are outside mesh bounds
     */
    bool isValidIndex(uint32_t i, uint32_t j) const {
      return (i < m_nx && j < m_ny);
    }

    /**
     * @brief Get a cell view at the specified indices
     *
     * This method creates a lightweight Cell view for the
     * specified location. This does not store the Cell;
     * it simply returns a view.
     *
     * @param i Index in x-direction (column)
     * @param j Index in y-direction (row)
     * @return Cell View of the cell at (i,j)
     */
    [[clang::always_inline]] Cell getCell(uint32_t i, uint32_t j) const {
      return Cell(const_cast<Mesh*>(this), static_cast<int>(i), static_cast<int>(j));
    }

    /**
     * @brief Get range for iterating over all cells
     *
     * Returns a CellRange that can be used in range-based for loops
     * to iterate over all cells in the mesh. Cells are visited in
     * row-major order (increasing i, then increasing j).
     *
     * Example:
     *   for (const auto& cell : mesh.cells()) {
     *     // Use cell here
     *   }
     *
     * @return CellRange Range for iterating over all cells
     */
    CellRange cells() const;

    /**
     * @brief Get range for iterating over interior cells
     *
     * Returns a CellRange for cells that are not on the boundary.
     * This is useful for algorithms that need to process interior
     * cells differently from boundary cells.
     *
     * Interior cells satisfy: 0 < i < nx-1 && 0 < j < ny-1
     *
     * @return CellRange Range for iterating over interior cells
     */
    CellRange interiorCells() const;

    /**
     * @brief Get range for iterating over boundary cells
     *
     * Returns a CellRange for cells on the mesh boundaries.
     * Boundary cells satisfy: i == 0 || j == 0 || i == nx-1 || j == ny-1
     *
     * @return CellRange Range for iterating over boundary cells
     */
    CellRange boundaryCells() const;

    /**
     * @brief Get range for iterating over a rectangular region
     *
     * Returns a CellRange for cells in the specified rectangular region.
     * This is useful for processing specific subregions of the mesh.
     *
     * @param startI Starting i-index (inclusive)
     * @param startJ Starting j-index (inclusive)
     * @param endI Ending i-index (exclusive)
     * @param endJ Ending j-index (exclusive)
     * @return CellRange Range for iterating over the region
     * @throws std::out_of_range If region extends outside mesh bounds
     */
    CellRange cellsInRegion(uint32_t startI, uint32_t startJ,
                            uint32_t endI, uint32_t endJ) const;

  protected:
    uint32_t m_nx;             ///< Number of cells in x-direction (columns)
    uint32_t m_ny;             ///< Number of cells in y-direction (rows)

    /**
     * @brief Friend declaration for Cell
     *
     * This allows Cell to access protected members of Mesh.
     */
    friend class Cell;

    // For specialized mesh implementations by code generator
    #ifdef GENERATED_MESH_FRIENDS
    #include "generated_mesh_friends.h"
    #endif
  };

} // namespace mesh

#endif // MESH_H
