/**
 * @file CellIterators.h
 * @brief Iterator interfaces and concrete implementations for cell traversal
 *        using C++20 concepts and static polymorphism.
 *
 * This file builds on our earlier concept‑based iterator implementation by adding
 * additional inline documentation and developer guidance. It demonstrates:
 *
 * 1. Use of C++20 concepts to enforce the cell iterator interface at compile time,
 *    eliminating runtime virtual dispatch.
 * 2. Concrete, fully‑inlined iterator types for traversing all cells, interior cells,
 *    boundary cells, and rectangular regions.
 *
 * DESIGN NOTES:
 * - All iterators generate Cell views on demand; they do not store cell objects.
 * - Developers should prefer these iterators over virtual‐based ones for performance-critical
 *   loops.
 * - Each iterator is intended to be used in a range‑based for loop. Use the provided
 *   helper functions to create ranges.
 *
 * @see Mesh.h, Cell.h
 */

#ifndef CELL_ITERATORS_H
#define CELL_ITERATORS_H

#include "Mesh.h"
#include "Cell.h"
#include <concepts>
#include <array>
#include <vector>

namespace mesh {

  // ----------------------------------------------------------------------------
  // Concept: CellIteratorConcept
  // ----------------------------------------------------------------------------
  /**
   * @brief Concept for a cell iterator.
   *
   * An iterator conforming to CellIteratorConcept must:
   *   - Be dereferenceable to produce a Cell.
   *   - Support pre-increment (which returns a reference to itself).
   *   - Support equality comparison.
   *
   * This concept enables static polymorphism and full inlining.
   */
  template<typename It>
  concept CellIteratorConcept = requires(It it, const It it2) {
    { *it } -> std::convertible_to<Cell>;
    { ++it } -> std::same_as<It&>;
    { it == it2 } -> std::convertible_to<bool>;
  };

  // ----------------------------------------------------------------------------
  // Concrete Iterator Implementations
  // ----------------------------------------------------------------------------

  /**
   * @brief BasicCellIterator traverses all cells in row-major order.
   *
   * This iterator starts at the top-left cell (0,0) and increments the column index first.
   * Once the end of a row is reached, it resets the column index and moves to the next row.
   *
   * @note This iterator is fully inline and free of virtual overhead.
   */
  class BasicCellIterator {
  public:
    [[clang::always_inline]] BasicCellIterator(Mesh* mesh, int i = 0, int j = 0)
      : m_mesh(mesh), m_i(i), m_j(j) {}

    [[clang::always_inline]] Cell operator*() const {
      return Cell(m_mesh, m_i, m_j);
    }
    [[clang::always_inline]] BasicCellIterator& operator++() {
      ++m_i;
      if (m_i >= m_mesh->nx()) {
	m_i = 0;
	++m_j;
      }
      return *this;
    }
    [[clang::always_inline]] bool operator==(const BasicCellIterator& other) const {
      return m_mesh == other.m_mesh && m_i == other.m_i && m_j == other.m_j;
    }
    [[clang::always_inline]] bool operator!=(const BasicCellIterator& other) const {
      return !(*this == other);
    }
  private:
    Mesh* m_mesh; ///< Pointer to the mesh being iterated
    int m_i, m_j;     ///< Current (i,j) indices in the mesh grid
  };

  /**
   * @brief InteriorCellIterator traverses only interior cells.
   *
   * This iterator skips the boundary cells, starting at (1,1) and ending before the last row/column.
   * It is useful when boundary conditions are treated separately.
   *
   * @note Make sure the mesh has at least 3 rows and 3 columns.
   */
  class InteriorCellIterator {
  public:
    [[clang::always_inline]] InteriorCellIterator(Mesh* mesh, int i = 1, int j = 1)
      : m_mesh(mesh), m_i(i), m_j(j) {}
    [[clang::always_inline]] Cell operator*() const {
      return Cell(m_mesh, m_i, m_j);
    }
    [[clang::always_inline]] InteriorCellIterator& operator++() {
      ++m_i;
      if (m_i >= m_mesh->nx() - 1) {
	m_i = 1;
	++m_j;
      }
      return *this;
    }
    [[clang::always_inline]] bool operator==(const InteriorCellIterator& other) const {
      return m_mesh == other.m_mesh && m_i == other.m_i && m_j == other.m_j;
    }
    [[clang::always_inline]] bool operator!=(const InteriorCellIterator& other) const {
      return !(*this == other);
    }
  private:
    Mesh* m_mesh; ///< Mesh pointer
    int m_i, m_j;     ///< Current interior indices (starting at 1)
  };

  /**
   * @brief BoundaryCellIterator traverses the boundary cells in a fixed order.
   *
   * The order is:
   *   1. Bottom row (left-to-right)
   *   2. Right column (excluding corners)
   *   3. Top row (right-to-left)
   *   4. Left column (excluding corners)
   *
   * This iterator is useful for algorithms that need to process boundary conditions.
   */
  class BoundaryCellIterator {
  public:
    [[clang::always_inline]] BoundaryCellIterator(Mesh* mesh, int side = 0, int pos = 0)
      : m_mesh(mesh), m_side(side), m_pos(pos) {}

    [[clang::always_inline]] Cell operator*() const {
      return Cell(m_mesh, i(), j());
    }
    [[clang::always_inline]] BoundaryCellIterator& operator++() {
      ++m_pos;
      switch (m_side) {
      case 0: // Bottom row
	if (m_pos >= m_mesh->nx()) {
	  m_side = 1; m_pos = 1; // Skip the bottom-right corner already visited
	}
	break;
      case 1: // Right column
	if (m_pos >= m_mesh->ny() - 1) {
	  m_side = 2; m_pos = 1;
	}
	break;
      case 2: // Top row (right-to-left)
	if (m_pos >= m_mesh->nx() - 1) {
	  m_side = 3; m_pos = 1;
	}
	break;
      case 3: // Left column (bottom-to-top)
	if (m_pos >= m_mesh->ny() - 2) {
	  m_side = 4; m_pos = 0; // End iterator indicator
	}
	break;
      default:
	break;
      }
      return *this;
    }
    [[clang::always_inline]] bool operator==(const BoundaryCellIterator& other) const {
      return m_mesh == other.m_mesh && m_side == other.m_side && m_pos == other.m_pos;
    }
    [[clang::always_inline]] bool operator!=(const BoundaryCellIterator& other) const {
      return !(*this == other);
    }
    [[clang::always_inline]] int i() const {
      switch (m_side) {
      case 0: return m_pos;
      case 1: return m_mesh->nx() - 1;
      case 2: return m_mesh->nx() - 1 - m_pos;
      case 3: return 0;
      default: return 0;
      }
    }
    [[clang::always_inline]] int j() const {
      switch (m_side) {
      case 0: return 0;
      case 1: return m_pos;
      case 2: return m_mesh->ny() - 1;
      case 3: return m_mesh->ny() - 1 - m_pos;
      default: return 0;
      }
    }
  private:
    Mesh* m_mesh; ///< Mesh pointer
    int m_side;       ///< Current side of the boundary (0: bottom, 1: right, 2: top, 3: left)
    int m_pos;        ///< Current position along the current side
  };

  /**
   * @brief RectangleCellIterator traverses a specified rectangular region.
   *
   * The region is defined by starting indices (inclusive) and ending indices (exclusive).
   * This iterator is useful for localized processing within the mesh.
   */
  class RectangleCellIterator {
  public:
    [[clang::always_inline]] RectangleCellIterator(Mesh* mesh, int startI, int startJ, int endI, int endJ,
					int i = -1, int j = -1)
      : m_mesh(mesh),
	m_startI(startI), m_startJ(startJ), m_endI(endI), m_endJ(endJ),
	m_i(i == -1 ? startI : i), m_j(j == -1 ? startJ : j)
    {}
    [[clang::always_inline]] Cell operator*() const {
      return Cell(m_mesh, m_i, m_j);
    }
    [[clang::always_inline]] RectangleCellIterator& operator++() {
      ++m_i;
      if (m_i >= m_endI) {
	m_i = m_startI;
	++m_j;
      }
      return *this;
    }
    [[clang::always_inline]] bool operator==(const RectangleCellIterator& other) const {
      return m_mesh == other.m_mesh && m_i == other.m_i && m_j == other.m_j;
    }
    [[clang::always_inline]] bool operator!=(const RectangleCellIterator& other) const {
      return !(*this == other);
    }
  private:
    Mesh* m_mesh; ///< Mesh pointer
    int m_startI, m_startJ, m_endI, m_endJ; ///< Region boundaries: [start, end)
    int m_i, m_j; ///< Current indices within the region
  };

  // ----------------------------------------------------------------------------
  // Templated Range Types (Static, No Virtual Overhead)
  // ----------------------------------------------------------------------------

  /**
   * @brief Templated cell range type that accepts any iterator satisfying CellIteratorConcept.
   *
   * This range is used in range-based for loops and contains the begin and end iterators.
   */
  template <CellIteratorConcept Iter>
  struct CellRange {
    [[clang::always_inline]] Iter begin() const { return m_begin; }
    [[clang::always_inline]] Iter end() const { return m_end; }
    Iter m_begin;
    Iter m_end;
  };

  // Define the main CellRange type used in Mesh methods
  using CellRange = CellRange<BasicCellIterator>;

  // ----------------------------------------------------------------------------
  // Helper functions to create common cell ranges
  // ----------------------------------------------------------------------------

  /**
   * @brief Create a range over all cells in the mesh (row-major order).
   *
   * @param mesh Pointer to the mesh.
   * @return CellRange Range of all cells.
   */
  inline CellRange makeBasicCellRange(Mesh* mesh) {
    return { BasicCellIterator(mesh, 0, 0),
	     BasicCellIterator(mesh, 0, mesh->ny()) };
  }

  /**
   * @brief Create a range over interior cells (excluding boundaries).
   *
   * @param mesh Pointer to the mesh.
   * @return CellRange<InteriorCellIterator> Range of interior cells.
   */
  inline CellRange<InteriorCellIterator> makeInteriorCellRange(Mesh* mesh) {
    return { InteriorCellIterator(mesh, 1, 1),
	     InteriorCellIterator(mesh, 1, mesh->ny() - 1) };
  }

  /**
   * @brief Create a range over boundary cells.
   *
   * @param mesh Pointer to the mesh.
   * @return CellRange<BoundaryCellIterator> Range of boundary cells.
   */
  inline CellRange<BoundaryCellIterator> makeBoundaryCellRange(Mesh* mesh) {
    return { BoundaryCellIterator(mesh, 0, 0),
	     BoundaryCellIterator(mesh, 4, 0) }; // 'm_side==4' signals end-of-range
  }

  /**
   * @brief Create a range over a rectangular region of cells.
   *
   * The region is defined by [startI, endI) x [startJ, endJ).
   *
   * @param mesh Pointer to the mesh.
   * @param startI Starting i-index (inclusive).
   * @param startJ Starting j-index (inclusive).
   * @param endI Ending i-index (exclusive).
   * @param endJ Ending j-index (exclusive).
   * @return CellRange<RectangleCellIterator> Range over the specified region.
   */
  inline CellRange<RectangleCellIterator> makeRectangleCellRange(Mesh* mesh,
								 int startI, int startJ,
								 int endI, int endJ) {
    return { RectangleCellIterator(mesh, startI, startJ, endI, endJ),
	     RectangleCellIterator(mesh, startI, startJ, endI, endJ, startI, endJ) };
  }

} // namespace mesh

#endif // CELL_ITERATORS_H
