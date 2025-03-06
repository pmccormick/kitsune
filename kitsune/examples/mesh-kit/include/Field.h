/**
 * @file Field.h
 * @brief Templated Field class for high‑performance, type‑safe data storage.
 *
 * ================================================================================
 * Design Considerations and Future Directions:
 * ================================================================================
 * This Field class is designed to support a structure‑of‑arrays (SoA) storage
 * model optimized for modern CPU vectorization and GPU memory coalescing. By
 * using a type tag as a template parameter, we capture the storage location of
 * the data (cell center, vertex, horizontal edge, or vertical edge) at compile
 * time. This approach provides:
 *
 *   • **Type Safety and Clarity:**
 *     Using distinct type tags (e.g., CellCenterTag, VertexTag,
 * HorizontalEdgeTag, VerticalEdgeTag) enforces that only the correct Field type
 * is used in any given computation. For example, a function expecting vertex
 * data can be declared to accept only a VertexField, reducing the chance of
 * accidental misuse.
 *
 *   • **Lightweight Abstraction:**
 *     The Field class wraps a std::vector<T> to ensure that the data is stored
 *     contiguously in memory. All indexing operations are implemented inline
 * (using if-constexpr) so that compilers can fully optimize away any overhead
 * in inner loops.
 *
 *   • **Flexible Dimension Handling:**
 *     The dimensions of the Field depend on the type tag. For a grid with nx x
 * ny cells:
 *       - Cell-centered data has dimensions:      (nx) x (ny)
 *       - Vertex-centered data has dimensions:     (nx+1) x (ny+1)
 *       - Horizontal edge data has dimensions:     (nx) x (ny+1)
 *       - Vertical edge data has dimensions:       (nx+1) x (ny)
 *
 *     Additionally, this implementation supports a third dimension (D) which
 * can be used for properties, vertices, or other components that have multiple
 * values per cell.
 *
 *   • **Future Considerations:**
 *     - **Custom Allocators and Alignment:** In the future, custom allocators
 * could be integrated (or memory alignment options added) for improved
 * performance on GPUs or with SIMD instructions.
 *     - **Enhanced Operations:** Additional operator overloads and element-wise
 * operations, as well as integration with parallel libraries, could further
 * enhance usability.
 *     - **Debugging and Error Checking:** Expanded debug modes (e.g., optional
 * bounds- checking with detailed error messages) could be provided for
 * development.
 *     - **Interoperability:** Interfaces to high-performance libraries (e.g.,
 * BLAS, Thrust, or CUDA libraries) may be added to enable direct use of the
 * Field data in scientific computations.
 *
 * ================================================================================
 * Role in Computational Science:
 * ================================================================================
 * In computational science, especially in areas like computational fluid
 * dynamics (CFD) and finite element analysis, performance and memory locality
 * are crucial. The Field class:
 *
 *   - Encapsulates raw simulation data (e.g., temperature, pressure, velocity)
 * in a contiguous layout to maximize memory bandwidth.
 *   - Separates the physical data storage from the simulation logic, allowing
 * simulation code (for example, using the Cell abstraction) to remain clean and
 * maintainable.
 *   - Provides compile-time type safety to ensure that functions operating on
 * cell centers, vertices, or edges are not inadvertently mixed.
 *
 * This design is a step toward portable, high‑performance scientific computing
 * that maintains both clarity at the high level and efficiency at the low
 * level.
 *
 * ================================================================================
 */

#ifndef FIELD_H
#define FIELD_H

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

//------------------------------------------------------------------------------
// Type tag structures to denote where a field's data is stored in a mesh grid.
// These tags are used solely for compile-time type safety and clarity.
struct CellCenterTag {};
struct VertexTag {};
struct HorizontalEdgeTag {};
struct VerticalEdgeTag {};

//------------------------------------------------------------------------------
// Field class template.
//   T: the data type (for example, double)
//   LocationTag: a type tag that indicates the storage location.
//                Use one of: CellCenterTag, VertexTag, HorizontalEdgeTag,
//                VerticalEdgeTag.
//   D: the third dimension size (default=1 for 2D fields)
template <typename T, typename LocationTag, size_t D = 1> class Field {
public:
  /**
   * @brief Constructor.
   *
   * @param nx Base number of cells in x-direction (for cell-centered data).
   * @param ny Base number of cells in y-direction (for cell-centered data).
   *
   * The actual dimensions for storage are determined by the LocationTag:
   *   - CellCenterTag:      Dimensions: nx x ny x D
   *   - VertexTag:          Dimensions: (nx+1) x (ny+1) x D
   *   - HorizontalEdgeTag:  Dimensions: nx x (ny+1) x D
   *   - VerticalEdgeTag:    Dimensions: (nx+1) x ny x D
   */
  Field(size_t nx, size_t ny) : m_nx(nx), m_ny(ny) {
    size_t total = computeTotalSize();
    m_data.resize(total);
  }

  /**
   * @brief Overloaded access operator for 2D fields (non-const).
   *
   * Provides inline 2D access to the field data.
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @return Reference to the element at (i,j).
   */
  inline T &operator()(size_t i, size_t j) {
    size_t idx = index(i, j, 0);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 2D fields (const).
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @return Const reference to the element at (i,j).
   */
  inline const T &operator()(size_t i, size_t j) const {
    size_t idx = index(i, j, 0);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 3D fields (non-const).
   *
   * Provides inline 3D access to the field data.
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @param k Index in the third dimension.
   * @return Reference to the element at (i,j,k).
   */
  inline T &operator()(size_t i, size_t j, size_t k) {
    size_t idx = index(i, j, k);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /**
   * @brief Overloaded access operator for 3D fields (const).
   *
   * @param i Index in the x-direction.
   * @param j Index in the y-direction.
   * @param k Index in the third dimension.
   * @return Const reference to the element at (i,j,k).
   */
  inline const T &operator()(size_t i, size_t j, size_t k) const {
    size_t idx = index(i, j, k);
    assert(idx < m_data.size());
    return m_data[idx];
  }

  /// @brief Returns the total number of elements in the field.
  inline size_t size() const { return m_data.size(); }

  /// @brief Returns the size in the x-direction.
  inline size_t nx() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag> ||
                  std::is_same_v<LocationTag, HorizontalEdgeTag>) {
      return m_nx;
    } else {
      return m_nx + 1;
    }
  }

  /// @brief Returns the size in the y-direction.
  inline size_t ny() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag> ||
                  std::is_same_v<LocationTag, VerticalEdgeTag>) {
      return m_ny;
    } else {
      return m_ny + 1;
    }
  }

  /// @brief Returns the size in the third dimension.
  inline constexpr size_t depth() const { return D; }

  /// @brief Returns the base number of cells in x-direction.
  inline size_t baseNx() const { return m_nx; }

  /// @brief Returns the base number of cells in y-direction.
  inline size_t baseNy() const { return m_ny; }

  /**
   * @brief Computes the one-dimensional index from three-dimensional (i,j,k)
   * indices.
   *
   * The valid range for (i,j) depends on the field type:
   *   - For CellCenterTag:      i in [0, nx),     j in [0, ny),     k in [0, D)
   *   - For VertexTag:          i in [0, nx+1),   j in [0, ny+1),   k in [0, D)
   *   - For HorizontalEdgeTag:  i in [0, nx),     j in [0, ny+1),   k in [0, D)
   *   - For VerticalEdgeTag:    i in [0, nx+1),   j in [0, ny),     k in [0, D)
   *
   * @param i Index in x-direction.
   * @param j Index in y-direction.
   * @param k Index in third dimension.
   * @return Linear index into the data vector.
   */
  inline size_t index(size_t i, size_t j, size_t k) const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag>) {
      return (i + j * m_nx) * D + k;
    } else if constexpr (std::is_same_v<LocationTag, VertexTag>) {
      return (i + j * (m_nx + 1)) * D + k;
    } else if constexpr (std::is_same_v<LocationTag, HorizontalEdgeTag>) {
      return (i + j * m_nx) * D + k;
    } else if constexpr (std::is_same_v<LocationTag, VerticalEdgeTag>) {
      return (i + j * (m_nx + 1)) * D + k;
    } else {
      static_assert(sizeof(LocationTag) == 0, "Unknown Field location tag");
      return 0; // unreachable
    }
  }

  /**
   * @brief Fill the entire field with a value.
   * @param value The value to fill with.
   */
  void fill(const T &value) { std::fill(m_data.begin(), m_data.end(), value); }

  /**
   * @brief Get direct access to the underlying data vector.
   * @return Reference to the data vector.
   */
  std::vector<T> &data() { return m_data; }

  /**
   * @brief Get const access to the underlying data vector.
   * @return Const reference to the data vector.
   */
  const std::vector<T> &data() const { return m_data; }

private:
  size_t m_nx;           // Base number of cells in x-direction.
  size_t m_ny;           // Base number of cells in y-direction.
  std::vector<T> m_data; // Contiguous storage for field data.

  /**
   * @brief Computes the total number of elements to allocate.
   *
   * The size depends on the field type (LocationTag):
   *   - For CellCenterTag:      nx * ny * D
   *   - For VertexTag:          (nx+1) * (ny+1) * D
   *   - For HorizontalEdgeTag:  nx * (ny+1) * D
   *   - For VerticalEdgeTag:    (nx+1) * ny * D
   *
   * @return Total number of elements.
   */
  inline size_t computeTotalSize() const {
    if constexpr (std::is_same_v<LocationTag, CellCenterTag>) {
      return m_nx * m_ny * D;
    } else if constexpr (std::is_same_v<LocationTag, VertexTag>) {
      return (m_nx + 1) * (m_ny + 1) * D;
    } else if constexpr (std::is_same_v<LocationTag, HorizontalEdgeTag>) {
      return m_nx * (m_ny + 1) * D;
    } else if constexpr (std::is_same_v<LocationTag, VerticalEdgeTag>) {
      return (m_nx + 1) * m_ny * D;
    } else {
      static_assert(sizeof(LocationTag) == 0, "Unknown Field location tag");
      return 0; // unreachable
    }
  }
};

//------------------------------------------------------------------------------
// Convenience type definitions for common field types. These aliases make code
// more self-documenting and reduce the need to explicitly write the type tag.
template <typename T = double> using CellCenterField = Field<T, CellCenterTag>;

template <typename T = double> using VertexField = Field<T, VertexTag>;

template <typename T = double>
using HorizontalEdgeField = Field<T, HorizontalEdgeTag>;

template <typename T = double>
using VerticalEdgeField = Field<T, VerticalEdgeTag>;

#endif // FIELD_H