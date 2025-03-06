/**
 * @file Field.h
 * @brief Templated Field class for high‑performance, type‑safe data storage.
 *
 * ================================================================================
 * Design Considerations and Future Directions:
 * ================================================================================
 * This Field class is designed to support a structure‑of‑arrays (SoA) storage model
 * optimized for modern CPU vectorization and GPU memory coalescing. By using a type tag
 * as a template parameter, we capture the storage location of the data (cell center,
 * vertex, horizontal edge, or vertical edge) at compile time. This approach provides:
 *
 *   • **Type Safety and Clarity:**
 *     Using distinct type tags (e.g., CellCenterTag, VertexTag, HorizontalEdgeTag,
 *     VerticalEdgeTag) enforces that only the correct Field type is used in any given
 *     computation. For example, a function expecting vertex data can be declared to
 *     accept only a VertexField, reducing the chance of accidental misuse.
 *
 *   • **Lightweight Abstraction:**
 *     The Field class wraps a std::vector<T> to ensure that the data is stored
 *     contiguously in memory. All indexing operations are implemented inline (using
 *     if-constexpr) so that compilers can fully optimize away any overhead in inner
 *     loops.
 *
 *   • **Flexible Dimension Handling:**
 *     The dimensions of the Field depend on the type tag. For a grid with nx x ny cells:
 *       - Cell-centered data has dimensions:      (nx) x (ny)
 *       - Vertex-centered data has dimensions:     (nx+1) x (ny+1)
 *       - Horizontal edge data has dimensions:     (nx) x (ny+1)
 *       - Vertical edge data has dimensions:       (nx+1) x (ny)
 *
 *   • **Future Considerations:**
 *     - **Custom Allocators and Alignment:** In the future, custom allocators could be
 *       integrated (or memory alignment options added) for improved performance on GPUs
 *       or with SIMD instructions.
 *     - **Enhanced Operations:** Additional operator overloads and element-wise operations,
 *       as well as integration with parallel libraries, could further enhance usability.
 *     - **Debugging and Error Checking:** Expanded debug modes (e.g., optional bounds-
 *       checking with detailed error messages) could be provided for development.
 *     - **Interoperability:** Interfaces to high-performance libraries (e.g., BLAS, Thrust,
 *       or CUDA libraries) may be added to enable direct use of the Field data in scientific
 *       computations.
 *
 * ================================================================================
 * Role in Computational Science:
 * ================================================================================
 * In computational science, especially in areas like computational fluid dynamics (CFD)
 * and finite element analysis, performance and memory locality are crucial. The Field class:
 *
 *   - Encapsulates raw simulation data (e.g., temperature, pressure, velocity) in a 
 *     contiguous layout to maximize memory bandwidth.
 *   - Separates the physical data storage from the simulation logic, allowing simulation
 *     code (for example, using the Cell abstraction) to remain clean and maintainable.
 *   - Provides compile-time type safety to ensure that functions operating on cell centers,
 *     vertices, or edges are not inadvertently mixed.
 *
 * This design is a step toward portable, high‑performance scientific computing that
 * maintains both clarity at the high level and efficiency at the low level.
 *
 * ================================================================================
 */

#ifndef FIELD_H
#define FIELD_H

#include <vector>
#include <cstddef>
#include <cassert>
#include <type_traits>

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
//                Use one of: CellCenterTag, VertexTag, HorizontalEdgeTag, VerticalEdgeTag.
template<typename T, typename LocationTag>
class Field {
public:
    /**
     * @brief Constructor.
     *
     * @param nx Base number of cells in x-direction (for cell-centered data).
     * @param ny Base number of cells in y-direction (for cell-centered data).
     *
     * The actual dimensions for storage are determined by the LocationTag:
     *   - CellCenterTag:      Dimensions: nx x ny
     *   - VertexTag:          Dimensions: (nx+1) x (ny+1)
     *   - HorizontalEdgeTag:  Dimensions: nx x (ny+1)
     *   - VerticalEdgeTag:    Dimensions: (nx+1) x ny
     */
    Field(size_t nx, size_t ny)
      : m_nx(nx), m_ny(ny)
    {
        size_t total = computeTotalSize();
        m_data.resize(total);
    }

    /**
     * @brief Overloaded access operator (non-const).
     *
     * Provides inline 2D access to the field data.
     *
     * @param i Index in the x-direction.
     * @param j Index in the y-direction.
     * @return Reference to the element at (i,j).
     */
    inline T& operator()(size_t i, size_t j) {
        size_t idx = index(i, j);
        assert(idx < m_data.size());
        return m_data[idx];
    }
    
    /**
     * @brief Overloaded access operator (const).
     *
     * @param i Index in the x-direction.
     * @param j Index in the y-direction.
     * @return Const reference to the element at (i,j).
     */
    inline const T& operator()(size_t i, size_t j) const {
        size_t idx = index(i, j);
        assert(idx < m_data.size());
        return m_data[idx];
    }

    /// @brief Returns the total number of elements in the field.
    inline size_t size() const { return m_data.size(); }

    /**
     * @brief Computes the one-dimensional index from two-dimensional (i,j) indices.
     *
     * The valid range for (i,j) depends on the field type:
     *   - For CellCenterTag:      i in [0, nx),     j in [0, ny)
     *   - For VertexTag:          i in [0, nx+1),   j in [0, ny+1)
     *   - For HorizontalEdgeTag:  i in [0, nx),     j in [0, ny+1)
     *   - For VerticalEdgeTag:    i in [0, nx+1),   j in [0, ny)
     *
     * @param i Index in x-direction.
     * @param j Index in y-direction.
     * @return Linear index into the data vector.
     */
    inline size_t index(size_t i, size_t j) const {
        if constexpr (std::is_same_v<LocationTag, CellCenterTag>) {
            return i + j * m_nx;
        } else if constexpr (std::is_same_v<LocationTag, VertexTag>) {
            return i + j * (m_nx + 1);
        } else if constexpr (std::is_same_v<LocationTag, HorizontalEdgeTag>) {
            return i + j * m_nx;
        } else if constexpr (std::is_same_v<LocationTag, VerticalEdgeTag>) {
            return i + j * (m_nx + 1);
        } else {
            static_assert(sizeof(LocationTag) == 0, "Unknown Field location tag");
            return 0; // unreachable
        }
    }

private:
    size_t m_nx; // Base number of cells in x-direction.
    size_t m_ny; // Base number of cells in y-direction.
    std::vector<T> m_data; // Contiguous storage for field data.

    /**
     * @brief Computes the total number of elements to allocate.
     *
     * The size depends on the field type (LocationTag):
     *   - For CellCenterTag:      nx * ny
     *   - For VertexTag:          (nx+1) * (ny+1)
     *   - For HorizontalEdgeTag:  nx * (ny+1)
     *   - For VerticalEdgeTag:    (nx+1) * ny
     *
     * @return Total number of elements.
     */
    inline size_t computeTotalSize() const {
        if constexpr (std::is_same_v<LocationTag, CellCenterTag>) {
            return m_nx * m_ny;
        } else if constexpr (std::is_same_v<LocationTag, VertexTag>) {
            return (m_nx + 1) * (m_ny + 1);
        } else if constexpr (std::is_same_v<LocationTag, HorizontalEdgeTag>) {
            return m_nx * (m_ny + 1);
        } else if constexpr (std::is_same_v<LocationTag, VerticalEdgeTag>) {
            return (m_nx + 1) * m_ny;
        } else {
            static_assert(sizeof(LocationTag) == 0, "Unknown Field location tag");
            return 0; // unreachable
        }
    }
};

//------------------------------------------------------------------------------
// Convenience type definitions for common field types. These aliases make code
// more self-documenting and reduce the need to explicitly write the type tag.
using CellCenterField = Field<double, CellCenterTag>;
using VertexField = Field<double, VertexTag>;
using HorizontalEdgeField = Field<double, HorizontalEdgeTag>;
using VerticalEdgeField = Field<double, VerticalEdgeTag>;

#endif // FIELD_H

