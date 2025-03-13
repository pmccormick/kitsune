#ifndef FIELD_H
#define FIELD_H

#include "Mesh.h"
#include <vector>
#include <cassert>
#include <cstdint>
#include <string>

namespace mesh {

  /**
   * @brief Templated Field class for high-performance mesh-based simulations.
   *
   * This class provides a minimal, inline interface for storing and accessing
   * data associated with each cell of a mesh. It decouples the index mapping by
   * calling mesh.linearIndex(i, j), which can be overridden or specialized to support
   * different storage layouts (e.g., row-major, blocked, Z-order). All operations
   * are defined inline to enable full compiler optimization.
   *
   * @see Mesh::linearIndex(), field::storage free functions in FieldStorage.h.
   */
  template <typename T>
  class Field {
  public:
    /**
     * @brief Construct a new Field object.
     *
     * Allocates a one-dimensional vector with size equal to the total number of
     * cells in the mesh and initializes all elements with defaultValue.
     *
     * @param mesh Reference to the associated Mesh.
     * @param defaultValue Default value for each field element.
     */
    Field(const Mesh& mesh, const T& defaultValue = T())
      : m_mesh(mesh),
	m_data(static_cast<size_t>(mesh.nx()) * mesh.ny(), defaultValue)
    {}

    /**
     * @brief Inlined element access operator (non-const).
     *
     * Returns a mutable reference to the field element at cell (i, j). The underlying
     * linear index is computed using the inline mesh.linearIndex(i, j) function.
     *
     * @param i Column index.
     * @param j Row index.
     * @return T& Reference to the field element.
     */
    [[clang::always_inline]] T& operator()(int i, int j) {
      int idx = m_mesh.linearIndex(i, j);
      assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
      return m_data[static_cast<size_t>(idx)];
    }

    /**
     * @brief Inlined element access operator (const).
     *
     * Provides read-only access to the field element at cell (i, j).
     *
     * @param i Column index.
     * @param j Row index.
     * @return const T& Constant reference to the field element.
     */
    [[clang::always_inline]] const T& operator()(int i, int j) const {
      int idx = m_mesh.linearIndex(i, j);
      assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
      return m_data[static_cast<size_t>(idx)];
    }

    /**
     * @brief Returns a pointer to the underlying raw data array (non-const).
     *
     * @return T* Pointer to the field data.
     */
    [[clang::always_inline]] T* data() { return m_data.data(); }

    /**
     * @brief Returns a pointer to the underlying raw data array (const).
     *
     * @return const T* Pointer to the field data.
     */
    [[clang::always_inline]] const T* data() const { return m_data.data(); }

    /**
     * @brief Returns the total number of elements in the field.
     *
     * @return size_t Total number of field elements.
     */
    [[clang::always_inline]] size_t size() const { return m_data.size(); }

    /**
     * @brief Returns the number of cells in the x-direction.
     *
     * @return uint32_t Number of cells in x-direction.
     */
    [[clang::always_inline]] uint32_t nx() const { return m_mesh.nx(); }

    /**
     * @brief Returns the number of cells in the y-direction.
     *
     * @return uint32_t Number of cells in y-direction.
     */
    [[clang::always_inline]] uint32_t ny() const { return m_mesh.ny(); }

  private:
    const Mesh& m_mesh; ///< Reference to the associated mesh.
    std::vector<T> m_data;  ///< Linear storage for field elements.
  };

} // namespace mesh

#endif // FIELD_H
