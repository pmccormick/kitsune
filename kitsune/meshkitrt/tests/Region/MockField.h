/**
 * @file MockField.h
 * @brief Mock field implementation for testing
 */

#ifndef MOCK_FIELD_H
#define MOCK_FIELD_H

#include "Field.h"
#include <vector>

namespace mesh {

/**
 * @class MockField
 * @brief Mock field implementation for testing
 */
template<typename T>
class MockField {
public:
    /**
     * @brief Construct a new Mock Field
     * 
     * @param mesh Reference to the mesh
     * @param defaultValue Default value for initialization
     */
    MockField(const Mesh& mesh, const T& defaultValue = T())
        : m_mesh(mesh) {
        // Initialize data storage
        m_data.resize(mesh.nx() * mesh.ny(), defaultValue);
    }

    /**
     * @brief Access field value at the specified coordinates
     * 
     * @param i Column index
     * @param j Row index
     * @return T& Reference to the field value
     */
    T& operator()(int i, int j) {
        int idx = m_mesh.linearIndex(i, j);
        assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
        return m_data[idx];
    }

    /**
     * @brief Access field value at the specified coordinates (const)
     * 
     * @param i Column index
     * @param j Row index
     * @return const T& Const reference to the field value
     */
    const T& operator()(int i, int j) const {
        int idx = m_mesh.linearIndex(i, j);
        assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
        return m_data[idx];
    }

    /**
     * @brief Get the size of the field
     * 
     * @return size_t Number of elements in the field
     */
    size_t size() const {
        return m_data.size();
    }

private:
    const Mesh& m_mesh;         ///< Reference to the mesh
    std::vector<T> m_data;      ///< Field data storage
};

} // namespace mesh

#endif // MOCK_FIELD_H