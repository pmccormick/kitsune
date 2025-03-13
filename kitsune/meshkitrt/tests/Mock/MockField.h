#ifndef MOCK_FIELD_H
#define MOCK_FIELD_H

#include <vector>
#include <stdexcept>
#include <cassert>

namespace mock {

/**
 * @brief Mock implementation of Field for testing
 * 
 * This template class provides a simplified Field implementation
 * that can be used for testing without depending on the actual Field class.
 * 
 * @tparam T Value type stored in the field
 */
template <typename T>
class MockField {
public:
    /**
     * @brief Construct a new Mock Field
     * 
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     * @param defaultValue Default value for each field element
     */
    MockField(uint32_t nx, uint32_t ny, const T& defaultValue = T())
        : m_nx(nx), m_ny(ny), m_data(nx * ny, defaultValue) {}
    
    /**
     * @brief Get value at (i,j) - non-const version
     * 
     * @param i Column index
     * @param j Row index
     * @return T& Reference to value
     */
    T& operator()(int i, int j) {
        int idx = index(i, j);
        assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
        return m_data[idx];
    }
    
    /**
     * @brief Get value at (i,j) - const version
     * 
     * @param i Column index
     * @param j Row index
     * @return const T& Const reference to value
     */
    const T& operator()(int i, int j) const {
        int idx = index(i, j);
        assert(idx >= 0 && idx < static_cast<int>(m_data.size()));
        return m_data[idx];
    }
    
    /**
     * @brief Get size in x-direction
     * 
     * @return uint32_t Number of cells in x-direction
     */
    uint32_t nx() const { return m_nx; }
    
    /**
     * @brief Get size in y-direction
     * 
     * @return uint32_t Number of cells in y-direction
     */
    uint32_t ny() const { return m_ny; }
    
    /**
     * @brief Get total size of the field
     * 
     * @return size_t Total number of elements
     */
    size_t size() const { return m_data.size(); }
    
    /**
     * @brief Get direct access to raw data
     * 
     * @return T* Pointer to raw data
     */
    T* data() { return m_data.data(); }
    
    /**
     * @brief Get direct access to raw data (const)
     * 
     * @return const T* Const pointer to raw data
     */
    const T* data() const { return m_data.data(); }
    
    /**
     * @brief Fill the entire field with a value
     * 
     * @param value Value to fill with
     */
    void fill(const T& value) {
        std::fill(m_data.begin(), m_data.end(), value);
    }

private:
    uint32_t m_nx;  ///< Number of cells in x-direction
    uint32_t m_ny;  ///< Number of cells in y-direction
    std::vector<T> m_data;  ///< Field data storage
    
    /**
     * @brief Calculate linear index from (i,j) coordinates
     * 
     * @param i Column index
     * @param j Row index
     * @return int Linear index
     */
    int index(int i, int j) const {
        if (i < 0 || i >= static_cast<int>(m_nx) || 
            j < 0 || j >= static_cast<int>(m_ny)) {
            throw std::out_of_range("Field indices out of bounds");
        }
        return i + j * m_nx;
    }
};

} // namespace mock

#endif // MOCK_FIELD_H
