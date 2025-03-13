#ifndef MOCK_CELL_H
#define MOCK_CELL_H

#include "Cell.h"
#include "mock/MockField.h"
#include <utility>

namespace mock {

/**
 * @brief Extended Cell class for testing purposes
 * 
 * This class extends the base Cell class to expose protected
 * methods and add testing-specific functionality.
 */
class MockCell : public mesh::Cell {
public:
    /**
     * @brief Default constructor (creates invalid cell)
     */
    MockCell() : Cell() {}
    
    /**
     * @brief Construct a MockCell with specific coordinates
     * 
     * @param mesh Pointer to mesh
     * @param i Column index
     * @param j Row index
     */
    MockCell(mesh::Mesh* mesh, int i, int j) : Cell(mesh, i, j) {}
    
    /**
     * @brief Expose protected direction offset calculation for testing
     * 
     * @param direction Direction bit flag
     * @return std::pair<int, int> (di, dj) offset
     */
    static std::pair<int, int> testGetDirectionOffset(uint8_t direction) {
        return getDirectionOffset(direction);
    }
    
    /**
     * @brief Access a field value using this cell's position
     * 
     * @tparam T Value type in the field
     * @param field Field to access
     * @return T Field value at this cell's position
     */
    template <typename T>
    T getFieldValue(const mock::MockField<T>& field) const {
        return field(i(), j());
    }
    
    /**
     * @brief Set a field value at this cell's position
     * 
     * @tparam T Value type in the field
     * @param field Field to modify
     * @param value Value to set
     */
    template <typename T>
    void setFieldValue(mock::MockField<T>& field, const T& value) const {
        field(i(), j()) = value;
    }
    
    /**
     * @brief Get a flag indicating specific testing conditions
     * 
     * This is a placeholder for testing-specific features that
     * don't match the actual Cell API but are useful in tests.
     * 
     * @param condition Condition identifier
     * @return bool Flag value
     */
    bool testCondition(int condition) const {
        // Example implementation - this would be customized per test
        switch (condition) {
            case 1: return i() % 2 == 0;  // Even column
            case 2: return j() % 2 == 0;  // Even row
            case 3: return i() == j();    // Diagonal
            default: return false;
        }
    }
};

} // namespace mock

#endif // MOCK_CELL_H
