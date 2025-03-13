#ifndef MOCK_MESH_BASE_H
#define MOCK_MESH_BASE_H

#include "Mesh.h"
#include "Cell.h"
#include <vector>
#include <stdexcept>
#include <utility>

namespace mock {

/**
 * @brief Mock implementation of Mesh for testing
 * 
 * This class provides a minimal implementation of Mesh that can be
 * used for testing Cell, Region, and other components that depend on Mesh.
 */
class MockMeshBase : public mesh::Mesh {
public:
    /**
     * @brief Construct a new Mock Mesh Base
     * 
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     */
    MockMeshBase(uint32_t nx, uint32_t ny) : mesh::Mesh(nx, ny) {}
    
    /**
     * @brief Linearized index calculation (row-major order)
     * 
     * @param i Column index
     * @param j Row index
     * @return int Linear index
     */
    int index(int i, int j) const override {
        if (i < 0 || i >= static_cast<int>(nx()) || 
            j < 0 || j >= static_cast<int>(ny())) {
            throw std::out_of_range("Cell indices out of bounds");
        }
        return i + j * nx();
    }
    
    /**
     * @brief Get a cell at specified indices
     * 
     * @param i Column index
     * @param j Row index
     * @return mesh::Cell Cell at the specified location
     */
    mesh::Cell getCell(int i, int j) const {
        if (i < 0 || i >= static_cast<int>(nx()) || 
            j < 0 || j >= static_cast<int>(ny())) {
            return mesh::Cell();  // Invalid cell
        }
        return mesh::Cell(const_cast<MockMeshBase*>(this), i, j);
    }
    
    /**
     * @brief Get a typed cell at specified indices
     * 
     * Generic version for use with different Cell types in tests
     * 
     * @tparam CellT Cell type
     * @param i Column index
     * @param j Row index
     * @return CellT* Pointer to cell (nullptr if invalid)
     */
    template <typename CellT>
    CellT* getTypedCell(int i, int j) const {
        if (i < 0 || i >= static_cast<int>(nx()) || 
            j < 0 || j >= static_cast<int>(ny())) {
            return nullptr;  // Invalid cell
        }
        
        // For real implementations, this would be more sophisticated
        // But for testing, we just need something that behaves like a cell
        static std::vector<CellT> cells;
        cells.emplace_back(const_cast<MockMeshBase*>(this), i, j);
        return &cells.back();
    }
    
    /**
     * @brief Convert linear index to 2D coordinates
     * 
     * @param linearIdx Linear index
     * @return std::pair<int, int> (i,j) coordinates
     */
    std::pair<int, int> toIndices(int linearIdx) const {
        if (linearIdx < 0 || linearIdx >= static_cast<int>(nx() * ny())) {
            throw std::out_of_range("Linear index out of bounds");
        }
        
        int j = linearIdx / nx();
        int i = linearIdx % nx();
        return {i, j};
    }
    
    /**
     * @brief Create a cell range for iteration (stub implementation)
     * 
     * @return mesh::CellRange Range over all cells
     */
    mesh::CellRange cells() const {
        // In real code, this would return a proper CellRange
        // For testing, we'll just need to mock what's used by the tests
        return mesh::CellRange(const_cast<MockMeshBase*>(this), 
                              0, 0, 
                              static_cast<int>(nx()), static_cast<int>(ny()));
    }
};

} // namespace mock

#endif // MOCK_MESH_BASE_H
