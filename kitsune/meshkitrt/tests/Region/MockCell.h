/**
 * @file MockCell.h
 * @brief Mock cell implementation for testing
 */

#ifndef MOCK_CELL_H
#define MOCK_CELL_H

#include "Cell.h"

namespace mesh {

/**
 * @class MockCell
 * @brief Mock cell implementation for testing
 */
class MockCell : public Cell {
public:
    /**
     * @brief Construct a new Mock Cell
     * 
     * @param mesh Pointer to the mesh
     * @param i Column index
     * @param j Row index
     * @param group Group identifier
     */
    MockCell(Mesh* mesh, int i, int j, int group = 0)
        : Cell(mesh, i, j), group(group), m_accessed(false) {}

    /**
     * @brief Mark the cell as accessed for testing
     */
    void markAccessed() { m_accessed = true; }
    
    /**
     * @brief Check if the cell was accessed
     * 
     * @return true if the cell was accessed
     */
    bool wasAccessed() const { return m_accessed; }
    
    /**
     * @brief Reset access tracking
     */
    void resetAccess() { m_accessed = false; }

    // Group identifier for testing
    int group;

private:
    bool m_accessed; // Track if the cell was accessed
};

} // namespace mesh

#endif // MOCK_CELL_H

/**
 * @file MockMesh.h
 * @brief Mock mesh implementation for testing
 */

#ifndef MOCK_MESH_H
#define MOCK_MESH_H

#include "Mesh.h"
#include "MockCell.h"
#include <vector>
#include <functional>

namespace mesh {

/**
 * @class MockMesh
 * @brief Mock mesh implementation for testing
 */
class MockMesh : public Mesh {
public:
    /**
     * @brief Construct a new Mock Mesh
     * 
     * @param nx Number of cells in x-direction
     * @param ny Number of cells in y-direction
     */
    MockMesh(uint32_t nx, uint32_t ny) : Mesh(nx, ny) {
        // Create cells
        m_cells.resize(ny);
        for (uint32_t j = 0; j < ny; ++j) {
            m_cells[j].resize(nx);
            for (uint32_t i = 0; i < nx; ++i) {
                m_cells[j][i] = std::make_unique<MockCell>(this, static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    /**
     * @brief Initialize cell groups using a function
     * 
     * @param groupFunc Function that determines the group for each cell
     */
    void initializeGroups(std::function<int(int,int)> groupFunc) {
        for (uint32_t j = 0; j < m_cells.size(); ++j) {
            for (uint32_t i = 0; i < m_cells[j].size(); ++i) {
                m_cells[j][i]->group = groupFunc(static_cast<int>(i), static_cast<int>(j));
            }
        }
    }

    /**
     * @brief Get a typed cell at the specified coordinates
     * 
     * @param i Column index
     * @param j Row index
     * @return MockCell* Pointer to the cell, or nullptr if out of bounds
     */
    MockCell* getTypedCell(int i, int j) {
        if (i < 0 || static_cast<uint32_t>(i) >= nx() || 
            j < 0 || static_cast<uint32_t>(j) >= ny()) {
            return nullptr;
        }
        return m_cells[j][i].get();
    }

    /**
     * @brief Get a cell at the specified coordinates
     * 
     * This overrides the base class method to return our typed cells.
     * 
     * @param i Column index
     * @param j Row index
     * @return Cell* Pointer to the cell, or nullptr if out of bounds
     */
    Cell* getCell(uint32_t i, uint32_t j) const override {
        if (i >= nx() || j >= ny()) {
            return nullptr;
        }
        return m_cells[j][i].get();
    }

    /**
     * @brief Get cells by group
     * 
     * @param groupId Group identifier
     * @return std::vector<MockCell*> Vector of cells in the group
     */
    std::vector<MockCell*> getCellsByGroup(int groupId) {
        std::vector<MockCell*> result;
        for (uint32_t j = 0; j < m_cells.size(); ++j) {
            for (uint32_t i = 0; i < m_cells[j].size(); ++i) {
                if (m_cells[j][i]->group == groupId) {
                    result.push_back(m_cells[j][i].get());
                }
            }
        }
        return result;
    }

    /**
     * @brief Reset access tracking for all cells
     */
    void resetAllCellAccess() {
        for (auto& row : m_cells) {
            for (auto& cell : row) {
                cell->resetAccess();
            }
        }
    }

    /**
     * @brief Count cells that were accessed
     * 
     * @return size_t Number of accessed cells
     */
    size_t countAccessedCells() {
        size_t count = 0;
        for (auto& row : m_cells) {
            for (auto& cell : row) {
                if (cell->wasAccessed()) {
                    count++;
                }
            }
        }
        return count;
    }

private:
    std::vector<std::vector<std::unique_ptr<MockCell>>> m_cells;
};

} // namespace mesh

#endif // MOCK_MESH_H