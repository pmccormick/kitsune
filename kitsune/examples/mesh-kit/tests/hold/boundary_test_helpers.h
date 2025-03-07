#pragma once

#include "BoundaryClass.h"
#include "Cell.h"
#include "Grid.h"
#include "Material.h"
#include <gmock/gmock.h>
#include <memory>
#include <vector>

// Forward declarations for any classes that might be needed
class Mesh;

/**
 * @class MockGrid
 * @brief Mock Grid class for testing boundary conditions
 *
 * This class mocks the necessary methods from the Grid/Mesh class
 * that are needed by boundary conditions during testing.
 */
class MockGrid {
public:
  MOCK_METHOD(double, getDx, (), (const));
  MOCK_METHOD(double, getDy, (), (const));
  MOCK_METHOD(Cell &, getCell, (size_t, size_t), ());
  MOCK_METHOD(const Cell &, getCell, (size_t, size_t), (const));

  // Add any other methods from Grid that boundaries might call
  MOCK_METHOD(double, physicalX, (size_t), (const));
  MOCK_METHOD(double, physicalY, (size_t), (const));
};

/**
 * @class MockCell
 * @brief Mock Cell class for testing boundary interactions
 *
 * This class provides a way to test boundary condition behavior
 * without requiring a full Cell implementation.
 */
class MockCell : public Cell {
public:
  MockCell() : Cell() {}
  MockCell(Cell::CellType type) : Cell(type) {}

  MOCK_METHOD(MockGrid *, getGrid, (), (const));
  MOCK_METHOD(void, setBoundaryCondition, (std::shared_ptr<BoundaryClass>));
  MOCK_METHOD(bool, hasBoundaryCondition, (), (const));
  MOCK_METHOD(std::shared_ptr<BoundaryClass>, getBoundaryCondition, (),
              (const));
};

/**
 * @brief Creates a test grid of cells for boundary testing
 *
 * @param nx Number of cells in x-direction
 * @param ny Number of cells in y-direction
 * @return 2D vector of cells with boundaries at the edges
 */
inline std::vector<std::vector<Cell>> createTestGrid(size_t nx, size_t ny) {
  std::vector<std::vector<Cell>> grid(ny, std::vector<Cell>(nx));

  // Set up boundary cells at the edges
  for (size_t j = 0; j < ny; j++) {
    for (size_t i = 0; i < nx; i++) {
      if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1) {
        grid[j][i].setType(Cell::CellType::BOUNDARY);
        grid[j][i].setBoundary(true);
      } else {
        grid[j][i].setType(Cell::CellType::FLUID);
      }
    }
  }

  return grid;
}

/**
 * @brief Creates a list of non-boundary neighbor cells for a boundary cell
 *
 * @param boundaryCell Reference to the boundary cell
 * @param grid The grid of cells
 * @param i x-index of the boundary cell
 * @param j y-index of the boundary cell
 * @param nx Total cells in x-direction
 * @param ny Total cells in y-direction
 * @return Vector of pointers to neighboring cells
 */
inline std::vector<Cell *>
createNeighborList(Cell &boundaryCell, std::vector<std::vector<Cell>> &grid,
                   size_t i, size_t j, size_t nx, size_t ny) {
  std::vector<Cell *> neighbors;

  // Add neighbor cells that are within bounds and are not boundaries
  if (i > 0 && !grid[j][i - 1].isBoundary())
    neighbors.push_back(&grid[j][i - 1]);
  if (i < nx - 1 && !grid[j][i + 1].isBoundary())
    neighbors.push_back(&grid[j][i + 1]);
  if (j > 0 && !grid[j - 1][i].isBoundary())
    neighbors.push_back(&grid[j - 1][i]);
  if (j < ny - 1 && !grid[j + 1][i].isBoundary())
    neighbors.push_back(&grid[j + 1][i]);

  return neighbors;
}

/**
 * @brief Helper to simulate boundary application on a grid
 *
 * @param boundary The boundary condition to apply
 * @param grid The grid of cells
 * @param nx Total cells in x-direction
 * @param ny Total cells in y-direction
 * @param dt Time step size
 */
inline void applyBoundaryToGrid(std::shared_ptr<BoundaryClass> boundary,
                                std::vector<std::vector<Cell>> &grid, size_t nx,
                                size_t ny, double dt = 0.0) {
  // Apply boundary condition to all boundary cells
  for (size_t j = 0; j < ny; j++) {
    for (size_t i = 0; i < nx; i++) {
      if (grid[j][i].isBoundary()) {
        auto neighbors = createNeighborList(grid[j][i], grid, i, j, nx, ny);
        double physX =
            static_cast<double>(i); // Use index as position for testing
        double physY = static_cast<double>(j);
        boundary->apply(grid[j][i], physX, physY, dt, &neighbors);
      }
    }
  }
}

/**
 * @brief Helper to set up boundary cells for a specific boundary type
 *
 * @param boundaryType The type of boundary ("Dirichlet", "Neumann", etc.)
 * @param boundaryName Optional name for the boundary
 * @return Shared pointer to the created boundary condition
 */
inline std::shared_ptr<BoundaryClass>
createBoundaryCondition(const std::string &boundaryType,
                        const std::string &boundaryName = "Test Boundary") {

  // Create boundary through factory - make sure
  // BoundaryFactory::registerBoundaryTypes() has been called before using this
  // function
  return BoundaryFactory::createBoundary(boundaryType, boundaryName);
}

/**
 * @brief Helper to create a simple test environment for a boundary with
 * materials
 *
 * @param boundaryMaterial Material for the boundary cell
 * @param fluidMaterial Material for the fluid cell
 * @return Pair of boundary and fluid cell pointers set up for testing
 */
inline std::pair<std::unique_ptr<Cell>, std::unique_ptr<Cell>>
createBoundaryTestCells(std::shared_ptr<Material> boundaryMaterial,
                        std::shared_ptr<Material> fluidMaterial) {

  auto boundaryCell = std::make_unique<Cell>(Cell::CellType::BOUNDARY);
  boundaryCell->setBoundary(true);
  boundaryCell->setMaterial(boundaryMaterial);

  auto fluidCell = std::make_unique<Cell>(Cell::CellType::FLUID);
  fluidCell->setMaterial(fluidMaterial);

  return {std::move(boundaryCell), std::move(fluidCell)};
}

/**
 * @class SimpleMockGrid
 * @brief Provides a simplified mock grid for boundary condition testing
 *
 * This class implements just enough of the Grid interface to support
 * testing NoSlip boundary conditions with heat flux.
 */
class SimpleMockGrid {
public:
  SimpleMockGrid(double dx = 0.01, double dy = 0.01) : m_dx(dx), m_dy(dy) {}

  double getDx() const { return m_dx; }
  double getDy() const { return m_dy; }

private:
  double m_dx;
  double m_dy;
};

/**
 * @brief Patch to allow a Cell to access a mock grid for testing
 *
 * This function patches a cell to return a mock grid when getGrid() is called.
 * It's specifically designed for testing boundary conditions that need grid
 * info.
 *
 * @param cell The cell to patch
 * @param grid Pointer to the mock grid
 */
inline void patchCellWithMockGrid(Cell &cell, SimpleMockGrid *grid) {
  // Ideally, we'd use a proper mock object, but this is a simplified approach
  // Store the grid pointer in a dynamic property
  cell.setDynamicProperty("mock_grid_dx", grid->getDx());
  cell.setDynamicProperty("mock_grid_dy", grid->getDy());
}

/**
 * @brief Extension to Cell to support getGrid for testing
 *
 * This would normally be a mock method, but for simplicity we implement
 * it as a helper function that uses dynamic properties set by
 * patchCellWithMockGrid
 *
 * @param cell The cell to get grid info from
 * @return A pointer to an object that behaves like Grid
 */
inline SimpleMockGrid *getGridForCell(const Cell &cell) {
  static SimpleMockGrid mockGrid(cell.getDynamicProperty("mock_grid_dx", 0.01),
                                 cell.getDynamicProperty("mock_grid_dy", 0.01));
  return &mockGrid;
}
