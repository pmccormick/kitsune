/**
 * @file CellBaseTest.cpp
 * @brief Unit tests for the CellBase class using Google Test framework
 * 
 * These tests verify the functionality of the CellBase class,
 * including direction handling, neighbor access, and location indices.
 */

#include "CellBase.h"
#include "MeshBase.h"
#include <gtest/gtest.h>
#include <memory>

namespace {

// Simple test cell implementation for testing
class TestCell : public CellBase {
public:
    TestCell(int i, int j, MeshBase* mesh) : CellBase(i, j, mesh) {}
    TestCell() : CellBase(0, 0, nullptr) {} // Default constructor for vector resizing
};

// Simple test mesh implementation for testing CellBase
class TestMesh : public MeshBase {
public:
    TestMesh(int nx, int ny) : MeshBase(nx, ny, 1.0, 1.0) {
        // Create storage for cells
        m_cells.resize(nx * ny);
        
        // Initialize cells
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                m_cells[linearIndex(i, j)] = TestCell(i, j, this);
            }
        }
    }
    
    CellBase* getCell(int i, int j) override {
        if (i >= 0 && i < nx() && j >= 0 && j < ny()) {
            return &m_cells[linearIndex(i, j)];
        }
        return nullptr;
    }

private:
    std::vector<TestCell> m_cells;
};

// Test fixture for CellBase tests
class CellBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for all tests
        mesh = std::make_unique<TestMesh>(5, 5);
    }
    
    std::unique_ptr<TestMesh> mesh;
};

// Test constructing a cell and getting its indices
TEST_F(CellBaseTest, ConstructAndGetIndices) {
    CellBase* cell = mesh->getCell(2, 3);
    
    ASSERT_NE(nullptr, cell);
    EXPECT_EQ(2, cell->i());
    EXPECT_EQ(3, cell->j());
    EXPECT_EQ(mesh.get(), cell->mesh());
    EXPECT_EQ(mesh->linearIndex(2, 3), cell->linearIndex());
}

// Test getting neighboring cells
TEST_F(CellBaseTest, NeighborAccess) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    // Test all four primary directions
    CellBase* north = cell->neighbor(CellBase::NORTH);
    ASSERT_NE(nullptr, north);
    EXPECT_EQ(2, north->i());
    EXPECT_EQ(3, north->j());
    
    CellBase* east = cell->neighbor(CellBase::EAST);
    ASSERT_NE(nullptr, east);
    EXPECT_EQ(3, east->i());
    EXPECT_EQ(2, east->j());
    
    CellBase* south = cell->neighbor(CellBase::SOUTH);
    ASSERT_NE(nullptr, south);
    EXPECT_EQ(2, south->i());
    EXPECT_EQ(1, south->j());
    
    CellBase* west = cell->neighbor(CellBase::WEST);
    ASSERT_NE(nullptr, west);
    EXPECT_EQ(1, west->i());
    EXPECT_EQ(2, west->j());
    
    // Test diagonal directions
    CellBase* northeast = cell->neighbor(CellBase::NORTHEAST);
    ASSERT_NE(nullptr, northeast);
    EXPECT_EQ(3, northeast->i());
    EXPECT_EQ(3, northeast->j());
    
    CellBase* southeast = cell->neighbor(CellBase::SOUTHEAST);
    ASSERT_NE(nullptr, southeast);
    EXPECT_EQ(3, southeast->i());
    EXPECT_EQ(1, southeast->j());
}

// Test boundary detection
TEST_F(CellBaseTest, BoundaryDetection) {
    // Interior cell should not be a boundary
    CellBase* interior = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, interior);
    EXPECT_FALSE(interior->isBoundary());
    
    // Boundary cells should report as boundaries
    CellBase* left = mesh->getCell(0, 2);
    ASSERT_NE(nullptr, left);
    EXPECT_TRUE(left->isBoundary());
    
    CellBase* right = mesh->getCell(4, 2);
    ASSERT_NE(nullptr, right);
    EXPECT_TRUE(right->isBoundary());
    
    CellBase* bottom = mesh->getCell(2, 0);
    ASSERT_NE(nullptr, bottom);
    EXPECT_TRUE(bottom->isBoundary());
    
    CellBase* top = mesh->getCell(2, 4);
    ASSERT_NE(nullptr, top);
    EXPECT_TRUE(top->isBoundary());
}

// Test neighbor index calculation
TEST_F(CellBaseTest, NeighborIndices) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    // Test primary directions
    auto [ni, nj] = cell->neighborIndices(CellBase::NORTH);
    EXPECT_EQ(2, ni);
    EXPECT_EQ(3, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::EAST);
    EXPECT_EQ(3, ni);
    EXPECT_EQ(2, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::SOUTH);
    EXPECT_EQ(2, ni);
    EXPECT_EQ(1, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::WEST);
    EXPECT_EQ(1, ni);
    EXPECT_EQ(2, nj);
    
    // Test diagonal directions
    std::tie(ni, nj) = cell->neighborIndices(CellBase::NORTHEAST);
    EXPECT_EQ(3, ni);
    EXPECT_EQ(3, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::SOUTHEAST);
    EXPECT_EQ(3, ni);
    EXPECT_EQ(1, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::SOUTHWEST);
    EXPECT_EQ(1, ni);
    EXPECT_EQ(1, nj);
    
    std::tie(ni, nj) = cell->neighborIndices(CellBase::NORTHWEST);
    EXPECT_EQ(1, ni);
    EXPECT_EQ(3, nj);
}

// Test location indices for different field locations
TEST_F(CellBaseTest, LocationIndices) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    // Cell center
    auto [i, j] = cell->locationIndices(CellBase::CELL_CENTER);
    EXPECT_EQ(2, i);
    EXPECT_EQ(2, j);
    
    // Vertices
    std::tie(i, j) = cell->locationIndices(CellBase::CELL_VERTEX, CellBase::NONE);
    EXPECT_EQ(2, i);
    EXPECT_EQ(2, j);
    
    std::tie(i, j) = cell->locationIndices(CellBase::CELL_VERTEX, CellBase::NORTH);
    EXPECT_EQ(2, i);
    EXPECT_EQ(3, j);
    
    std::tie(i, j) = cell->locationIndices(CellBase::CELL_VERTEX, CellBase::EAST);
    EXPECT_EQ(3, i);
    EXPECT_EQ(2, j);
    
    std::tie(i, j) = cell->locationIndices(CellBase::CELL_VERTEX, CellBase::NORTHEAST);
    EXPECT_EQ(3, i);
    EXPECT_EQ(3, j);
    
    // Horizontal edges
    std::tie(i, j) = cell->locationIndices(CellBase::HORIZONTAL_EDGE, CellBase::NONE);
    EXPECT_EQ(2, i);
    EXPECT_EQ(2, j);
    
    std::tie(i, j) = cell->locationIndices(CellBase::HORIZONTAL_EDGE, CellBase::NORTH);
    EXPECT_EQ(2, i);
    EXPECT_EQ(3, j);
    
    // Vertical edges
    std::tie(i, j) = cell->locationIndices(CellBase::VERTICAL_EDGE, CellBase::NONE);
    EXPECT_EQ(2, i);
    EXPECT_EQ(2, j);
    
    std::tie(i, j) = cell->locationIndices(CellBase::VERTICAL_EDGE, CellBase::EAST);
    EXPECT_EQ(3, i);
    EXPECT_EQ(2, j);
}

// Test getting all neighbors
TEST_F(CellBaseTest, AllNeighbors) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    auto neighbors = cell->neighbors();
    ASSERT_EQ(4, neighbors.size());
    
    // Check all neighbors
    EXPECT_EQ(2, neighbors[CellBase::NORTH_IDX]->i());
    EXPECT_EQ(3, neighbors[CellBase::NORTH_IDX]->j());
    
    EXPECT_EQ(3, neighbors[CellBase::EAST_IDX]->i());
    EXPECT_EQ(2, neighbors[CellBase::EAST_IDX]->j());
    
    EXPECT_EQ(2, neighbors[CellBase::SOUTH_IDX]->i());
    EXPECT_EQ(1, neighbors[CellBase::SOUTH_IDX]->j());
    
    EXPECT_EQ(1, neighbors[CellBase::WEST_IDX]->i());
    EXPECT_EQ(2, neighbors[CellBase::WEST_IDX]->j());
}

// Test getting neighbors at a boundary
TEST_F(CellBaseTest, BoundaryNeighbors) {
    CellBase* cell = mesh->getCell(0, 0);
    ASSERT_NE(nullptr, cell);
    
    // Neighbor in the WEST direction should be nullptr
    EXPECT_EQ(nullptr, cell->neighbor(CellBase::WEST));
    
    // Neighbor in the SOUTH direction should be nullptr
    EXPECT_EQ(nullptr, cell->neighbor(CellBase::SOUTH));
    
    // Neighbor in the SOUTHWEST direction should be nullptr
    EXPECT_EQ(nullptr, cell->neighbor(CellBase::SOUTHWEST));
    
    // Neighbor in the EAST direction should be valid
    EXPECT_NE(nullptr, cell->neighbor(CellBase::EAST));
    
    // Neighbor in the NORTH direction should be valid
    EXPECT_NE(nullptr, cell->neighbor(CellBase::NORTH));
    
    // Neighbor in the NORTHEAST direction should be valid
    EXPECT_NE(nullptr, cell->neighbor(CellBase::NORTHEAST));
}

// Test traditional direction index access
TEST_F(CellBaseTest, DirectionIndexAccess) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    // Test using traditional direction indices (for backward compatibility)
    CellBase* north = cell->neighborByIndex(CellBase::NORTH_IDX);
    ASSERT_NE(nullptr, north);
    EXPECT_EQ(2, north->i());
    EXPECT_EQ(3, north->j());
    
    CellBase* east = cell->neighborByIndex(CellBase::EAST_IDX);
    ASSERT_NE(nullptr, east);
    EXPECT_EQ(3, east->i());
    EXPECT_EQ(2, east->j());
    
    CellBase* south = cell->neighborByIndex(CellBase::SOUTH_IDX);
    ASSERT_NE(nullptr, south);
    EXPECT_EQ(2, south->i());
    EXPECT_EQ(1, south->j());
    
    CellBase* west = cell->neighborByIndex(CellBase::WEST_IDX);
    ASSERT_NE(nullptr, west);
    EXPECT_EQ(1, west->i());
    EXPECT_EQ(2, west->j());
}

// Test getting neighbors with bit flags
TEST_F(CellBaseTest, GetNeighborsWithFlags) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    // Get neighbors in NORTH and EAST directions
    auto neighbors = cell->getNeighbors(CellBase::NORTH | CellBase::EAST);
    ASSERT_EQ(2, neighbors.size());
    
    // Verify the neighbors
    bool foundNorth = false;
    bool foundEast = false;
    
    for (auto* neighbor : neighbors) {
        if (neighbor->i() == 2 && neighbor->j() == 3) {
            foundNorth = true;
        } else if (neighbor->i() == 3 && neighbor->j() == 2) {
            foundEast = true;
        }
    }
    
    EXPECT_TRUE(foundNorth);
    EXPECT_TRUE(foundEast);
}

// Test physical position calculation
TEST_F(CellBaseTest, PhysicalPosition) {
    CellBase* cell = mesh->getCell(2, 2);
    ASSERT_NE(nullptr, cell);
    
    auto [x, y] = cell->position();
    EXPECT_DOUBLE_EQ(2.0, x);
    EXPECT_DOUBLE_EQ(2.0, y);
}

}  // namespace

