/**
 * @file MeshBaseTest.cpp
 * @brief Unit tests for the MeshBase class using Google Test framework
 */

#include "MeshBase.h"
#include "CellBase.h"
#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

namespace {

// Simple test cell implementation
class TestCell : public CellBase {
public:
    TestCell(int i, int j, MeshBase* mesh) : CellBase(i, j, mesh) {}
    TestCell() : CellBase(0, 0, nullptr) {} // Default constructor for vector resizing
};

// Simple test mesh implementation
class TestMesh : public MeshBase {
public:
    TestMesh(int nx, int ny, double dx = 1.0, double dy = 1.0)
        : MeshBase(nx, ny, dx, dy) {
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

// Test fixture for MeshBase tests
class MeshBaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for all tests
        mesh = std::make_unique<TestMesh>(5, 5, 1.0, 1.0);
    }
    
    std::unique_ptr<TestMesh> mesh;
};

// Test mesh construction and basic properties
TEST_F(MeshBaseTest, Construction) {
    EXPECT_EQ(5, mesh->nx());
    EXPECT_EQ(5, mesh->ny());
    EXPECT_DOUBLE_EQ(1.0, mesh->dx());
    EXPECT_DOUBLE_EQ(1.0, mesh->dy());
}

// Test construction with invalid parameters
TEST_F(MeshBaseTest, InvalidConstruction) {
    EXPECT_THROW(TestMesh(-1, 5), std::invalid_argument);
    EXPECT_THROW(TestMesh(5, -1), std::invalid_argument);
    EXPECT_THROW(TestMesh(5, 5, -1.0, 1.0), std::invalid_argument);
    EXPECT_THROW(TestMesh(5, 5, 1.0, -1.0), std::invalid_argument);
}

// Test index conversion between linear and 2D indices
TEST_F(MeshBaseTest, IndexConversion) {
    // Test linearIndex
    EXPECT_EQ(0, mesh->linearIndex(0, 0));
    EXPECT_EQ(4, mesh->linearIndex(4, 0));
    EXPECT_EQ(5, mesh->linearIndex(0, 1));
    EXPECT_EQ(24, mesh->linearIndex(4, 4));
    
    // Test toIndices
    auto [i0, j0] = mesh->toIndices(0);
    EXPECT_EQ(0, i0);
    EXPECT_EQ(0, j0);
    
    auto [i4, j0_] = mesh->toIndices(4);
    EXPECT_EQ(4, i4);
    EXPECT_EQ(0, j0_);
    
    auto [i0_, j1] = mesh->toIndices(5);
    EXPECT_EQ(0, i0_);
    EXPECT_EQ(1, j1);
    
    auto [i4_, j4] = mesh->toIndices(24);
    EXPECT_EQ(4, i4_);
    EXPECT_EQ(4, j4);
}

// Test physical position calculation
TEST_F(MeshBaseTest, PhysicalPosition) {
    // Test position at origin
    auto [x0, y0] = mesh->position(0, 0);
    EXPECT_DOUBLE_EQ(0.0, x0);
    EXPECT_DOUBLE_EQ(0.0, y0);
    
    // Test position at middle
    auto [x2, y2] = mesh->position(2, 2);
    EXPECT_DOUBLE_EQ(2.0, x2);
    EXPECT_DOUBLE_EQ(2.0, y2);
    
    // Test position at far corner
    auto [x4, y4] = mesh->position(4, 4);
    EXPECT_DOUBLE_EQ(4.0, x4);
    EXPECT_DOUBLE_EQ(4.0, y4);
    
    // Test with non-unit grid spacing
    TestMesh nonUnitMesh(3, 3, 0.5, 2.0);
    auto [x1, y1] = nonUnitMesh.position(1, 1);
    EXPECT_DOUBLE_EQ(0.5, x1);
    EXPECT_DOUBLE_EQ(2.0, y1);
}

// Test cell access
TEST_F(MeshBaseTest, CellAccess) {
    // Test valid cell access
    CellBase* cell = mesh->getCell(2, 3);
    ASSERT_NE(nullptr, cell);
    EXPECT_EQ(2, cell->i());
    EXPECT_EQ(3, cell->j());
    
    // Test out-of-bounds cell access
    EXPECT_EQ(nullptr, mesh->getCell(-1, 0));
    EXPECT_EQ(nullptr, mesh->getCell(0, -1));
    EXPECT_EQ(nullptr, mesh->getCell(5, 0));
    EXPECT_EQ(nullptr, mesh->getCell(0, 5));
}

} // namespace


