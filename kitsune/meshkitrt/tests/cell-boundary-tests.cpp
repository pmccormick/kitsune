/**
 * @file CellBoundaryTests.cpp
 * @brief Tests for Cell boundary detection
 * 
 * These tests focus on the isBoundary() method and ensuring that
 * boundary cells are correctly identified.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"

namespace mesh {
namespace testing {

// Fixture for Cell boundary tests
class CellBoundaryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for boundary tests
        mesh = new Mesh(5, 5);
    }

    void TearDown() override {
        delete mesh;
    }

    Mesh* mesh;
};

// Test isBoundary for corner cells
TEST_F(CellBoundaryTest, CornerCellsAreBoundaries) {
    // Test all four corners
    Cell bottomLeft = mesh->getCell(0, 0);
    EXPECT_TRUE(bottomLeft.isBoundary());
    
    Cell bottomRight = mesh->getCell(4, 0);
    EXPECT_TRUE(bottomRight.isBoundary());
    
    Cell topLeft = mesh->getCell(0, 4);
    EXPECT_TRUE(topLeft.isBoundary());
    
    Cell topRight = mesh->getCell(4, 4);
    EXPECT_TRUE(topRight.isBoundary());
}

// Test isBoundary for edge cells (not corners)
TEST_F(CellBoundaryTest, EdgeCellsAreBoundaries) {
    // Bottom edge (not corner)
    Cell bottomEdge = mesh->getCell(2, 0);
    EXPECT_TRUE(bottomEdge.isBoundary());
    
    // Left edge (not corner)
    Cell leftEdge = mesh->getCell(0, 2);
    EXPECT_TRUE(leftEdge.isBoundary());
    
    // Right edge (not corner)
    Cell rightEdge = mesh->getCell(4, 2);
    EXPECT_TRUE(rightEdge.isBoundary());
    
    // Top edge (not corner)
    Cell topEdge = mesh->getCell(2, 4);
    EXPECT_TRUE(topEdge.isBoundary());
}

// Test isBoundary for interior cells
TEST_F(CellBoundaryTest, InteriorCellsAreNotBoundaries) {
    // Test various interior cells
    Cell interiorCell1 = mesh->getCell(1, 1);
    EXPECT_FALSE(interiorCell1.isBoundary());
    
    Cell interiorCell2 = mesh->getCell(3, 3);
    EXPECT_FALSE(interiorCell2.isBoundary());
    
    Cell interiorCell3 = mesh->getCell(2, 2);
    EXPECT_FALSE(interiorCell3.isBoundary());
    
    Cell interiorCell4 = mesh->getCell(1, 3);
    EXPECT_FALSE(interiorCell4.isBoundary());
    
    Cell interiorCell5 = mesh->getCell(3, 1);
    EXPECT_FALSE(interiorCell5.isBoundary());
}

// Test isBoundary for invalid cells
TEST_F(CellBoundaryTest, InvalidCellBoundaryCheck) {
    // Create an invalid cell
    Cell invalidCell;
    
    // Boundary check should throw for invalid cells
    EXPECT_THROW(invalidCell.isBoundary(), std::logic_error);
    
    // Also test for a cell with invalid indices but valid mesh
    Cell outOfBoundsCell(mesh, 10, 10);
    EXPECT_THROW(outOfBoundsCell.isBoundary(), std::logic_error);
}

// Test isBoundary for a 1x1 mesh (special case)
TEST_F(CellBoundaryTest, SingleCellMeshBoundary) {
    // Create a 1x1 mesh
    Mesh singleCellMesh(1, 1);
    
    // The only cell is both an interior and boundary cell
    Cell onlyCell = singleCellMesh.getCell(0, 0);
    EXPECT_TRUE(onlyCell.isBoundary());
}

// Test isBoundary for a 2x2 mesh (all cells are boundary cells)
TEST_F(CellBoundaryTest, TwoByTwoMeshBoundary) {
    // Create a 2x2 mesh
    Mesh smallMesh(2, 2);
    
    // All cells should be boundary cells
    Cell cell00 = smallMesh.getCell(0, 0);
    EXPECT_TRUE(cell00.isBoundary());
    
    Cell cell01 = smallMesh.getCell(0, 1);
    EXPECT_TRUE(cell01.isBoundary());
    
    Cell cell10 = smallMesh.getCell(1, 0);
    EXPECT_TRUE(cell10.isBoundary());
    
    Cell cell11 = smallMesh.getCell(1, 1);
    EXPECT_TRUE(cell11.isBoundary());
}

// Test boundary detection in a non-square mesh
TEST_F(CellBoundaryTest, NonSquareMeshBoundary) {
    // Create a 3x5 mesh (rectangular)
    Mesh rectMesh(3, 5);
    
    // Check boundary detection
    
    // Interior cell
    Cell interior = rectMesh.getCell(1, 2);
    EXPECT_FALSE(interior.isBoundary());
    
    // Boundary cells on the long edges
    Cell leftEdge = rectMesh.getCell(0, 2);
    EXPECT_TRUE(leftEdge.isBoundary());
    
    Cell rightEdge = rectMesh.getCell(2, 2);
    EXPECT_TRUE(rightEdge.isBoundary());
    
    // Boundary cells on the short edges
    Cell bottomEdge = rectMesh.getCell(1, 0);
    EXPECT_TRUE(bottomEdge.isBoundary());
    
    Cell topEdge = rectMesh.getCell(1, 4);
    EXPECT_TRUE(topEdge.isBoundary());
}

// Test that all cells in a mesh with dimensions ≤ 2 are boundary cells
TEST_F(CellBoundaryTest, SmallMeshAllBoundary) {
    // Create various small meshes
    Mesh mesh1x2(1, 2);
    Mesh mesh2x1(2, 1);
    Mesh mesh2x2(2, 2);
    
    // Test every cell in the 1x2 mesh
    for (uint32_t i = 0; i < mesh1x2.nx(); i++) {
        for (uint32_t j = 0; j < mesh1x2.ny(); j++) {
            EXPECT_TRUE(mesh1x2.getCell(i, j).isBoundary());
        }
    }
    
    // Test every cell in the 2x1 mesh
    for (uint32_t i = 0; i < mesh2x1.nx(); i++) {
        for (uint32_t j = 0; j < mesh2x1.ny(); j++) {
            EXPECT_TRUE(mesh2x1.getCell(i, j).isBoundary());
        }
    }
    
    // Test every cell in the 2x2 mesh
    for (uint32_t i = 0; i < mesh2x2.nx(); i++) {
        for (uint32_t j = 0; j < mesh2x2.ny(); j++) {
            EXPECT_TRUE(mesh2x2.getCell(i, j).isBoundary());
        }
    }
}

} // namespace testing
} // namespace mesh
