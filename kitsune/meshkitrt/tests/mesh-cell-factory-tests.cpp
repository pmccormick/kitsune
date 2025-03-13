/**
 * @file MeshCellFactoryTests.cpp
 * @brief Tests for Mesh as a cell factory
 * 
 * These tests focus on the Mesh class's role as a factory for Cell objects,
 * testing the creation, validation, and management of cells.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include <vector>
#include <stdexcept>

namespace mesh {
namespace testing {

// Test suite for Mesh as a Cell factory
class MeshCellFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create meshes of different sizes for testing
        standardMesh = new Mesh(5, 5);
        largeMesh = new Mesh(100, 100);
        singleCellMesh = new Mesh(1, 1);
        rectangularMesh = new Mesh(3, 7);
    }

    void TearDown() override {
        delete standardMesh;
        delete largeMesh;
        delete singleCellMesh;
        delete rectangularMesh;
    }

    Mesh* standardMesh;
    Mesh* largeMesh;
    Mesh* singleCellMesh;
    Mesh* rectangularMesh;
};

// Test that getCell produces valid cells for valid indices
TEST_F(MeshCellFactoryTest, GetCellWithValidIndices) {
    // Test with standard mesh
    Cell origin = standardMesh->getCell(0, 0);
    EXPECT_TRUE(origin.isValid());
    EXPECT_EQ(origin.i(), 0);
    EXPECT_EQ(origin.j(), 0);
    EXPECT_EQ(origin.mesh(), standardMesh);
    
    Cell interior = standardMesh->getCell(2, 2);
    EXPECT_TRUE(interior.isValid());
    EXPECT_EQ(interior.i(), 2);
    EXPECT_EQ(interior.j(), 2);
    EXPECT_EQ(interior.mesh(), standardMesh);
    
    Cell corner = standardMesh->getCell(4, 4);
    EXPECT_TRUE(corner.isValid());
    EXPECT_EQ(corner.i(), 4);
    EXPECT_EQ(corner.j(), 4);
    EXPECT_EQ(corner.mesh(), standardMesh);
    
    // Test with rectangular mesh
    Cell rectOrigin = rectangularMesh->getCell(0, 0);
    EXPECT_TRUE(rectOrigin.isValid());
    EXPECT_EQ(rectOrigin.i(), 0);
    EXPECT_EQ(rectOrigin.j(), 0);
    EXPECT_EQ(rectOrigin.mesh(), rectangularMesh);
    
    Cell rectCorner = rectangularMesh->getCell(2, 6);
    EXPECT_TRUE(rectCorner.isValid());
    EXPECT_EQ(rectCorner.i(), 2);
    EXPECT_EQ(rectCorner.j(), 6);
    EXPECT_EQ(rectCorner.mesh(), rectangularMesh);
    
    // Test with single cell mesh
    Cell singleCell = singleCellMesh->getCell(0, 0);
    EXPECT_TRUE(singleCell.isValid());
    EXPECT_EQ(singleCell.i(), 0);
    EXPECT_EQ(singleCell.j(), 0);
    EXPECT_EQ(singleCell.mesh(), singleCellMesh);
}

// Test that getCell behavior with out-of-bounds indices
TEST_F(MeshCellFactoryTest, GetCellWithInvalidIndices) {
    // The getCell method itself doesn't validate indices in the current implementation
    // But the resulting cells should be invalid when checked
    
    // Test with standard mesh
    Cell outOfBoundsX = standardMesh->getCell(5, 2);
    EXPECT_FALSE(outOfBoundsX.isValid());
    
    Cell outOfBoundsY = standardMesh->getCell(2, 5);
    EXPECT_FALSE(outOfBoundsY.isValid());
    
    Cell outOfBoundsBoth = standardMesh->getCell(5, 5);
    EXPECT_FALSE(outOfBoundsBoth.isValid());
    
    // Test with far out-of-bounds indices
    Cell farOutOfBounds = standardMesh->getCell(1000, 1000);
    EXPECT_FALSE(farOutOfBounds.isValid());
    
    // Test with single cell mesh
    Cell singleCellOutOfBounds = singleCellMesh->getCell(1, 0);
    EXPECT_FALSE(singleCellOutOfBounds.isValid());
}

// Test cell creation consistency across the entire mesh
TEST_F(MeshCellFactoryTest, CellCreationConsistencyAcrossMesh) {
    // Every valid cell in the mesh should be properly created
    for (uint32_t j = 0; j < standardMesh->ny(); j++) {
        for (uint32_t i = 0; i < standardMesh->nx(); i++) {
            Cell cell = standardMesh->getCell(i, j);
            
            EXPECT_TRUE(cell.isValid()) << "Cell at (" << i << "," << j << ") should be valid";
            EXPECT_EQ(cell.i(), i) << "Cell i-index mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(cell.j(), j) << "Cell j-index mismatch at (" << i << "," << j << ")";
            EXPECT_EQ(cell.mesh(), standardMesh) << "Cell mesh pointer mismatch at (" << i << "," << j << ")";
        }
    }
}

// Test creating cells with a large mesh
TEST_F(MeshCellFactoryTest, GetCellWithLargeMesh) {
    // Test key positions in a large mesh
    Cell origin = largeMesh->getCell(0, 0);
    EXPECT_TRUE(origin.isValid());
    EXPECT_EQ(origin.i(), 0);
    EXPECT_EQ(origin.j(), 0);
    
    Cell interior = largeMesh->getCell(50, 50);
    EXPECT_TRUE(interior.isValid());
    EXPECT_EQ(interior.i(), 50);
    EXPECT_EQ(interior.j(), 50);
    
    Cell corner = largeMesh->getCell(99, 99);
    EXPECT_TRUE(corner.isValid());
    EXPECT_EQ(corner.i(), 99);
    EXPECT_EQ(corner.j(), 99);
    
    // Test out-of-bounds
    Cell outOfBounds = largeMesh->getCell(100, 100);
    EXPECT_FALSE(outOfBounds.isValid());
}

// Test creating cells at mesh boundaries
TEST_F(MeshCellFactoryTest, GetCellAtBoundaries) {
    // Test cells at all four edges and corners
    
    // Corners
    Cell bottomLeft = standardMesh->getCell(0, 0);
    EXPECT_TRUE(bottomLeft.isValid());
    EXPECT_TRUE(bottomLeft.isBoundary());
    
    Cell bottomRight = standardMesh->getCell(4, 0);
    EXPECT_TRUE(bottomRight.isValid());
    EXPECT_TRUE(bottomRight.isBoundary());
    
    Cell topLeft = standardMesh->getCell(0, 4);
    EXPECT_TRUE(topLeft.isValid());
    EXPECT_TRUE(topLeft.isBoundary());
    
    Cell topRight = standardMesh->getCell(4, 4);
    EXPECT_TRUE(topRight.isValid());
    EXPECT_TRUE(topRight.isBoundary());
    
    // Edges (not corners)
    Cell leftEdge = standardMesh->getCell(0, 2);
    EXPECT_TRUE(leftEdge.isValid());
    EXPECT_TRUE(leftEdge.isBoundary());
    
    Cell rightEdge = standardMesh->getCell(4, 2);
    EXPECT_TRUE(rightEdge.isValid());
    EXPECT_TRUE(rightEdge.isBoundary());
    
    Cell bottomEdge = standardMesh->getCell(2, 0);
    EXPECT_TRUE(bottomEdge.isValid());
    EXPECT_TRUE(bottomEdge.isBoundary());
    
    Cell topEdge = standardMesh->getCell(2, 4);
    EXPECT_TRUE(topEdge.isValid());
    EXPECT_TRUE(topEdge.isBoundary());
}

// Test that creating cells from different meshes produces distinct cells
TEST_F(MeshCellFactoryTest, CellCreationFromDifferentMeshes) {
    // Create identical position cells from different meshes
    Cell cell1 = standardMesh->getCell(2, 2);
    Cell cell2 = largeMesh->getCell(2, 2);
    
    // Both cells should be valid
    EXPECT_TRUE(cell1.isValid());
    EXPECT_TRUE(cell2.isValid());
    
    // They should have the same indices
    EXPECT_EQ(cell1.i(), cell2.i());
    EXPECT_EQ(cell1.j(), cell2.j());
    
    // But different mesh pointers
    EXPECT_NE(cell1.mesh(), cell2.mesh());
    
    // And therefore not be equal
    EXPECT_FALSE(cell1 == cell2);
    
    // Their linear indices may be the same if the meshes use the same indexing formula
    // But we don't test that since it's implementation-dependent
}

// Test creating cells with negative indices (should result in invalid cells)
TEST_F(MeshCellFactoryTest, GetCellWithNegativeIndices) {
    // Note: This test assumes that getCell doesn't validate indices before creating the cell
    // and allows the cells themselves to check their validity
    
    // Create cells with negative indices
    Cell cell1(standardMesh, -1, 0);
    Cell cell2(standardMesh, 0, -1);
    Cell cell3(standardMesh, -1, -1);
    
    // These cells should not be valid
    EXPECT_FALSE(cell1.isValid());
    EXPECT_FALSE(cell2.isValid());
    EXPECT_FALSE(cell3.isValid());
    
    // Their mesh pointers should still be correct
    EXPECT_EQ(cell1.mesh(), standardMesh);
    EXPECT_EQ(cell2.mesh(), standardMesh);
    EXPECT_EQ(cell3.mesh(), standardMesh);
    
    // The negative indices should be preserved
    EXPECT_EQ(cell1.i(), -1);
    EXPECT_EQ(cell2.j(), -1);
    EXPECT_EQ(cell3.i(), -1);
    EXPECT_EQ(cell3.j(), -1);
}

// Test that each getCell call produces a new Cell object
TEST_F(MeshCellFactoryTest, GetCellProducesNewObjects) {
    // Get the same cell twice
    Cell cell1 = standardMesh->getCell(2, 2);
    Cell cell2 = standardMesh->getCell(2, 2);
    
    // The cells should be equal in value
    EXPECT_TRUE(cell1 == cell2);
    
    // But they should be different objects in memory
    EXPECT_TRUE(&cell1 != &cell2);
}

// Test that cells identify boundary status correctly
TEST_F(MeshCellFactoryTest, CellBoundaryDetection) {
    // Create a 3x3 mesh for straightforward boundary testing
    Mesh testMesh(3, 3);
    
    // Check all cells in the mesh
    for (uint32_t j = 0; j < testMesh.ny(); j++) {
        for (uint32_t i = 0; i < testMesh.nx(); i++) {
            Cell cell = testMesh.getCell(i, j);
            
            // Determine if this should be a boundary cell
            bool expectedBoundary = (i == 0 || j == 0 || i == testMesh.nx() - 1 || j == testMesh.ny() - 1);
            
            // Check boundary detection
            EXPECT_EQ(cell.isBoundary(), expectedBoundary) 
                << "Incorrect boundary detection at (" << i << "," << j << ")";
        }
    }
}

// Test getCell with extreme indices that might cause integer overflow
TEST_F(MeshCellFactoryTest, GetCellWithExtremeIndices) {
    // This test is for indices that might cause overflow
    // but are still clearly invalid (beyond any reasonable mesh size)
    
    // Create cells with extreme indices
    Cell cell1(standardMesh, std::numeric_limits<int>::max(), 0);
    Cell cell2(standardMesh, 0, std::numeric_limits<int>::max());
    Cell cell3(standardMesh, std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
    
    // These cells should not be valid
    EXPECT_FALSE(cell1.isValid());
    EXPECT_FALSE(cell2.isValid());
    EXPECT_FALSE(cell3.isValid());
    
    // Also test with minimum possible int values
    Cell cell4(standardMesh, std::numeric_limits<int>::min(), 0);
    Cell cell5(standardMesh, 0, std::numeric_limits<int>::min());
    
    // These should also be invalid
    EXPECT_FALSE(cell4.isValid());
    EXPECT_FALSE(cell5.isValid());
}

} // namespace testing
} // namespace mesh
