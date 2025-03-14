/**
 * @file CellEdgeCasesTests.cpp
 * @brief Unit tests for Cell class edge cases and error handling
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell edge cases and error handling tests
 */
class CellEdgeCasesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create standard 10x10 mesh for testing
        standardMesh = new Mesh(10, 10);
        
        // Create a small 2x2 mesh for testing extreme edge cases
        smallMesh = new Mesh(2, 2);
    }

    void TearDown() override {
        delete standardMesh;
        delete smallMesh;
    }

    // Meshes for testing
    Mesh* standardMesh;
    Mesh* smallMesh;
};

/**
 * @brief Test operations on invalid cells throw appropriate exceptions
 */
TEST_F(CellEdgeCasesTest, InvalidCellOperationsThrow) {
    // Create an invalid cell using default constructor
    Cell invalidCell;
    
    // Each operation should throw a logic_error
    EXPECT_THROW(invalidCell.linearIndex(), std::logic_error);
    EXPECT_THROW(invalidCell.isBoundary(), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(NORTH), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(EAST), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(SOUTH), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(WEST), std::logic_error);
    
    // Verify the exception message
    try {
        invalidCell.linearIndex();
        FAIL() << "Expected std::logic_error";
    } catch (const std::logic_error& e) {
        EXPECT_STREQ("Cannot compute linear index: invalid cell", e.what());
    }
}

/**
 * @brief Test operations at mesh boundaries
 */
TEST_F(CellEdgeCasesTest, MeshBoundaryOperations) {
    // Test with standard 10x10 mesh
    
    // Test all 4 corners
    Cell topLeft = standardMesh->getCell(0, 0);
    Cell topRight = standardMesh->getCell(9, 0);
    Cell bottomLeft = standardMesh->getCell(0, 9);
    Cell bottomRight = standardMesh->getCell(9, 9);
    
    // Verify they're all valid
    EXPECT_TRUE(topLeft.isValid());
    EXPECT_TRUE(topRight.isValid());
    EXPECT_TRUE(bottomLeft.isValid());
    EXPECT_TRUE(bottomRight.isValid());
    
    // Verify linear indices
    EXPECT_EQ(topLeft.linearIndex(), 0);
    EXPECT_EQ(topRight.linearIndex(), 9);
    EXPECT_EQ(bottomLeft.linearIndex(), 90);
    EXPECT_EQ(bottomRight.linearIndex(), 99);
    
    // Test neighbors at boundaries - they should return invalid cells
    EXPECT_FALSE(topLeft.neighbor(WEST).isValid());
    EXPECT_FALSE(topLeft.neighbor(SOUTH).isValid());
    EXPECT_FALSE(topRight.neighbor(EAST).isValid());
    EXPECT_FALSE(topRight.neighbor(SOUTH).isValid());
    EXPECT_FALSE(bottomLeft.neighbor(WEST).isValid());
    EXPECT_FALSE(bottomLeft.neighbor(NORTH).isValid());
    EXPECT_FALSE(bottomRight.neighbor(EAST).isValid());
    EXPECT_FALSE(bottomRight.neighbor(NORTH).isValid());
}

/**
 * @brief Test behavior with null mesh pointers
 */
TEST_F(CellEdgeCasesTest, NullMeshPointer) {
    // Create a cell with null mesh pointer
    Cell nullMeshCell;
    
    // Check that the cell is invalid
    EXPECT_FALSE(nullMeshCell.isValid());
    
    // Operations should throw
    EXPECT_THROW(nullMeshCell.linearIndex(), std::logic_error);
    EXPECT_THROW(nullMeshCell.isBoundary(), std::logic_error);
    EXPECT_THROW(nullMeshCell.neighbor(NORTH), std::logic_error);
}

/**
 * @brief Test operations with out-of-bounds indices
 */
TEST_F(CellEdgeCasesTest, OutOfBoundsIndices) {
    // We can't directly create cells with out-of-bounds indices
    // but we can test behavior at boundaries
    
    // Get cell at edge
    Cell edgeCell = standardMesh->getCell(9, 5);
    
    // Neighbor outside bounds should be invalid
    Cell outOfBounds = edgeCell.neighbor(EAST);
    EXPECT_FALSE(outOfBounds.isValid());
    
    // Operations on invalid out-of-bounds cell should throw
    EXPECT_THROW(outOfBounds.linearIndex(), std::logic_error);
    EXPECT_THROW(outOfBounds.isBoundary(), std::logic_error);
    EXPECT_THROW(outOfBounds.neighbor(NORTH), std::logic_error);
}

/**
 * @brief Test small 2x2 mesh edge cases
 */
TEST_F(CellEdgeCasesTest, SmallMeshEdgeCases) {
    // With a 2x2 mesh, every cell is a boundary cell
    Cell topLeft = smallMesh->getCell(0, 0);
    Cell topRight = smallMesh->getCell(1, 0);
    Cell bottomLeft = smallMesh->getCell(0, 1);
    Cell bottomRight = smallMesh->getCell(1, 1);
    
    // Verify all are boundary cells
    EXPECT_TRUE(topLeft.isBoundary());
    EXPECT_TRUE(topRight.isBoundary());
    EXPECT_TRUE(bottomLeft.isBoundary());
    EXPECT_TRUE(bottomRight.isBoundary());
    
    // Test corner cases with diagonal neighbors
    EXPECT_FALSE(topLeft.neighbor(NORTH | WEST).isValid());
    EXPECT_FALSE(topRight.neighbor(NORTH | EAST).isValid());
    EXPECT_FALSE(bottomLeft.neighbor(SOUTH | WEST).isValid());
    EXPECT_FALSE(bottomRight.neighbor(SOUTH | EAST).isValid());
    
    // But some diagonal neighbors within the mesh should be valid
    EXPECT_TRUE(topLeft.neighbor(NORTH | EAST).isValid());
    EXPECT_TRUE(bottomRight.neighbor(SOUTH | WEST).isValid());
}

/**
 * @brief Test linearIndex for 1x1 mesh (extreme edge case)
 */
TEST_F(CellEdgeCasesTest, OneCellMesh) {
    // Create a 1x1 mesh
    Mesh oneCellMesh(1, 1);
    
    // Get the only cell
    Cell onlyCell = oneCellMesh.getCell(0, 0);
    
    // Verify properties
    EXPECT_TRUE(onlyCell.isValid());
    EXPECT_TRUE(onlyCell.isBoundary());
    EXPECT_EQ(onlyCell.linearIndex(), 0);
    
    // All neighbors should be invalid
    EXPECT_FALSE(onlyCell.neighbor(NORTH).isValid());
    EXPECT_FALSE(onlyCell.neighbor(EAST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(SOUTH).isValid());
    EXPECT_FALSE(onlyCell.neighbor(WEST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(NORTH | EAST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(NORTH | WEST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(SOUTH | EAST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(SOUTH | WEST).isValid());
}

/**
 * @brief Test with unusual direction flags
 */
TEST_F(CellEdgeCasesTest, UnusualDirectionFlags) {
    Cell cell = standardMesh->getCell(5, 5);
    
    // Test with empty flag (0) - should return same position
    Cell sameCell = cell.neighbor(0);
    EXPECT_EQ(sameCell.i(), 5);
    EXPECT_EQ(sameCell.j(), 5);
    
    // Test with all direction flags set
    Cell allFlags = cell.neighbor(NORTH | EAST | SOUTH | WEST);
    EXPECT_EQ(allFlags.i(), 5);
    EXPECT_EQ(allFlags.j(), 5);
    
    // Test with unusual combinations (opposing directions)
    Cell northSouth = cell.neighbor(NORTH | SOUTH);
    EXPECT_EQ(northSouth.i(), 5);
    EXPECT_EQ(northSouth.j(), 5);
    
    Cell eastWest = cell.neighbor(EAST | WEST);
    EXPECT_EQ(eastWest.i(), 5);
    EXPECT_EQ(eastWest.j(), 5);
}

/**
 * @brief Test sequence of operations that might lead to issues
 */
TEST_F(CellEdgeCasesTest, OperationSequences) {
    // Start with a valid cell
    Cell cell = standardMesh->getCell(5, 5);
    
    // Get an invalid neighbor (out of bounds)
    Cell invalid = standardMesh->getCell(0, 0).neighbor(WEST);
    EXPECT_FALSE(invalid.isValid());
    
    // Try to navigate from an invalid cell - should throw
    EXPECT_THROW(invalid.neighbor(EAST), std::logic_error);
    
    // We don't need to test this twice since it's the same operation
    
    // Chain of valid operations should work
    Cell chained = cell.neighbor(NORTH).neighbor(EAST).neighbor(NORTH);
    EXPECT_TRUE(chained.isValid());
    EXPECT_EQ(chained.i(), 6);
    EXPECT_EQ(chained.j(), 7);
    
    // Chain that goes out of bounds at the end
    Cell outOfBounds = standardMesh->getCell(9, 9);
    
    // First go NORTH (which will be invalid)
    Cell northNeighbor = outOfBounds.neighbor(NORTH);
    EXPECT_FALSE(northNeighbor.isValid());
    
    // Don't try to chain neighbor calls on invalid cells
}

} // namespace testing
} // namespace mesh


