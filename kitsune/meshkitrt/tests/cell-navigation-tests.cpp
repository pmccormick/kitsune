/**
 * @file CellNavigationTests.cpp
 * @brief Tests for Cell navigation functionality
 * 
 * These tests focus on the neighbor() method and cell navigation
 * in different directions, including boundary conditions.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"

namespace mesh {
namespace testing {

// Fixture for Cell navigation tests
class CellNavigationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for navigation tests
        mesh = new Mesh(5, 5);
    }

    void TearDown() override {
        delete mesh;
    }

    Mesh* mesh;
};

// Test navigation in cardinal directions from interior cell
TEST_F(CellNavigationTest, NavigateFromInteriorCell) {
    Cell center = mesh->getCell(2, 2);
    
    // Test navigation in each cardinal direction
    Cell north = center.neighbor(NORTH);
    EXPECT_TRUE(north.isValid());
    EXPECT_EQ(north.i(), 2);
    EXPECT_EQ(north.j(), 3);
    
    Cell east = center.neighbor(EAST);
    EXPECT_TRUE(east.isValid());
    EXPECT_EQ(east.i(), 3);
    EXPECT_EQ(east.j(), 2);
    
    Cell south = center.neighbor(SOUTH);
    EXPECT_TRUE(south.isValid());
    EXPECT_EQ(south.i(), 2);
    EXPECT_EQ(south.j(), 1);
    
    Cell west = center.neighbor(WEST);
    EXPECT_TRUE(west.isValid());
    EXPECT_EQ(west.i(), 1);
    EXPECT_EQ(west.j(), 2);
}

// Test navigation in diagonal directions from interior cell
TEST_F(CellNavigationTest, NavigateDiagonallyFromInteriorCell) {
    Cell center = mesh->getCell(2, 2);
    
    // Test diagonal navigation
    Cell northeast = center.neighbor(NORTH | EAST);
    EXPECT_TRUE(northeast.isValid());
    EXPECT_EQ(northeast.i(), 3);
    EXPECT_EQ(northeast.j(), 3);
    
    Cell southeast = center.neighbor(SOUTH | EAST);
    EXPECT_TRUE(southeast.isValid());
    EXPECT_EQ(southeast.i(), 3);
    EXPECT_EQ(southeast.j(), 1);
    
    Cell southwest = center.neighbor(SOUTH | WEST);
    EXPECT_TRUE(southwest.isValid());
    EXPECT_EQ(southwest.i(), 1);
    EXPECT_EQ(southwest.j(), 1);
    
    Cell northwest = center.neighbor(NORTH | WEST);
    EXPECT_TRUE(northwest.isValid());
    EXPECT_EQ(northwest.i(), 1);
    EXPECT_EQ(northwest.j(), 3);
}

// Test navigation from boundary cells
TEST_F(CellNavigationTest, NavigateFromBoundary) {
    // Test from left edge
    Cell leftEdge = mesh->getCell(0, 2);
    Cell leftToEast = leftEdge.neighbor(EAST);
    EXPECT_TRUE(leftToEast.isValid());
    EXPECT_EQ(leftToEast.i(), 1);
    EXPECT_EQ(leftToEast.j(), 2);
    
    Cell leftToWest = leftEdge.neighbor(WEST);
    EXPECT_FALSE(leftToWest.isValid()); // Should be invalid (outside mesh)
    
    // Test from right edge
    Cell rightEdge = mesh->getCell(4, 2);
    Cell rightToWest = rightEdge.neighbor(WEST);
    EXPECT_TRUE(rightToWest.isValid());
    EXPECT_EQ(rightToWest.i(), 3);
    EXPECT_EQ(rightToWest.j(), 2);
    
    Cell rightToEast = rightEdge.neighbor(EAST);
    EXPECT_FALSE(rightToEast.isValid()); // Should be invalid (outside mesh)
    
    // Test from bottom edge
    Cell bottomEdge = mesh->getCell(2, 0);
    Cell bottomToNorth = bottomEdge.neighbor(NORTH);
    EXPECT_TRUE(bottomToNorth.isValid());
    EXPECT_EQ(bottomToNorth.i(), 2);
    EXPECT_EQ(bottomToNorth.j(), 1);
    
    Cell bottomToSouth = bottomEdge.neighbor(SOUTH);
    EXPECT_FALSE(bottomToSouth.isValid()); // Should be invalid (outside mesh)
    
    // Test from top edge
    Cell topEdge = mesh->getCell(2, 4);
    Cell topToSouth = topEdge.neighbor(SOUTH);
    EXPECT_TRUE(topToSouth.isValid());
    EXPECT_EQ(topToSouth.i(), 2);
    EXPECT_EQ(topToSouth.j(), 3);
    
    Cell topToNorth = topEdge.neighbor(NORTH);
    EXPECT_FALSE(topToNorth.isValid()); // Should be invalid (outside mesh)
}

// Test navigation from corner cells
TEST_F(CellNavigationTest, NavigateFromCorners) {
    // Bottom-left corner
    Cell bottomLeft = mesh->getCell(0, 0);
    EXPECT_FALSE(bottomLeft.neighbor(WEST).isValid());
    EXPECT_FALSE(bottomLeft.neighbor(SOUTH).isValid());
    EXPECT_FALSE(bottomLeft.neighbor(SOUTH | WEST).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(NORTH).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(EAST).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(NORTH | EAST).isValid());
    
    // Top-right corner
    Cell topRight = mesh->getCell(4, 4);
    EXPECT_FALSE(topRight.neighbor(EAST).isValid());
    EXPECT_FALSE(topRight.neighbor(NORTH).isValid());
    EXPECT_FALSE(topRight.neighbor(NORTH | EAST).isValid());
    EXPECT_TRUE(topRight.neighbor(SOUTH).isValid());
    EXPECT_TRUE(topRight.neighbor(WEST).isValid());
    EXPECT_TRUE(topRight.neighbor(SOUTH | WEST).isValid());
}

// Test navigation with no direction specified
TEST_F(CellNavigationTest, NavigateWithNoDirection) {
    Cell cell = mesh->getCell(2, 2);
    Cell result = cell.neighbor(0); // No direction flags set
    
    // Should return the same position (no movement)
    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.i(), 2);
    EXPECT_EQ(result.j(), 2);
}

// Test navigation from invalid cell
TEST_F(CellNavigationTest, NavigateFromInvalidCell) {
    Cell invalidCell; // Default constructor creates invalid cell
    
    // Trying to get neighbor from invalid cell should throw
    EXPECT_THROW(invalidCell.neighbor(NORTH), std::logic_error);
}

// Test multiple navigation steps (chaining neighbors)
TEST_F(CellNavigationTest, MultiStepNavigation) {
    Cell start = mesh->getCell(1, 1);
    
    // Navigate multiple steps
    Cell result = start.neighbor(NORTH).neighbor(EAST).neighbor(NORTH);
    
    EXPECT_TRUE(result.isValid());
    EXPECT_EQ(result.i(), 2);
    EXPECT_EQ(result.j(), 3);
}

// Test that neighbor() properly handles combined directions
TEST_F(CellNavigationTest, CombinedDirections) {
    Cell center = mesh->getCell(2, 2);
    
    // Test with multiple directions set
    Cell diagResult = center.neighbor(NORTH | EAST);
    Cell stepResult = center.neighbor(NORTH).neighbor(EAST);
    
    // The results should be the same
    EXPECT_EQ(diagResult.i(), stepResult.i());
    EXPECT_EQ(diagResult.j(), stepResult.j());
    
    // Should be at (3,3)
    EXPECT_EQ(diagResult.i(), 3);
    EXPECT_EQ(diagResult.j(), 3);
}

} // namespace testing
} // namespace mesh
