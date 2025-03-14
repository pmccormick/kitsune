/**
 * @file CellNavigationTests.cpp
 * @brief Unit tests for the Cell's navigation functionality
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell navigation tests
 */
class CellNavigationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create standard 10x10 mesh for navigation tests
        standardMesh = new Mesh(10, 10);
    }

    void TearDown() override {
        delete standardMesh;
    }

    // Mesh for testing
    Mesh* standardMesh;
};

/**
 * @brief Test getting neighbors in cardinal directions
 */
TEST_F(CellNavigationTest, CardinalNeighbors) {
    // Get a cell in the middle of the mesh
    Cell cell = standardMesh->getCell(5, 5);
    
    // Test getting neighbor to the north (increasing j)
    Cell north = cell.neighbor(NORTH);
    EXPECT_EQ(north.i(), 5);
    EXPECT_EQ(north.j(), 6);
    EXPECT_TRUE(north.isValid());
    
    // Test getting neighbor to the east (increasing i)
    Cell east = cell.neighbor(EAST);
    EXPECT_EQ(east.i(), 6);
    EXPECT_EQ(east.j(), 5);
    EXPECT_TRUE(east.isValid());
    
    // Test getting neighbor to the south (decreasing j)
    Cell south = cell.neighbor(SOUTH);
    EXPECT_EQ(south.i(), 5);
    EXPECT_EQ(south.j(), 4);
    EXPECT_TRUE(south.isValid());
    
    // Test getting neighbor to the west (decreasing i)
    Cell west = cell.neighbor(WEST);
    EXPECT_EQ(west.i(), 4);
    EXPECT_EQ(west.j(), 5);
    EXPECT_TRUE(west.isValid());
}

/**
 * @brief Test getting neighbors in diagonal directions
 */
TEST_F(CellNavigationTest, DiagonalNeighbors) {
    // Get a cell in the middle of the mesh
    Cell cell = standardMesh->getCell(5, 5);
    
    // Test getting neighbor to the northeast
    Cell northeast = cell.neighbor(NORTH | EAST);
    EXPECT_EQ(northeast.i(), 6);
    EXPECT_EQ(northeast.j(), 6);
    EXPECT_TRUE(northeast.isValid());
    
    // Test getting neighbor to the southeast
    Cell southeast = cell.neighbor(SOUTH | EAST);
    EXPECT_EQ(southeast.i(), 6);
    EXPECT_EQ(southeast.j(), 4);
    EXPECT_TRUE(southeast.isValid());
    
    // Test getting neighbor to the southwest
    Cell southwest = cell.neighbor(SOUTH | WEST);
    EXPECT_EQ(southwest.i(), 4);
    EXPECT_EQ(southwest.j(), 4);
    EXPECT_TRUE(southwest.isValid());
    
    // Test getting neighbor to the northwest
    Cell northwest = cell.neighbor(NORTH | WEST);
    EXPECT_EQ(northwest.i(), 4);
    EXPECT_EQ(northwest.j(), 6);
    EXPECT_TRUE(northwest.isValid());
}

/**
 * @brief Test boundary conditions for neighbors
 */
TEST_F(CellNavigationTest, BoundaryNeighbors) {
    // Test at bottom-left corner (0,0)
    Cell bottomLeft = standardMesh->getCell(0, 0);
    
    // West should be invalid (out of bounds)
    Cell westOfBottomLeft = bottomLeft.neighbor(WEST);
    EXPECT_FALSE(westOfBottomLeft.isValid());
    
    // South should be invalid (out of bounds)
    Cell southOfBottomLeft = bottomLeft.neighbor(SOUTH);
    EXPECT_FALSE(southOfBottomLeft.isValid());
    
    // Southwest should be invalid (out of bounds)
    Cell southwestOfBottomLeft = bottomLeft.neighbor(SOUTH | WEST);
    EXPECT_FALSE(southwestOfBottomLeft.isValid());
    
    // North and East should be valid
    EXPECT_TRUE(bottomLeft.neighbor(NORTH).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(EAST).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(NORTH | EAST).isValid());
    
    // Test at top-right corner (9,9)
    Cell topRight = standardMesh->getCell(9, 9);
    
    // East should be invalid (out of bounds)
    Cell eastOfTopRight = topRight.neighbor(EAST);
    EXPECT_FALSE(eastOfTopRight.isValid());
    
    // North should be invalid (out of bounds)
    Cell northOfTopRight = topRight.neighbor(NORTH);
    EXPECT_FALSE(northOfTopRight.isValid());
    
    // Northeast should be invalid (out of bounds)
    Cell northeastOfTopRight = topRight.neighbor(NORTH | EAST);
    EXPECT_FALSE(northeastOfTopRight.isValid());
    
    // South and West should be valid
    EXPECT_TRUE(topRight.neighbor(SOUTH).isValid());
    EXPECT_TRUE(topRight.neighbor(WEST).isValid());
    EXPECT_TRUE(topRight.neighbor(SOUTH | WEST).isValid());
}

/**
 * @brief Test that neighbor method throws for invalid cells
 */
TEST_F(CellNavigationTest, NeighborThrowsForInvalidCell) {
    Cell invalidCell;
    EXPECT_THROW(invalidCell.neighbor(NORTH), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(EAST), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(SOUTH), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(WEST), std::logic_error);
    EXPECT_THROW(invalidCell.neighbor(NORTH | EAST), std::logic_error);
}

/**
 * @brief Test direction offset calculation
 */
TEST_F(CellNavigationTest, DirectionOffsetCalculation) {
    // Test cardinal directions
    auto northOffset = Cell::getDirectionOffset(NORTH);
    EXPECT_EQ(northOffset.first, 0);
    EXPECT_EQ(northOffset.second, 1);
    
    auto eastOffset = Cell::getDirectionOffset(EAST);
    EXPECT_EQ(eastOffset.first, 1);
    EXPECT_EQ(eastOffset.second, 0);
    
    auto southOffset = Cell::getDirectionOffset(SOUTH);
    EXPECT_EQ(southOffset.first, 0);
    EXPECT_EQ(southOffset.second, -1);
    
    auto westOffset = Cell::getDirectionOffset(WEST);
    EXPECT_EQ(westOffset.first, -1);
    EXPECT_EQ(westOffset.second, 0);
    
    // Test diagonal directions
    auto northeastOffset = Cell::getDirectionOffset(NORTH | EAST);
    EXPECT_EQ(northeastOffset.first, 1);
    EXPECT_EQ(northeastOffset.second, 1);
    
    auto southeastOffset = Cell::getDirectionOffset(SOUTH | EAST);
    EXPECT_EQ(southeastOffset.first, 1);
    EXPECT_EQ(southeastOffset.second, -1);
    
    auto southwestOffset = Cell::getDirectionOffset(SOUTH | WEST);
    EXPECT_EQ(southwestOffset.first, -1);
    EXPECT_EQ(southwestOffset.second, -1);
    
    auto northwestOffset = Cell::getDirectionOffset(NORTH | WEST);
    EXPECT_EQ(northwestOffset.first, -1);
    EXPECT_EQ(northwestOffset.second, 1);
}

/**
 * @brief Test navigation with zero direction
 */
TEST_F(CellNavigationTest, ZeroDirectionNavigation) {
    Cell cell = standardMesh->getCell(5, 5);
    
    // Zero direction should return a neighbor at the same position
    Cell same = cell.neighbor(0);
    EXPECT_EQ(same.i(), 5);
    EXPECT_EQ(same.j(), 5);
    EXPECT_TRUE(same.isValid());
}

/**
 * @brief Test navigation with all directions combined
 */
TEST_F(CellNavigationTest, CombinedDirectionNavigation) {
    Cell cell = standardMesh->getCell(5, 5);
    
    // Combine NORTH and SOUTH (should cancel out j-component)
    Cell northAndSouth = cell.neighbor(NORTH | SOUTH);
    EXPECT_EQ(northAndSouth.i(), 5);
    EXPECT_EQ(northAndSouth.j(), 5);
    
    // Combine EAST and WEST (should cancel out i-component)
    Cell eastAndWest = cell.neighbor(EAST | WEST);
    EXPECT_EQ(eastAndWest.i(), 5);
    EXPECT_EQ(eastAndWest.j(), 5);
    
    // Combine all directions (should cancel out)
    Cell allDirections = cell.neighbor(NORTH | EAST | SOUTH | WEST);
    EXPECT_EQ(allDirections.i(), 5);
    EXPECT_EQ(allDirections.j(), 5);
}

/**
 * @brief Test multiple steps of navigation
 */
TEST_F(CellNavigationTest, MultiStepNavigation) {
    Cell start = standardMesh->getCell(5, 5);
    
    // Take multiple steps in different directions
    Cell step1 = start.neighbor(NORTH);
    Cell step2 = step1.neighbor(EAST);
    Cell step3 = step2.neighbor(NORTH);
    
    // Check final position
    EXPECT_EQ(step3.i(), 6);
    EXPECT_EQ(step3.j(), 7);
    
    // Try different path to same destination
    Cell altPath = start.neighbor(NORTH | EAST).neighbor(NORTH);
    EXPECT_EQ(altPath.i(), 6);
    EXPECT_EQ(altPath.j(), 7);
    
    // Verify both paths lead to the same cell
    EXPECT_TRUE(step3 == altPath);
}

} // namespace testing
} // namespace mesh


