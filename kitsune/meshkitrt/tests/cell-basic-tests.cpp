/**
 * @file CellBasicTests.cpp
 * @brief Basic unit tests for the Cell class
 * 
 * These tests focus on cell construction, equality comparison,
 * and other fundamental operations.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"

namespace mesh {
namespace testing {

// Fixture for Cell tests that need a mesh
class CellBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for cell operations
        mesh = new Mesh(5, 5);
    }

    void TearDown() override {
        delete mesh;
    }

    Mesh* mesh;
};

// Test the default constructor creates an invalid cell
TEST_F(CellBasicTest, DefaultConstructorCreatesInvalidCell) {
    Cell cell;
    EXPECT_FALSE(cell.isValid());
    EXPECT_EQ(cell.i(), -1);
    EXPECT_EQ(cell.j(), -1);
    EXPECT_EQ(cell.mesh(), nullptr);
}

// Test getting a valid cell from the mesh
TEST_F(CellBasicTest, GetValidCell) {
    Cell cell = mesh->getCell(2, 3);
    EXPECT_TRUE(cell.isValid());
    EXPECT_EQ(cell.i(), 2);
    EXPECT_EQ(cell.j(), 3);
    EXPECT_EQ(cell.mesh(), mesh);
}

// Test cell equality operator
TEST_F(CellBasicTest, EqualityOperator) {
    Cell cell1 = mesh->getCell(1, 2);
    Cell cell2 = mesh->getCell(1, 2);
    Cell cell3 = mesh->getCell(2, 1);
    
    EXPECT_TRUE(cell1 == cell2);
    EXPECT_FALSE(cell1 == cell3);
}

// Test cell inequality operator
TEST_F(CellBasicTest, InequalityOperator) {
    Cell cell1 = mesh->getCell(1, 2);
    Cell cell2 = mesh->getCell(1, 2);
    Cell cell3 = mesh->getCell(2, 1);
    
    EXPECT_FALSE(cell1 != cell2);
    EXPECT_TRUE(cell1 != cell3);
}

// Test the indices() method
TEST_F(CellBasicTest, IndicesMethod) {
    Cell cell = mesh->getCell(2, 3);
    auto indices = cell.indices();
    
    EXPECT_EQ(indices.first, 2);
    EXPECT_EQ(indices.second, 3);
}

// Test isValid for cells at different locations
TEST_F(CellBasicTest, ValidityAtDifferentLocations) {
    // Test corners
    EXPECT_TRUE(mesh->getCell(0, 0).isValid());
    EXPECT_TRUE(mesh->getCell(0, 4).isValid());
    EXPECT_TRUE(mesh->getCell(4, 0).isValid());
    EXPECT_TRUE(mesh->getCell(4, 4).isValid());
    
    // Test interior
    EXPECT_TRUE(mesh->getCell(2, 2).isValid());
    
    // Test edges
    EXPECT_TRUE(mesh->getCell(0, 2).isValid());
    EXPECT_TRUE(mesh->getCell(2, 0).isValid());
    EXPECT_TRUE(mesh->getCell(4, 2).isValid());
    EXPECT_TRUE(mesh->getCell(2, 4).isValid());
}

// Test isValid for out-of-bounds cells
TEST_F(CellBasicTest, InvalidForOutOfBoundsCells) {
    // Test with negative indices
    Cell invalidNegative(mesh, -1, 2);
    EXPECT_FALSE(invalidNegative.isValid());
    
    // Test with out-of-bounds positive indices
    Cell invalidTooLarge(mesh, 5, 2);
    EXPECT_FALSE(invalidTooLarge.isValid());
    
    // Test with nullptr mesh
    Cell invalidNullMesh(nullptr, 2, 2);
    EXPECT_FALSE(invalidNullMesh.isValid());
}

// Test getDirectionOffset static method
TEST_F(CellBasicTest, DirectionOffset) {
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

} // namespace testing
} // namespace mesh
