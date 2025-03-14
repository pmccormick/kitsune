/**
 * @file CellBasicTests.cpp
 * @brief Unit tests for the basic construction and properties of the Cell class
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include "CellImplementation.h" // Include for complete Cell implementation

namespace mesh {
namespace testing {

/**
 * @brief Test fixture for Cell basic tests
 * 
 * Sets up common mesh configurations for testing Cell properties
 */
class CellBasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create meshes of different sizes for testing
        smallMesh = new Mesh(3, 3);
        standardMesh = new Mesh(10, 10);
        rectangularMesh = new Mesh(5, 15);
    }

    void TearDown() override {
        delete smallMesh;
        delete standardMesh;
        delete rectangularMesh;
    }

    // Meshes with different configurations for testing
    Mesh* smallMesh;
    Mesh* standardMesh;
    Mesh* rectangularMesh;
};

/**
 * @brief Test that the default constructor creates an invalid cell
 */
TEST_F(CellBasicTest, DefaultConstructorCreatesInvalidCell) {
    Cell cell;
    
    // Check that properties indicate an invalid cell
    EXPECT_FALSE(cell.isValid());
    EXPECT_EQ(cell.mesh(), nullptr);
    EXPECT_EQ(cell.i(), -1);
    EXPECT_EQ(cell.j(), -1);
}

/**
 * @brief Test that accessors correctly return cell properties
 */
TEST_F(CellBasicTest, AccessorsReturnCorrectValues) {
    // Test cell in standard mesh
    Cell cell = standardMesh->getCell(3, 4);
    
    EXPECT_EQ(cell.i(), 3);
    EXPECT_EQ(cell.j(), 4);
    EXPECT_EQ(cell.mesh(), standardMesh);
    EXPECT_EQ(cell.indices(), std::make_pair(3, 4));
    EXPECT_TRUE(cell.isValid());
}

/**
 * @brief Test linearIndex calculation for different mesh configurations
 */
TEST_F(CellBasicTest, LinearIndexCalculation) {
    // Test in small mesh
    Cell smallCell = smallMesh->getCell(1, 2);
    // For a 3x3 mesh with row-major ordering, index should be 1 + 2*3 = 7
    EXPECT_EQ(smallCell.linearIndex(), 7);
    
    // Test in standard mesh
    Cell standardCell = standardMesh->getCell(3, 4);
    // For a 10x10 mesh, index should be 3 + 4*10 = 43
    EXPECT_EQ(standardCell.linearIndex(), 43);
    
    // Test in rectangular mesh
    Cell rectangularCell = rectangularMesh->getCell(4, 7);
    // For a 5x15 mesh, index should be 4 + 7*5 = 39
    EXPECT_EQ(rectangularCell.linearIndex(), 39);
}

/**
 * @brief Test that linearIndex throws for invalid cells
 */
TEST_F(CellBasicTest, LinearIndexThrowsForInvalidCell) {
    Cell invalidCell;
    EXPECT_THROW(invalidCell.linearIndex(), std::logic_error);
}

/**
 * @brief Test boundary detection for various cell positions
 */
TEST_F(CellBasicTest, BoundaryDetection) {
    // Test with standard mesh (10x10)
    
    // Corners should be boundaries
    EXPECT_TRUE(standardMesh->getCell(0, 0).isBoundary());  // Bottom-left
    EXPECT_TRUE(standardMesh->getCell(9, 0).isBoundary());  // Bottom-right
    EXPECT_TRUE(standardMesh->getCell(0, 9).isBoundary());  // Top-left
    EXPECT_TRUE(standardMesh->getCell(9, 9).isBoundary());  // Top-right
    
    // Edges should be boundaries
    EXPECT_TRUE(standardMesh->getCell(5, 0).isBoundary());  // Bottom edge
    EXPECT_TRUE(standardMesh->getCell(0, 5).isBoundary());  // Left edge
    EXPECT_TRUE(standardMesh->getCell(9, 5).isBoundary());  // Right edge
    EXPECT_TRUE(standardMesh->getCell(5, 9).isBoundary());  // Top edge
    
    // Interior cells should not be boundaries
    EXPECT_FALSE(standardMesh->getCell(1, 1).isBoundary());
    EXPECT_FALSE(standardMesh->getCell(5, 5).isBoundary());
    EXPECT_FALSE(standardMesh->getCell(8, 8).isBoundary());
}

/**
 * @brief Test that isBoundary throws for invalid cells
 */
TEST_F(CellBasicTest, IsBoundaryThrowsForInvalidCell) {
    Cell invalidCell;
    EXPECT_THROW(invalidCell.isBoundary(), std::logic_error);
}

/**
 * @brief Test isValid method with different cell states
 */
TEST_F(CellBasicTest, IsValidCorrectlyIdentifiesValidCells) {
    // Valid cells
    EXPECT_TRUE(standardMesh->getCell(0, 0).isValid());
    EXPECT_TRUE(standardMesh->getCell(9, 9).isValid());
    EXPECT_TRUE(standardMesh->getCell(5, 5).isValid());
    
    // Invalid cells
    Cell defaultConstructed;
    EXPECT_FALSE(defaultConstructed.isValid());
    
    // Cell with null mesh should be invalid
    Cell nullMeshCell;
    EXPECT_FALSE(nullMeshCell.isValid());
}

/**
 * @brief Test equality operators for cells
 */
TEST_F(CellBasicTest, EqualityOperators) {
    Cell cell1 = standardMesh->getCell(3, 4);
    Cell cell2 = standardMesh->getCell(3, 4);
    Cell cell3 = standardMesh->getCell(4, 3);
    
    // Same indices, same mesh
    EXPECT_TRUE(cell1 == cell2);
    EXPECT_FALSE(cell1 != cell2);
    
    // Different indices, same mesh
    EXPECT_FALSE(cell1 == cell3);
    EXPECT_TRUE(cell1 != cell3);
    
    // Different meshes
    Cell otherMeshCell = smallMesh->getCell(1, 1);
    EXPECT_FALSE(cell1 == otherMeshCell);
    EXPECT_TRUE(cell1 != otherMeshCell);
    
    // Invalid cells
    Cell invalidCell1;
    Cell invalidCell2;
    EXPECT_TRUE(invalidCell1 == invalidCell2);
    EXPECT_FALSE(invalidCell1 != invalidCell2);
}

/**
 * @brief Test construction with edge case indices
 */
TEST_F(CellBasicTest, ConstructionWithEdgeCaseIndices) {
    // Test cells at mesh boundaries
    Cell topLeftCell = standardMesh->getCell(0, 0);
    EXPECT_EQ(topLeftCell.i(), 0);
    EXPECT_EQ(topLeftCell.j(), 0);
    EXPECT_TRUE(topLeftCell.isValid());
    
    // Test cells at maximum indices
    Cell bottomRightCell = standardMesh->getCell(standardMesh->nx() - 1, standardMesh->ny() - 1);
    EXPECT_EQ(bottomRightCell.i(), 9);
    EXPECT_EQ(bottomRightCell.j(), 9);
    EXPECT_TRUE(bottomRightCell.isValid());
}

} // namespace testing
} // namespace mesh

