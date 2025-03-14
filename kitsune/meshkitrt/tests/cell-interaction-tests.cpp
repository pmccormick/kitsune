/**
 * @file CellMeshInteractionTests.cpp
 * @brief Unit tests for the interaction between Cell and Mesh classes
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"

namespace mesh {
namespace testing {

/**
 * @brief Mock Mesh class for testing Cell interactions with Mesh
 */
class MockMesh : public Mesh {
public:
    MockMesh(uint32_t nx, uint32_t ny) : Mesh(nx, ny) {}
};

/**
 * @brief Test fixture for Cell-Mesh interaction tests
 */
class CellMeshInteractionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create mock mesh for testing
        mockMesh = new MockMesh(10, 10);
    }

    void TearDown() override {
        delete mockMesh;
    }

    // Mock mesh for testing
    MockMesh* mockMesh;
};

/**
 * @brief Test that linearIndex calculations are correct
 */
TEST_F(CellMeshInteractionTest, LinearIndexCalculation) {
    // Get a cell from the mock mesh
    Cell cell = mockMesh->getCell(3, 4);
    
    // Call linearIndex
    int result = cell.linearIndex();
    
    // Verify result is correct based on the mesh dimensions
    EXPECT_EQ(result, 3 + 4 * mockMesh->nx());
}

/**
 * @brief Test that neighbor finding respects mesh dimensions
 */
TEST_F(CellMeshInteractionTest, NeighborRespectsMeshDimensions) {
    // Get a cell from the mock mesh
    Cell cell = mockMesh->getCell(0, 0); // Corner cell
    
    // Try to get a neighbor outside the mesh
    Cell westNeighbor = cell.neighbor(WEST);
    
    // Verify neighbor is invalid
    EXPECT_FALSE(westNeighbor.isValid());
    
    // Try a valid neighbor
    Cell eastNeighbor = cell.neighbor(EAST);
    
    // Verify neighbor is valid and at expected position
    EXPECT_TRUE(eastNeighbor.isValid());
    EXPECT_EQ(eastNeighbor.i(), 1);
    EXPECT_EQ(eastNeighbor.j(), 0);
}

/**
 * @brief Test cell creation in mesh of different dimensions
 */
TEST_F(CellMeshInteractionTest, MeshDimensionsAffectCells) {
    // Create meshes with different dimensions
    MockMesh smallMesh(5, 5);
    MockMesh rectangularMesh(10, 20);
    
    // Create cells in different meshes
    Cell smallCell = smallMesh.getCell(4, 4);
    Cell rectangularCell = rectangularMesh.getCell(9, 19);
    
    // Verify cells are valid
    EXPECT_TRUE(smallCell.isValid());
    EXPECT_TRUE(rectangularCell.isValid());
    
    // Verify boundary status
    EXPECT_TRUE(smallCell.isBoundary());
    EXPECT_TRUE(rectangularCell.isBoundary());
    
    // Verify linearIndex reflects mesh dimensions
    EXPECT_EQ(smallCell.linearIndex(), 24); // 4 + 4*5
    EXPECT_EQ(rectangularCell.linearIndex(), 199); // 9 + 19*10
}

/**
 * @brief Test that cells with same indices but different meshes are distinct
 */
TEST_F(CellMeshInteractionTest, CellsMeshAwareness) {
    // Create another mesh
    MockMesh anotherMesh(10, 10);
    
    // Create cells with same indices in different meshes
    Cell cell1 = mockMesh->getCell(3, 4);
    Cell cell2 = anotherMesh.getCell(3, 4);
    
    // Verify cells have same indices
    EXPECT_EQ(cell1.i(), cell2.i());
    EXPECT_EQ(cell1.j(), cell2.j());
    
    // But cells should not be equal due to different mesh pointers
    EXPECT_FALSE(cell1 == cell2);
    
    // Verify each cell references its own mesh
    EXPECT_EQ(cell1.mesh(), mockMesh);
    EXPECT_EQ(cell2.mesh(), &anotherMesh);
}

/**
 * @brief Test that cells maintain correct mesh reference
 */
TEST_F(CellMeshInteractionTest, CellMeshReferenceMaintained) {
    // Create a cell from the mock mesh
    Cell cell = mockMesh->getCell(5, 5);
    
    // Verify mesh reference
    EXPECT_EQ(cell.mesh(), mockMesh);
    
    // Get neighbors and verify they maintain the same mesh reference
    Cell northNeighbor = cell.neighbor(NORTH);
    Cell eastNeighbor = cell.neighbor(EAST);
    
    EXPECT_EQ(northNeighbor.mesh(), mockMesh);
    EXPECT_EQ(eastNeighbor.mesh(), mockMesh);
}

/**
 * @brief Test cell behavior at mesh boundaries
 */
TEST_F(CellMeshInteractionTest, CellsAtMeshBoundaries) {
    // Create mesh with custom dimensions
    MockMesh customMesh(7, 13);
    
    // Test cells at each edge
    Cell leftEdge = customMesh.getCell(0, 5);
    Cell rightEdge = customMesh.getCell(6, 5);
    Cell bottomEdge = customMesh.getCell(3, 0);
    Cell topEdge = customMesh.getCell(3, 12);
    
    // Verify all are boundary cells
    EXPECT_TRUE(leftEdge.isBoundary());
    EXPECT_TRUE(rightEdge.isBoundary());
    EXPECT_TRUE(bottomEdge.isBoundary());
    EXPECT_TRUE(topEdge.isBoundary());
    
    // Test neighbors outside the mesh
    EXPECT_FALSE(leftEdge.neighbor(WEST).isValid());
    EXPECT_FALSE(rightEdge.neighbor(EAST).isValid());
    EXPECT_FALSE(bottomEdge.neighbor(SOUTH).isValid());
    EXPECT_FALSE(topEdge.neighbor(NORTH).isValid());
    
    // Test neighbors inside the mesh
    EXPECT_TRUE(leftEdge.neighbor(EAST).isValid());
    EXPECT_TRUE(rightEdge.neighbor(WEST).isValid());
    EXPECT_TRUE(bottomEdge.neighbor(NORTH).isValid());
    EXPECT_TRUE(topEdge.neighbor(SOUTH).isValid());
}

/**
 * @brief Test cells at mesh corners
 */
TEST_F(CellMeshInteractionTest, CellsAtMeshCorners) {
    // Test all four corners of the mock mesh
    Cell bottomLeft = mockMesh->getCell(0, 0);
    Cell bottomRight = mockMesh->getCell(9, 0);
    Cell topLeft = mockMesh->getCell(0, 9);
    Cell topRight = mockMesh->getCell(9, 9);
    
    // Verify all are boundary cells
    EXPECT_TRUE(bottomLeft.isBoundary());
    EXPECT_TRUE(bottomRight.isBoundary());
    EXPECT_TRUE(topLeft.isBoundary());
    EXPECT_TRUE(topRight.isBoundary());
    
    // Test valid neighbors for each corner
    EXPECT_TRUE(bottomLeft.neighbor(NORTH).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(EAST).isValid());
    EXPECT_TRUE(bottomLeft.neighbor(NORTH | EAST).isValid());
    
    EXPECT_TRUE(bottomRight.neighbor(NORTH).isValid());
    EXPECT_TRUE(bottomRight.neighbor(WEST).isValid());
    EXPECT_TRUE(bottomRight.neighbor(NORTH | WEST).isValid());
    
    EXPECT_TRUE(topLeft.neighbor(SOUTH).isValid());
    EXPECT_TRUE(topLeft.neighbor(EAST).isValid());
    EXPECT_TRUE(topLeft.neighbor(SOUTH | EAST).isValid());
    
    EXPECT_TRUE(topRight.neighbor(SOUTH).isValid());
    EXPECT_TRUE(topRight.neighbor(WEST).isValid());
    EXPECT_TRUE(topRight.neighbor(SOUTH | WEST).isValid());
}

/**
 * @brief Test cell behavior with interior cells
 */
TEST_F(CellMeshInteractionTest, InteriorCells) {
    // Create an interior cell
    Cell interior = mockMesh->getCell(5, 5);
    
    // Verify it's not a boundary cell
    EXPECT_FALSE(interior.isBoundary());
    
    // All neighbors should be valid
    EXPECT_TRUE(interior.neighbor(NORTH).isValid());
    EXPECT_TRUE(interior.neighbor(EAST).isValid());
    EXPECT_TRUE(interior.neighbor(SOUTH).isValid());
    EXPECT_TRUE(interior.neighbor(WEST).isValid());
    
    // All diagonal neighbors should be valid
    EXPECT_TRUE(interior.neighbor(NORTH | EAST).isValid());
    EXPECT_TRUE(interior.neighbor(NORTH | WEST).isValid());
    EXPECT_TRUE(interior.neighbor(SOUTH | EAST).isValid());
    EXPECT_TRUE(interior.neighbor(SOUTH | WEST).isValid());
}

} // namespace testing
} // namespace mesh

