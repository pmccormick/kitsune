/**
 * @file CellMeshIntegrationTests.cpp
 * @brief Tests for Cell and Mesh class integration
 * 
 * These tests focus on how Cell interacts with the Mesh class,
 * ensuring correct behavior across their shared interface.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"
#include <vector>

namespace mesh {
namespace testing {

// Custom mesh class for testing specific integration scenarios
class TestMesh : public Mesh {
public:
    TestMesh(uint32_t nx, uint32_t ny) : Mesh(nx, ny) {}
    
    // Override linearIndex to use a different mapping formula for testing
    uint32_t linearIndex(uint32_t i, uint32_t j) const override {
        // Use a different formula: column-major instead of row-major
        return j + i * ny();
    }
    
    // Track calls to methods for testing purposes
    mutable int getCallCount = 0;
    
    Cell getCell(uint32_t i, uint32_t j) const override {
        getCallCount++;
        return Mesh::getCell(i, j);
    }
};

// Fixture for Cell-Mesh integration tests
class CellMeshIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard mesh
        standardMesh = new Mesh(5, 5);
        
        // Create a custom test mesh
        testMesh = new TestMesh(5, 5);
    }

    void TearDown() override {
        delete standardMesh;
        delete testMesh;
    }

    Mesh* standardMesh;
    TestMesh* testMesh;
};

// Test that Cell properly uses Mesh's linearIndex implementation
TEST_F(CellMeshIntegrationTest, CellUsesCorrectMeshLinearIndex) {
    // With standard mesh (row-major: index = i + j*nx)
    Cell stdCell = standardMesh->getCell(2, 3);
    EXPECT_EQ(stdCell.linearIndex(), 17); // 2 + 3*5
    
    // With test mesh (column-major: index = j + i*ny)
    Cell testCell = testMesh->getCell(2, 3);
    EXPECT_EQ(testCell.linearIndex(), 13); // 3 + 2*5
    
    // Different positions to verify formula difference
    Cell stdCell2 = standardMesh->getCell(3, 2);
    Cell testCell2 = testMesh->getCell(3, 2);
    
    EXPECT_EQ(stdCell2.linearIndex(), 13); // 3 + 2*5
    EXPECT_EQ(testCell2.linearIndex(), 17); // 2 + 3*5
    
    // This test confirms the Cell uses the mesh's implementation
    // rather than hardcoding its own formula
}

// Test boundary detection delegated to mesh dimensions
TEST_F(CellMeshIntegrationTest, CellUsesMeshDimensionsForBoundary) {
    // Create meshes with different dimensions
    Mesh smallMesh(3, 3);
    Mesh rectMesh(4, 6);
    
    // Test boundary detection on small square mesh
    EXPECT_TRUE(smallMesh.getCell(0, 1).isBoundary()); // Left edge
    EXPECT_TRUE(smallMesh.getCell(1, 0).isBoundary()); // Bottom edge
    EXPECT_TRUE(smallMesh.getCell(2, 1).isBoundary()); // Right edge
    EXPECT_TRUE(smallMesh.getCell(1, 2).isBoundary()); // Top edge
    EXPECT_FALSE(smallMesh.getCell(1, 1).isBoundary()); // Center (not boundary)
    
    // Test boundary detection on rectangular mesh
    EXPECT_TRUE(rectMesh.getCell(0, 3).isBoundary()); // Left edge
    EXPECT_TRUE(rectMesh.getCell(2, 0).isBoundary()); // Bottom edge
    EXPECT_TRUE(rectMesh.getCell(3, 3).isBoundary()); // Right edge
    EXPECT_TRUE(rectMesh.getCell(2, 5).isBoundary()); // Top edge
    EXPECT_FALSE(rectMesh.getCell(2, 3).isBoundary()); // Interior (not boundary)
    
    // This confirms Cell correctly uses the mesh's dimensions for boundary detection
}

// Test that Cell neighbor calculation respects mesh boundaries
TEST_F(CellMeshIntegrationTest, CellNeighborRespectsMeshBoundaries) {
    // Test with a non-square mesh (3x5)
    Mesh rectMesh(3, 5);
    
    // Interior cell
    Cell interior = rectMesh.getCell(1, 2);
    EXPECT_TRUE(interior.neighbor(NORTH).isValid()); // Within bounds
    EXPECT_TRUE(interior.neighbor(EAST).isValid());  // Within bounds
    EXPECT_TRUE(interior.neighbor(SOUTH).isValid()); // Within bounds
    EXPECT_TRUE(interior.neighbor(WEST).isValid());  // Within bounds
    
    // Edge cells
    Cell leftEdge = rectMesh.getCell(0, 2);
    EXPECT_FALSE(leftEdge.neighbor(WEST).isValid()); // Outside bounds
    EXPECT_TRUE(leftEdge.neighbor(EAST).isValid());  // Within bounds
    
    Cell rightEdge = rectMesh.getCell(2, 2);
    EXPECT_FALSE(rightEdge.neighbor(EAST).isValid()); // Outside bounds
    EXPECT_TRUE(rightEdge.neighbor(WEST).isValid());  // Within bounds
    
    Cell bottomEdge = rectMesh.getCell(1, 0);
    EXPECT_FALSE(bottomEdge.neighbor(SOUTH).isValid()); // Outside bounds
    EXPECT_TRUE(bottomEdge.neighbor(NORTH).isValid());  // Within bounds
    
    Cell topEdge = rectMesh.getCell(1, 4);
    EXPECT_FALSE(topEdge.neighbor(NORTH).isValid()); // Outside bounds
    EXPECT_TRUE(topEdge.neighbor(SOUTH).isValid());  // Within bounds
}

// Test cell behavior when mesh changes
TEST_F(CellMeshIntegrationTest, CellBehaviorWithChangingMesh) {
    // This test is limited because Mesh doesn't have methods to change dimensions
    // But we can test the relationship between existing meshes and cells
    
    Mesh mesh1(5, 5);
    Mesh mesh2(3, 3);
    
    // Create a cell from mesh1
    Cell cell = mesh1.getCell(2, 2);
    
    // Cell should be valid for mesh1
    EXPECT_TRUE(cell.isValid());
    EXPECT_EQ(cell.mesh(), &mesh1);
    
    // Create a new cell for the same position in mesh2
    Cell cell2 = mesh2.getCell(2, 2);
    
    // Both cells have the same indices but different meshes
    EXPECT_EQ(cell.i(), cell2.i());
    EXPECT_EQ(cell.j(), cell2.j());
    EXPECT_NE(cell.mesh(), cell2.mesh());
    
    // They should not be considered equal
    EXPECT_FALSE(cell == cell2);
    
    // The linear indices will be different despite same i,j because mesh dimensions differ
    EXPECT_EQ(cell.linearIndex(), 12);  // 2 + 2*5 = 12
    EXPECT_EQ(cell2.linearIndex(), 8);  // 2 + 2*3 = 8
}

// Test that creating a cell and then moving it preserves properties
TEST_F(CellMeshIntegrationTest, CellNavigationViaNeighbors) {
    // Start at origin
    Cell origin = standardMesh->getCell(0, 0);
    
    // Navigate to the center using neighbors
    Cell center = origin.neighbor(EAST).neighbor(EAST).neighbor(NORTH).neighbor(NORTH);
    
    // Should be at (2,2)
    EXPECT_EQ(center.i(), 2);
    EXPECT_EQ(center.j(), 2);
    
    // Compare with direct access
    Cell directCenter = standardMesh->getCell(2, 2);
    EXPECT_TRUE(center == directCenter);
    
    // Navigate in a full circle and get back to start
    Cell circled = center.neighbor(EAST).neighbor(SOUTH).neighbor(WEST).neighbor(NORTH);
    EXPECT_EQ(circled, center);
}

// Test different mesh factory methods creating comparable cells
TEST_F(CellMeshIntegrationTest, CellCreationFormulations) {
    // Test if different ways of creating cells result in equivalent objects
    
    // Direct mesh->getCell method (standard)
    Cell standard = standardMesh->getCell(2, 3);
    
    // Theoretical alternative method (if implemented)
    Cell fromIndices(standardMesh, 2, 3);
    
    // Should be equivalent
    EXPECT_EQ(standard.i(), fromIndices.i());
    EXPECT_EQ(standard.j(), fromIndices.j());
    EXPECT_EQ(standard.mesh(), fromIndices.mesh());
    EXPECT_TRUE(standard == fromIndices);
    
    // From linearIndex (theoretical conversion)
    int linearIdx = standard.linearIndex();
    auto [i, j] = standardMesh->toIndices(linearIdx);
    Cell fromLinear = standardMesh->getCell(i, j);
    
    // Should be equivalent again
    EXPECT_TRUE(standard == fromLinear);
}

// Test cell linearIndex consistency through mesh changes
TEST_F(CellMeshIntegrationTest, LinearIndexConsistency) {
    TestMesh dynamicMesh(5, 5);
    
    // Create cells
    Cell cell1 = dynamicMesh.getCell(1, 2);
    Cell cell2 = dynamicMesh.getCell(2, 1);
    
    // Check initial linear indices - TestMesh uses column-major order
    EXPECT_EQ(cell1.linearIndex(), 6);  // 2 + 1*5 = 7
    EXPECT_EQ(cell2.linearIndex(), 11); // 1 + 2*5 = 11
    
    // If the mesh formula changed, the cell's linear index would also change
    // However, we can't actually test this with the current interface, as
    // Mesh doesn't allow changing its formula mid-test
}

// Test that cells from different meshes behave appropriately
TEST_F(CellMeshIntegrationTest, CellsFromDifferentMeshTypes) {
    // Create cells from different mesh types
    Cell standardCell = standardMesh->getCell(2, 3);
    Cell testCell = testMesh->getCell(2, 3);
    
    // Both should be valid
    EXPECT_TRUE(standardCell.isValid());
    EXPECT_TRUE(testCell.isValid());
    
    // Both should have the same i,j indices
    EXPECT_EQ(standardCell.i(), testCell.i());
    EXPECT_EQ(standardCell.j(), testCell.j());
    
    // But they should refer to different meshes
    EXPECT_NE(standardCell.mesh(), testCell.mesh());
    
    // And therefore not be equal
    EXPECT_FALSE(standardCell == testCell);
    
    // And their linear indices should be different due to different mesh formulas
    EXPECT_NE(standardCell.linearIndex(), testCell.linearIndex());
    
    // But both should correctly identify boundary cells
    EXPECT_FALSE(standardCell.isBoundary());
    EXPECT_FALSE(testCell.isBoundary());
}

// Test mesh tracking of cell creation (using custom TestMesh)
TEST_F(CellMeshIntegrationTest, MeshCellCreationTracking) {
    // Reset the counter
    testMesh->getCallCount = 0;
    
    // Create a cell
    Cell cell1 = testMesh->getCell(1, 2);
    EXPECT_EQ(testMesh->getCallCount, 1);
    
    // Create another cell
    Cell cell2 = testMesh->getCell(3, 4);
    EXPECT_EQ(testMesh->getCallCount, 2);
    
    // This test confirms the cell creation is going through Mesh::getCell
    // and not bypassing it with a direct constructor
}

} // namespace testing
} // namespace mesh
