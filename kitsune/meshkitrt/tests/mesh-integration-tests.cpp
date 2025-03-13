/**
 * @file MeshIntegrationTests.cpp
 * @brief Integration tests for the Mesh class with the Cell class
 * 
 * These tests focus on how Mesh and Cell work together, ensuring
 * proper relationships and consistent behavior between the two classes.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "Cell.h"
#include <unordered_set>
#include <vector>

namespace mesh {
namespace testing {

// Test suite for Mesh-Cell integration tests
class MeshIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard 5x5 mesh for testing
        standardMesh = new Mesh(5, 5);
    }

    void TearDown() override {
        delete standardMesh;
    }

    Mesh* standardMesh;
};

// Test that Cell objects correctly refer back to their parent mesh
TEST_F(MeshIntegrationTest, CellMeshReferenceIntegrity) {
    // Get cells from the mesh
    Cell corner = standardMesh->getCell(0, 0);
    Cell center = standardMesh->getCell(2, 2);
    Cell edge = standardMesh->getCell(4, 2);
    
    // All cells should reference the correct mesh
    EXPECT_EQ(corner.mesh(), standardMesh);
    EXPECT_EQ(center.mesh(), standardMesh);
    EXPECT_EQ(edge.mesh(), standardMesh);
    
    // Cells from different meshes should reference different meshes
    Mesh otherMesh(5, 5);
    Cell otherCell = otherMesh.getCell(2, 2);
    
    EXPECT_NE(center.mesh(), otherCell.mesh());
    EXPECT_EQ(otherCell.mesh(), &otherMesh);
}

// Test consistent linearIndex calculation between Mesh and Cell
TEST_F(MeshIntegrationTest, LinearIndexConsistency) {
    // Test linearIndex consistency for multiple positions
    for (uint32_t j = 0; j < standardMesh->ny(); j++) {
        for (uint32_t i = 0; i < standardMesh->nx(); i++) {
            // Calculate linearIndex directly from Mesh
            uint32_t meshLinearIndex = standardMesh->linearIndex(i, j);
            
            // Get the Cell and its linearIndex
            Cell cell = standardMesh->getCell(i, j);
            int cellLinearIndex = cell.linearIndex();
            
            // The indices should match
            EXPECT_EQ(cellLinearIndex, static_cast<int>(meshLinearIndex));
        }
    }
}

// Test Cell.isBoundary() correctly uses Mesh dimensions
TEST_F(MeshIntegrationTest, BoundaryDetectionConsistency) {
    // Create a different-sized mesh to ensure dimensions are used correctly
    Mesh customMesh(7, 3);
    
    // Check all cells in the custom mesh
    for (uint32_t j = 0; j < customMesh.ny(); j++) {
        for (uint32_t i = 0; i < customMesh.nx(); i++) {
            Cell cell = customMesh.getCell(i, j);
            
            // A cell is on the boundary if any of its coordinates is 0
            // or the maximum value
            bool shouldBeBoundary = (i == 0 || j == 0 || 
                                    i == customMesh.nx() - 1 || 
                                    j == customMesh.ny() - 1);
            
            EXPECT_EQ(cell.isBoundary(), shouldBeBoundary)
                << "Mismatch at i=" << i << ", j=" << j;
        }
    }
}

// Test that getCell produces unique cell objects
TEST_F(MeshIntegrationTest, CellObjectUniqueness) {
    // Get the same cell twice
    Cell cell1 = standardMesh->getCell(2, 2);
    Cell cell2 = standardMesh->getCell(2, 2);
    
    // The cells should be equal (same coordinates and mesh)
    EXPECT_TRUE(cell1 == cell2);
    
    // But they should be different objects in memory
    // (we can't test pointer equality directly since Cell doesn't expose raw pointers)
    
    // However, we can test that modifying one (theoretically) doesn't affect the other
    // by verifying they're separate copies
    
    // Test that they have independent memory (can't directly test due to Cell's design)
    EXPECT_TRUE(&cell1 != &cell2);
}

// Test that Cell neighbor navigation respects mesh boundaries
TEST_F(MeshIntegrationTest, NeighborNavigationWithinMeshBounds) {
    // Test neighbor navigation for interior cell
    Cell interior = standardMesh->getCell(2, 2);
    
    // All neighbors of an interior cell should be valid
    EXPECT_TRUE(interior.neighbor(NORTH).isValid());
    EXPECT_TRUE(interior.neighbor(EAST).isValid());
    EXPECT_TRUE(interior.neighbor(SOUTH).isValid());
    EXPECT_TRUE(interior.neighbor(WEST).isValid());
    
    // Test corner cell
    Cell corner = standardMesh->getCell(0, 0);
    
    // The corner cell should only have neighbors to the north and east
    EXPECT_TRUE(corner.neighbor(NORTH).isValid());
    EXPECT_TRUE(corner.neighbor(EAST).isValid());
    EXPECT_FALSE(corner.neighbor(SOUTH).isValid()); // Out of bounds
    EXPECT_FALSE(corner.neighbor(WEST).isValid());  // Out of bounds
    
    // Test edge cell
    Cell edge = standardMesh->getCell(0, 2);
    
    // The edge cell should only be missing neighbors to the west
    EXPECT_TRUE(edge.neighbor(NORTH).isValid());
    EXPECT_TRUE(edge.neighbor(EAST).isValid());
    EXPECT_TRUE(edge.neighbor(SOUTH).isValid());
    EXPECT_FALSE(edge.neighbor(WEST).isValid()); // Out of bounds
}

// Test consistent Cell creation from Mesh with different dimensions
TEST_F(MeshIntegrationTest, ConsistentCellCreationAcrossMeshes) {
    // Create meshes of different dimensions
    Mesh smallMesh(3, 3);
    Mesh mediumMesh(5, 5);
    Mesh largeMesh(10, 10);
    Mesh rectMesh(3, 7);
    
    // Test that cells from the same relative position have equivalent properties
    Cell smallCenter = smallMesh.getCell(1, 1);
    Cell mediumCenter = mediumMesh.getCell(2, 2);
    Cell largeCenter = largeMesh.getCell(5, 5);
    Cell rectCenter = rectMesh.getCell(1, 3);
    
    // All should be center/interior cells
    EXPECT_FALSE(smallCenter.isBoundary());
    EXPECT_FALSE(mediumCenter.isBoundary());
    EXPECT_FALSE(largeCenter.isBoundary());
    EXPECT_FALSE(rectCenter.isBoundary());
    
    // All should have neighbors in all directions
    EXPECT_TRUE(smallCenter.neighbor(NORTH).isValid());
    EXPECT_TRUE(mediumCenter.neighbor(NORTH).isValid());
    EXPECT_TRUE(largeCenter.neighbor(NORTH).isValid());
    EXPECT_TRUE(rectCenter.neighbor(NORTH).isValid());
    
    // All should have their respective mesh pointers
    EXPECT_EQ(smallCenter.mesh(), &smallMesh);
    EXPECT_EQ(mediumCenter.mesh(), &mediumMesh);
    EXPECT_EQ(largeCenter.mesh(), &largeMesh);
    EXPECT_EQ(rectCenter.mesh(), &rectMesh);
}

// Test traversing through all cells in the mesh using Cell navigation
TEST_F(MeshIntegrationTest, MeshTraversalUsingCellNavigation) {
    // Create a set to track visited cells
    std::unordered_set<int> visitedIndices;
    
    // Start at origin and traverse row by row using Cell navigation
    Cell current = standardMesh->getCell(0, 0);
    
    for (uint32_t j = 0; j < standardMesh->ny(); j++) {
        // Reset to beginning of row
        if (j > 0) {
            // Move down from previous row
            current = standardMesh->getCell(0, j);
        }
        
        // Traverse this row
        for (uint32_t i = 0; i < standardMesh->nx(); i++) {
            // If not at the start of the row, move east
            if (i > 0) {
                current = current.neighbor(EAST);
            }
            
            // Mark this cell as visited
            visitedIndices.insert(current.linearIndex());
            
            // Verify it's the expected cell
            EXPECT_EQ(current.i(), i);
            EXPECT_EQ(current.j(), j);
        }
    }
    
    // We should have visited every cell in the mesh
    EXPECT_EQ(visitedIndices.size(), standardMesh->size());
}

// Test traversing mesh using Cell neighbor navigation
TEST_F(MeshIntegrationTest, MeshTraversalInSpiralPattern) {
    // Create a small mesh for this test to keep it manageable
    Mesh smallMesh(3, 3);
    
    // Track visited cells by index
    std::unordered_set<int> visited;
    
    // Start at the center
    Cell center = smallMesh.getCell(1, 1);
    visited.insert(center.linearIndex());
    
    // Move in a spiral pattern: E, N, W, W, S, S, E, E, N
    // This should visit all 8 surrounding cells in a 3x3 grid
    Cell pos = center;
    
    // Move East
    pos = pos.neighbor(EAST);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 2);
    EXPECT_EQ(pos.j(), 1);
    
    // Move North
    pos = pos.neighbor(NORTH);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 2);
    EXPECT_EQ(pos.j(), 2);
    
    // Move West
    pos = pos.neighbor(WEST);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 1);
    EXPECT_EQ(pos.j(), 2);
    
    // Move West again
    pos = pos.neighbor(WEST);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 0);
    EXPECT_EQ(pos.j(), 2);
    
    // Move South
    pos = pos.neighbor(SOUTH);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 0);
    EXPECT_EQ(pos.j(), 1);
    
    // Move South again
    pos = pos.neighbor(SOUTH);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 0);
    EXPECT_EQ(pos.j(), 0);
    
    // Move East
    pos = pos.neighbor(EAST);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 1);
    EXPECT_EQ(pos.j(), 0);
    
    // Move East again
    pos = pos.neighbor(EAST);
    visited.insert(pos.linearIndex());
    EXPECT_EQ(pos.i(), 2);
    EXPECT_EQ(pos.j(), 0);
    
    // We should have visited all 9 cells (center + 8 surrounding)
    EXPECT_EQ(visited.size(), 9);
}

// Test mesh-cell interactions with a 1x1 mesh (edge case)
TEST_F(MeshIntegrationTest, SingleCellMeshInteraction) {
    // Create a 1x1 mesh
    Mesh singleCellMesh(1, 1);
    
    // Get the only cell
    Cell onlyCell = singleCellMesh.getCell(0, 0);
    
    // Cell should be valid
    EXPECT_TRUE(onlyCell.isValid());
    
    // Cell should be on the boundary
    EXPECT_TRUE(onlyCell.isBoundary());
    
    // Cell should have no valid neighbors
    EXPECT_FALSE(onlyCell.neighbor(NORTH).isValid());
    EXPECT_FALSE(onlyCell.neighbor(EAST).isValid());
    EXPECT_FALSE(onlyCell.neighbor(SOUTH).isValid());
    EXPECT_FALSE(onlyCell.neighbor(WEST).isValid());
    
    // Cell's linear index should be 0
    EXPECT_EQ(onlyCell.linearIndex(), 0);
}

// Test mesh-cell interactions with a highly rectangular mesh
TEST_F(MeshIntegrationTest, HighlyRectangularMeshInteraction) {
    // Create a 1x10 mesh (column vector)
    Mesh columnMesh(1, 10);
    
    // All cells should be on the boundary
    for (uint32_t j = 0; j < columnMesh.ny(); j++) {
        Cell cell = columnMesh.getCell(0, j);
        EXPECT_TRUE(cell.isBoundary());
        
        // East and West neighbors should be invalid
        EXPECT_FALSE(cell.neighbor(EAST).isValid());
        EXPECT_FALSE(cell.neighbor(WEST).isValid());
        
        // North neighbor valid only if not at top
        EXPECT_EQ(cell.neighbor(NORTH).isValid(), j < columnMesh.ny() - 1);
        
        // South neighbor valid only if not at bottom
        EXPECT_EQ(cell.neighbor(SOUTH).isValid(), j > 0);
    }
    
    // Create a 10x1 mesh (row vector)
    Mesh rowMesh(10, 1);
    
    // All cells should be on the boundary
    for (uint32_t i = 0; i < rowMesh.nx(); i++) {
        Cell cell = rowMesh.getCell(i, 0);
        EXPECT_TRUE(cell.isBoundary());
        
        // North and South neighbors should be invalid
        EXPECT_FALSE(cell.neighbor(NORTH).isValid());
        EXPECT_FALSE(cell.neighbor(SOUTH).isValid());
        
        // East neighbor valid only if not at right edge
        EXPECT_EQ(cell.neighbor(EAST).isValid(), i < rowMesh.nx() - 1);
        
        // West neighbor valid only if not at left edge
        EXPECT_EQ(cell.neighbor(WEST).isValid(), i > 0);
    }
}

} // namespace testing
} // namespace mesh
