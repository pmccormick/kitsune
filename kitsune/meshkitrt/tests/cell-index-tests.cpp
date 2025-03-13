/**
 * @file CellIndexTests.cpp
 * @brief Tests for Cell linear index computation
 * 
 * These tests focus on the linearIndex() method and its interaction
 * with the Mesh class.
 */

#include <gtest/gtest.h>
#include "Cell.h"
#include "Mesh.h"

namespace mesh {
namespace testing {

// Fixture for Cell index conversion tests
class CellIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 5x5 test mesh for index conversion tests
        smallMesh = new Mesh(5, 5);
        
        // Create a larger mesh for more comprehensive tests
        largeMesh = new Mesh(10, 8);
    }

    void TearDown() override {
        delete smallMesh;
        delete largeMesh;
    }

    Mesh* smallMesh;
    Mesh* largeMesh;
};

// Test linearIndex for various cell positions
TEST_F(CellIndexTest, LinearIndexComputation) {
    // Test cells at different positions in the small mesh
    
    // Origin (0,0)
    Cell origin = smallMesh->getCell(0, 0);
    EXPECT_EQ(origin.linearIndex(), 0);
    
    // Different positions
    Cell pos1 = smallMesh->getCell(1, 0);
    EXPECT_EQ(pos1.linearIndex(), 1);
    
    Cell pos2 = smallMesh->getCell(0, 1);
    EXPECT_EQ(pos2.linearIndex(), 5); // 0 + 1*5
    
    Cell pos3 = smallMesh->getCell(2, 3);
    EXPECT_EQ(pos3.linearIndex(), 17); // 2 + 3*5
    
    // Last cell
    Cell lastCell = smallMesh->getCell(4, 4);
    EXPECT_EQ(lastCell.linearIndex(), 24); // 4 + 4*5
}

// Test linearIndex for cells in the larger mesh
TEST_F(CellIndexTest, LinearIndexLargerMesh) {
    // Test a few positions in the larger mesh to ensure
    // calculations scale correctly
    
    // Origin in larger mesh
    Cell origin = largeMesh->getCell(0, 0);
    EXPECT_EQ(origin.linearIndex(), 0);
    
    // First row
    Cell row0col5 = largeMesh->getCell(5, 0);
    EXPECT_EQ(row0col5.linearIndex(), 5);
    
    // Interior cell
    Cell interior = largeMesh->getCell(3, 4);
    EXPECT_EQ(interior.linearIndex(), 43); // 3 + 4*10
    
    // Cell on last row
    Cell lastRow = largeMesh->getCell(7, 7);
    EXPECT_EQ(lastRow.linearIndex(), 77); // 7 + 7*10
    
    // Last cell
    Cell lastCell = largeMesh->getCell(9, 7);
    EXPECT_EQ(lastCell.linearIndex(), 79); // 9 + 7*10
}

// Test linearIndex for a single-cell mesh
TEST_F(CellIndexTest, LinearIndexSingleCellMesh) {
    Mesh singleCellMesh(1, 1);
    Cell onlyCell = singleCellMesh.getCell(0, 0);
    
    EXPECT_EQ(onlyCell.linearIndex(), 0);
}

// Test linearIndex for a non-square mesh
TEST_F(CellIndexTest, LinearIndexNonSquareMesh) {
    Mesh rectMesh(3, 5); // 3 columns, 5 rows
    
    // First row cells
    EXPECT_EQ(rectMesh.getCell(0, 0).linearIndex(), 0);
    EXPECT_EQ(rectMesh.getCell(1, 0).linearIndex(), 1);
    EXPECT_EQ(rectMesh.getCell(2, 0).linearIndex(), 2);
    
    // First column cells
    EXPECT_EQ(rectMesh.getCell(0, 1).linearIndex(), 3);  // 0 + 1*3
    EXPECT_EQ(rectMesh.getCell(0, 2).linearIndex(), 6);  // 0 + 2*3
    EXPECT_EQ(rectMesh.getCell(0, 3).linearIndex(), 9);  // 0 + 3*3
    EXPECT_EQ(rectMesh.getCell(0, 4).linearIndex(), 12); // 0 + 4*3
    
    // Last cell
    EXPECT_EQ(rectMesh.getCell(2, 4).linearIndex(), 14); // 2 + 4*3
}

// Test linearIndex for invalid cells
TEST_F(CellIndexTest, LinearIndexInvalidCell) {
    // Create an invalid cell with null mesh
    Cell invalidCell;
    
    // linearIndex should throw for invalid cells
    EXPECT_THROW(invalidCell.linearIndex(), std::logic_error);
    
    // Also test for a cell with invalid indices but valid mesh
    Cell outOfBoundsCell(smallMesh, 10, 10);
    EXPECT_THROW(outOfBoundsCell.linearIndex(), std::logic_error);
}

// Test linearIndex consistency with Mesh::linearIndex
TEST_F(CellIndexTest, LinearIndexConsistencyWithMesh) {
    // Test that Cell::linearIndex matches Mesh::linearIndex
    // for the same coordinates
    
    for (uint32_t j = 0; j < smallMesh->ny(); j++) {
        for (uint32_t i = 0; i < smallMesh->nx(); i++) {
            Cell cell = smallMesh->getCell(i, j);
            EXPECT_EQ(cell.linearIndex(), static_cast<int>(smallMesh->linearIndex(i, j)));
        }
    }
}

// Test linearIndex error propagation from mesh
TEST_F(CellIndexTest, LinearIndexErrorPropagation) {
    // Create a custom Mesh class that throws an exception for linearIndex
    class ThrowingMesh : public Mesh {
    public:
        ThrowingMesh() : Mesh(5, 5) {}
        
        uint32_t linearIndex(uint32_t i, uint32_t j) const override {
            throw std::runtime_error("Simulated error in Mesh::linearIndex");
        }
    };
    
    ThrowingMesh throwingMesh;
    Cell cell = throwingMesh.getCell(2, 2);
    
    // The Cell should propagate the error from the Mesh
    EXPECT_THROW(cell.linearIndex(), std::logic_error);
}

// Test linearIndex for cells at the mesh size limit
TEST_F(CellIndexTest, LinearIndexAtMaximumSize) {
    // Create a mesh with the maximum theoretical size
    // that would still have a valid linearIndex
    Mesh maxMesh(65535, 65535); // Theoretical large mesh (in practice we'd hit memory limits first)
    
    // Test a few points to ensure consistency
    Cell origin = maxMesh.getCell(0, 0);
    EXPECT_EQ(origin.linearIndex(), 0);
    
    Cell smallOffset = maxMesh.getCell(1, 1);
    EXPECT_EQ(smallOffset.linearIndex(), 65536); // 1 + 1*65535
    
    // The actual maximum valid index would be (65534, 65534), but we don't test it
    // because it would exceed the range of a 32-bit integer
}

} // namespace testing
} // namespace mesh
