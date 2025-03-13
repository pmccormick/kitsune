/**
 * @file MeshIndexTests.cpp
 * @brief Thorough tests for the Mesh class index operations
 * 
 * These tests focus specifically on the index conversion methods
 * and related operations in the Mesh class, including linearIndex,
 * toIndices, and boundary cases.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include <vector>
#include <stdexcept>
#include <limits>

namespace mesh {
namespace testing {

// Test suite for Mesh index operations
class MeshIndexTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create meshes of different sizes for testing
        smallSquareMesh = new Mesh(5, 5);
        rectangularMesh = new Mesh(10, 5);
        singleCellMesh = new Mesh(1, 1);
    }

    void TearDown() override {
        delete smallSquareMesh;
        delete rectangularMesh;
        delete singleCellMesh;
    }

    Mesh* smallSquareMesh;
    Mesh* rectangularMesh;
    Mesh* singleCellMesh;
};

// Test linearIndex calculations for different mesh sizes and positions
TEST_F(MeshIndexTest, LinearIndexCalculations) {
    // Small square mesh (5x5)
    // Origin (top-left)
    EXPECT_EQ(smallSquareMesh->linearIndex(0, 0), 0);
    
    // First row
    EXPECT_EQ(smallSquareMesh->linearIndex(1, 0), 1);
    EXPECT_EQ(smallSquareMesh->linearIndex(4, 0), 4);
    
    // First column
    EXPECT_EQ(smallSquareMesh->linearIndex(0, 1), 5);
    EXPECT_EQ(smallSquareMesh->linearIndex(0, 4), 20);
    
    // Interior
    EXPECT_EQ(smallSquareMesh->linearIndex(2, 3), 17); // 2 + 3*5
    
    // Last cell (bottom-right)
    EXPECT_EQ(smallSquareMesh->linearIndex(4, 4), 24); // 4 + 4*5
    
    // Rectangular mesh (10x5)
    // Origin
    EXPECT_EQ(rectangularMesh->linearIndex(0, 0), 0);
    
    // First row
    EXPECT_EQ(rectangularMesh->linearIndex(9, 0), 9);
    
    // First column
    EXPECT_EQ(rectangularMesh->linearIndex(0, 4), 40);
    
    // Interior
    EXPECT_EQ(rectangularMesh->linearIndex(5, 2), 25); // 5 + 2*10
    
    // Last cell
    EXPECT_EQ(rectangularMesh->linearIndex(9, 4), 49); // 9 + 4*10
    
    // Single cell mesh (1x1)
    EXPECT_EQ(singleCellMesh->linearIndex(0, 0), 0);
}

// Test toIndices calculations for different mesh sizes and positions
TEST_F(MeshIndexTest, ToIndicesCalculations) {
    // Small square mesh (5x5)
    // Origin
    auto indices0 = smallSquareMesh->toIndices(0);
    EXPECT_EQ(indices0.first, 0);
    EXPECT_EQ(indices0.second, 0);
    
    // First row
    auto indices1 = smallSquareMesh->toIndices(1);
    EXPECT_EQ(indices1.first, 1);
    EXPECT_EQ(indices1.second, 0);
    
    auto indices4 = smallSquareMesh->toIndices(4);
    EXPECT_EQ(indices4.first, 4);
    EXPECT_EQ(indices4.second, 0);
    
    // First column
    auto indices5 = smallSquareMesh->toIndices(5);
    EXPECT_EQ(indices5.first, 0);
    EXPECT_EQ(indices5.second, 1);
    
    auto indices20 = smallSquareMesh->toIndices(20);
    EXPECT_EQ(indices20.first, 0);
    EXPECT_EQ(indices20.second, 4);
    
    // Interior
    auto indices17 = smallSquareMesh->toIndices(17);
    EXPECT_EQ(indices17.first, 2);
    EXPECT_EQ(indices17.second, 3);
    
    // Last cell
    auto indices24 = smallSquareMesh->toIndices(24);
    EXPECT_EQ(indices24.first, 4);
    EXPECT_EQ(indices24.second, 4);
    
    // Rectangular mesh (10x5)
    // Origin
    auto rectIndices0 = rectangularMesh->toIndices(0);
    EXPECT_EQ(rectIndices0.first, 0);
    EXPECT_EQ(rectIndices0.second, 0);
    
    // Mid-row
    auto rectIndices7 = rectangularMesh->toIndices(7);
    EXPECT_EQ(rectIndices7.first, 7);
    EXPECT_EQ(rectIndices7.second, 0);
    
    // Row start
    auto rectIndices30 = rectangularMesh->toIndices(30);
    EXPECT_EQ(rectIndices30.first, 0);
    EXPECT_EQ(rectIndices30.second, 3);
    
    // Interior
    auto rectIndices25 = rectangularMesh->toIndices(25);
    EXPECT_EQ(rectIndices25.first, 5);
    EXPECT_EQ(rectIndices25.second, 2);
    
    // Last cell
    auto rectIndices49 = rectangularMesh->toIndices(49);
    EXPECT_EQ(rectIndices49.first, 9);
    EXPECT_EQ(rectIndices49.second, 4);
    
    // Single cell mesh (1x1)
    auto singleIndices0 = singleCellMesh->toIndices(0);
    EXPECT_EQ(singleIndices0.first, 0);
    EXPECT_EQ(singleIndices0.second, 0);
}

// Test out-of-bounds handling for linearIndex
TEST_F(MeshIndexTest, LinearIndexOutOfBounds) {
    // Small square mesh (5x5)
    
    // Out-of-bounds on x
    EXPECT_THROW(smallSquareMesh->linearIndex(5, 0), std::out_of_range);
    EXPECT_THROW(smallSquareMesh->linearIndex(100, 0), std::out_of_range);
    
    // Out-of-bounds on y
    EXPECT_THROW(smallSquareMesh->linearIndex(0, 5), std::out_of_range);
    EXPECT_THROW(smallSquareMesh->linearIndex(0, 100), std::out_of_range);
    
    // Out-of-bounds on both
    EXPECT_THROW(smallSquareMesh->linearIndex(5, 5), std::out_of_range);
    EXPECT_THROW(smallSquareMesh->linearIndex(100, 100), std::out_of_range);
    
    // Rectangular mesh (10x5)
    EXPECT_THROW(rectangularMesh->linearIndex(10, 0), std::out_of_range);
    EXPECT_THROW(rectangularMesh->linearIndex(0, 5), std::out_of_range);
    
    // Single cell mesh (1x1)
    EXPECT_THROW(singleCellMesh->linearIndex(1, 0), std::out_of_range);
    EXPECT_THROW(singleCellMesh->linearIndex(0, 1), std::out_of_range);
}

// Test out-of-bounds handling for toIndices
TEST_F(MeshIndexTest, ToIndicesOutOfBounds) {
    // Small square mesh (5x5)
    EXPECT_THROW(smallSquareMesh->toIndices(25), std::out_of_range); // Just beyond end
    EXPECT_THROW(smallSquareMesh->toIndices(100), std::out_of_range); // Far beyond end
    
    // Rectangular mesh (10x5)
    EXPECT_THROW(rectangularMesh->toIndices(50), std::out_of_range); // Just beyond end
    EXPECT_THROW(rectangularMesh->toIndices(100), std::out_of_range); // Far beyond end
    
    // Single cell mesh (1x1)
    EXPECT_THROW(singleCellMesh->toIndices(1), std::out_of_range); // Just beyond end
    EXPECT_THROW(singleCellMesh->toIndices(10), std::out_of_range); // Far beyond end
    
    // Test with extremely large values
    EXPECT_THROW(smallSquareMesh->toIndices(std::numeric_limits<uint32_t>::max()), 
                 std::out_of_range);
}

// Test roundtrip conversion between indices and linear index
TEST_F(MeshIndexTest, IndexRoundtripConversion) {
    // Test with small square mesh (5x5)
    for (uint32_t j = 0; j < smallSquareMesh->ny(); j++) {
        for (uint32_t i = 0; i < smallSquareMesh->nx(); i++) {
            // Convert to linear index
            uint32_t linear = smallSquareMesh->linearIndex(i, j);
            
            // Convert back to (i,j)
            auto [i2, j2] = smallSquareMesh->toIndices(linear);
            
            // Should match original indices
            EXPECT_EQ(i, i2);
            EXPECT_EQ(j, j2);
        }
    }
    
    // Test with rectangular mesh (10x5)
    for (uint32_t j = 0; j < rectangularMesh->ny(); j++) {
        for (uint32_t i = 0; i < rectangularMesh->nx(); i++) {
            // Convert to linear index
            uint32_t linear = rectangularMesh->linearIndex(i, j);
            
            // Convert back to (i,j)
            auto [i2, j2] = rectangularMesh->toIndices(linear);
            
            // Should match original indices
            EXPECT_EQ(i, i2);
            EXPECT_EQ(j, j2);
        }
    }
    
    // Test with single cell mesh (1x1)
    {
        // Convert to linear index
        uint32_t linear = singleCellMesh->linearIndex(0, 0);
        
        // Convert back to (i,j)
        auto [i2, j2] = singleCellMesh->toIndices(linear);
        
        // Should match original indices
        EXPECT_EQ(0u, i2);
        EXPECT_EQ(0u, j2);
    }
}

// Test row-major indexing pattern
TEST_F(MeshIndexTest, RowMajorIndexingPattern) {
    // Small square mesh (5x5)
    // The linear index should increase by 1 as i increases (row-major order)
    for (uint32_t j = 0; j < smallSquareMesh->ny(); j++) {
        for (uint32_t i = 0; i < smallSquareMesh->nx() - 1; i++) {
            uint32_t current = smallSquareMesh->linearIndex(i, j);
            uint32_t next = smallSquareMesh->linearIndex(i + 1, j);
            EXPECT_EQ(next, current + 1);
        }
    }
    
    // The linear index should increase by nx as j increases
    for (uint32_t j = 0; j < smallSquareMesh->ny() - 1; j++) {
        for (uint32_t i = 0; i < smallSquareMesh->nx(); i++) {
            uint32_t current = smallSquareMesh->linearIndex(i, j);
            uint32_t next = smallSquareMesh->linearIndex(i, j + 1);
            EXPECT_EQ(next, current + smallSquareMesh->nx());
        }
    }
}

// Test isValidIndex method with valid and invalid indices
TEST_F(MeshIndexTest, IsValidIndexFunction) {
    // Test with small square mesh (5x5)
    
    // Valid indices
    EXPECT_TRUE(smallSquareMesh->isValidIndex(0, 0));
    EXPECT_TRUE(smallSquareMesh->isValidIndex(4, 4));
    EXPECT_TRUE(smallSquareMesh->isValidIndex(2, 3));
    
    // Invalid indices
    EXPECT_FALSE(smallSquareMesh->isValidIndex(5, 0));
    EXPECT_FALSE(smallSquareMesh->isValidIndex(0, 5));
    EXPECT_FALSE(smallSquareMesh->isValidIndex(5, 5));
    EXPECT_FALSE(smallSquareMesh->isValidIndex(100, 100));
    
    // Test with rectangular mesh (10x5)
    
    // Valid indices
    EXPECT_TRUE(rectangularMesh->isValidIndex(0, 0));
    EXPECT_TRUE(rectangularMesh->isValidIndex(9, 4));
    EXPECT_TRUE(rectangularMesh->isValidIndex(5, 2));
    
    // Invalid indices
    EXPECT_FALSE(rectangularMesh->isValidIndex(10, 0));
    EXPECT_FALSE(rectangularMesh->isValidIndex(0, 5));
    EXPECT_FALSE(rectangularMesh->isValidIndex(10, 5));
    
    // Test with single cell mesh (1x1)
    
    // Valid index
    EXPECT_TRUE(singleCellMesh->isValidIndex(0, 0));
    
    // Invalid indices
    EXPECT_FALSE(singleCellMesh->isValidIndex(1, 0));
    EXPECT_FALSE(singleCellMesh->isValidIndex(0, 1));
    EXPECT_FALSE(singleCellMesh->isValidIndex(1, 1));
}

// Test size computations for different mesh dimensions
TEST_F(MeshIndexTest, MeshSizeCalculation) {
    // Test size of small square mesh (5x5)
    EXPECT_EQ(smallSquareMesh->size(), 25);
    
    // Test size of rectangular mesh (10x5)
    EXPECT_EQ(rectangularMesh->size(), 50);
    
    // Test size of single cell mesh (1x1)
    EXPECT_EQ(singleCellMesh->size(), 1);
    
    // Create and test additional mesh sizes
    Mesh rowMesh(100, 1);
    EXPECT_EQ(rowMesh.size(), 100);
    
    Mesh columnMesh(1, 100);
    EXPECT_EQ(columnMesh.size(), 100);
    
    // Test a large mesh (limited by available memory in actual usage)
    Mesh largeMesh(1000, 1000);
    EXPECT_EQ(largeMesh.size(), 1000000);
}

// Test behavior with maximum allowed indices
TEST_F(MeshIndexTest, MaximumIndicesBehavior) {
    // Create a mesh of maximum size that won't exhaust memory
    // For testing, we'll use a more reasonable size that still exercises the logic
    const uint32_t MAX_TEST_SIZE = 1000;
    Mesh maxMesh(MAX_TEST_SIZE, MAX_TEST_SIZE);
    
    // Test linearIndex at the limits
    EXPECT_EQ(maxMesh.linearIndex(0, 0), 0);
    EXPECT_EQ(maxMesh.linearIndex(MAX_TEST_SIZE - 1, 0), MAX_TEST_SIZE - 1);
    EXPECT_EQ(maxMesh.linearIndex(0, MAX_TEST_SIZE - 1), (MAX_TEST_SIZE - 1) * MAX_TEST_SIZE);
    EXPECT_EQ(maxMesh.linearIndex(MAX_TEST_SIZE - 1, MAX_TEST_SIZE - 1), 
              (MAX_TEST_SIZE * MAX_TEST_SIZE) - 1);
    
    // Test toIndices at the limits
    auto indices0 = maxMesh.toIndices(0);
    EXPECT_EQ(indices0.first, 0);
    EXPECT_EQ(indices0.second, 0);
    
    auto indicesMax = maxMesh.toIndices((MAX_TEST_SIZE * MAX_TEST_SIZE) - 1);
    EXPECT_EQ(indicesMax.first, MAX_TEST_SIZE - 1);
    EXPECT_EQ(indicesMax.second, MAX_TEST_SIZE - 1);
    
    // Test out-of-bounds
    EXPECT_THROW(maxMesh.linearIndex(MAX_TEST_SIZE, 0), std::out_of_range);
    EXPECT_THROW(maxMesh.linearIndex(0, MAX_TEST_SIZE), std::out_of_range);
    EXPECT_THROW(maxMesh.toIndices(MAX_TEST_SIZE * MAX_TEST_SIZE), std::out_of_range);
}

// Test consecutive linear indices for adjacent cells
TEST_F(MeshIndexTest, ConsecutiveIndicesForAdjacentCells) {
    // For a small square mesh (5x5)
    
    // Test that cells in the same row have consecutive indices
    for (uint32_t j = 0; j < smallSquareMesh->ny(); j++) {
        for (uint32_t i = 0; i < smallSquareMesh->nx() - 1; i++) {
            uint32_t currIdx = smallSquareMesh->linearIndex(i, j);
            uint32_t nextIdx = smallSquareMesh->linearIndex(i + 1, j);
            EXPECT_EQ(nextIdx, currIdx + 1) << "Failed at i=" << i << ", j=" << j;
        }
    }
    
    // Test cells in adjacent rows
    for (uint32_t j = 0; j < smallSquareMesh->ny() - 1; j++) {
        uint32_t lastInRow = smallSquareMesh->linearIndex(smallSquareMesh->nx() - 1, j);
        uint32_t firstInNextRow = smallSquareMesh->linearIndex(0, j + 1);
        EXPECT_EQ(firstInNextRow, lastInRow + 1) 
            << "Failed between rows " << j << " and " << (j + 1);
    }
}

// Test index consistency with cell positions (spatial continuity)
TEST_F(MeshIndexTest, IndexConsistencyWithCellPositions) {
    // For a square mesh, cells that are adjacent in grid coordinates
    // should have indices that differ by 1 (if in same row) or by nx (if in same column)
    
    // Same row adjacency
    for (uint32_t j = 0; j < smallSquareMesh->ny(); j++) {
        for (uint32_t i = 0; i < smallSquareMesh->nx() - 1; i++) {
            uint32_t left = smallSquareMesh->linearIndex(i, j);
            uint32_t right = smallSquareMesh->linearIndex(i + 1, j);
            EXPECT_EQ(right - left, 1);
        }
    }
    
    // Same column adjacency
    for (uint32_t i = 0; i < smallSquareMesh->nx(); i++) {
        for (uint32_t j = 0; j < smallSquareMesh->ny() - 1; j++) {
            uint32_t bottom = smallSquareMesh->linearIndex(i, j);
            uint32_t top = smallSquareMesh->linearIndex(i, j + 1);
            EXPECT_EQ(top - bottom, smallSquareMesh->nx());
        }
    }
}

} // namespace testing
} // namespace mesh
