/**
 * @file MeshBasicTests.cpp
 * @brief Basic unit tests for the Mesh class
 * 
 * These tests focus on mesh construction, dimension validation,
 * and other fundamental operations of the Mesh class.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include <stdexcept>

namespace mesh {
namespace testing {

// Test suite for basic Mesh functionality
class MeshBasicTest : public ::testing::Test {
protected:
    // No setup needed for most tests
};

// Test valid mesh construction with different dimensions
TEST_F(MeshBasicTest, ConstructionWithValidDimensions) {
    // Test small mesh
    Mesh smallMesh(5, 5);
    EXPECT_EQ(smallMesh.nx(), 5);
    EXPECT_EQ(smallMesh.ny(), 5);
    EXPECT_EQ(smallMesh.size(), 25);
    
    // Test larger mesh
    Mesh largeMesh(100, 200);
    EXPECT_EQ(largeMesh.nx(), 100);
    EXPECT_EQ(largeMesh.ny(), 200);
    EXPECT_EQ(largeMesh.size(), 20000);
    
    // Test non-square mesh
    Mesh rectangularMesh(10, 20);
    EXPECT_EQ(rectangularMesh.nx(), 10);
    EXPECT_EQ(rectangularMesh.ny(), 20);
    EXPECT_EQ(rectangularMesh.size(), 200);
    
    // Test minimal valid mesh (1x1)
    Mesh minimalMesh(1, 1);
    EXPECT_EQ(minimalMesh.nx(), 1);
    EXPECT_EQ(minimalMesh.ny(), 1);
    EXPECT_EQ(minimalMesh.size(), 1);
}

// Test that invalid mesh dimensions throw exceptions
TEST_F(MeshBasicTest, ConstructionWithInvalidDimensions) {
    // Test with zero width
    EXPECT_THROW(Mesh zeroWidthMesh(0, 5), std::invalid_argument);
    
    // Test with zero height
    EXPECT_THROW(Mesh zeroHeightMesh(5, 0), std::invalid_argument);
    
    // Test with both dimensions zero
    EXPECT_THROW(Mesh zeroMesh(0, 0), std::invalid_argument);
    
    // Note: Testing with negative dimensions is unnecessary 
    // since the parameters are unsigned integers
}

// Test linearIndex calculation
TEST_F(MeshBasicTest, LinearIndexCalculation) {
    Mesh testMesh(5, 5);
    
    // Test various positions
    EXPECT_EQ(testMesh.linearIndex(0, 0), 0);
    EXPECT_EQ(testMesh.linearIndex(1, 0), 1);
    EXPECT_EQ(testMesh.linearIndex(0, 1), 5);
    EXPECT_EQ(testMesh.linearIndex(2, 3), 17); // 2 + 3*5
    EXPECT_EQ(testMesh.linearIndex(4, 4), 24); // 4 + 4*5
    
    // Test with non-square mesh
    Mesh rectMesh(3, 4);
    EXPECT_EQ(rectMesh.linearIndex(0, 0), 0);
    EXPECT_EQ(rectMesh.linearIndex(2, 0), 2);
    EXPECT_EQ(rectMesh.linearIndex(0, 3), 9);  // 0 + 3*3
    EXPECT_EQ(rectMesh.linearIndex(2, 3), 11); // 2 + 3*3
}

// Test linearIndex validation
TEST_F(MeshBasicTest, LinearIndexValidation) {
    Mesh testMesh(5, 5);
    
    // Test valid indices
    EXPECT_NO_THROW(testMesh.linearIndex(0, 0));
    EXPECT_NO_THROW(testMesh.linearIndex(4, 4));
    
    // Test out-of-bounds indices
    EXPECT_THROW(testMesh.linearIndex(5, 0), std::out_of_range);
    EXPECT_THROW(testMesh.linearIndex(0, 5), std::out_of_range);
    EXPECT_THROW(testMesh.linearIndex(5, 5), std::out_of_range);
}

// Test toIndices (inverse of linearIndex)
TEST_F(MeshBasicTest, ToIndicesCalculation) {
    Mesh testMesh(5, 5);
    
    // Test various linear indices
    auto indices0 = testMesh.toIndices(0);
    EXPECT_EQ(indices0.first, 0);
    EXPECT_EQ(indices0.second, 0);
    
    auto indices1 = testMesh.toIndices(1);
    EXPECT_EQ(indices1.first, 1);
    EXPECT_EQ(indices1.second, 0);
    
    auto indices5 = testMesh.toIndices(5);
    EXPECT_EQ(indices5.first, 0);
    EXPECT_EQ(indices5.second, 1);
    
    auto indices17 = testMesh.toIndices(17);
    EXPECT_EQ(indices17.first, 2);
    EXPECT_EQ(indices17.second, 3);
    
    auto indices24 = testMesh.toIndices(24);
    EXPECT_EQ(indices24.first, 4);
    EXPECT_EQ(indices24.second, 4);
}

// Test toIndices validation
TEST_F(MeshBasicTest, ToIndicesValidation) {
    Mesh testMesh(5, 5);
    
    // Test valid linear indices
    EXPECT_NO_THROW(testMesh.toIndices(0));
    EXPECT_NO_THROW(testMesh.toIndices(24));
    
    // Test out-of-bounds linear index
    EXPECT_THROW(testMesh.toIndices(25), std::out_of_range);
    EXPECT_THROW(testMesh.toIndices(100), std::out_of_range);
}

// Test roundtrip conversion between (i,j) and linear indices
TEST_F(MeshBasicTest, RoundtripConversion) {
    Mesh testMesh(5, 5);
    
    // Test roundtrip for several positions
    for (uint32_t j = 0; j < testMesh.ny(); j++) {
        for (uint32_t i = 0; i < testMesh.nx(); i++) {
            // Convert to linear index
            uint32_t linear = testMesh.linearIndex(i, j);
            
            // Convert back to (i,j)
            auto [i2, j2] = testMesh.toIndices(linear);
            
            // Should get back the original indices
            EXPECT_EQ(i, i2);
            EXPECT_EQ(j, j2);
        }
    }
}

// Test isValidIndex method
TEST_F(MeshBasicTest, IsValidIndex) {
    Mesh testMesh(5, 5);
    
    // Test valid indices
    EXPECT_TRUE(testMesh.isValidIndex(0, 0));
    EXPECT_TRUE(testMesh.isValidIndex(4, 4));
    EXPECT_TRUE(testMesh.isValidIndex(2, 3));
    
    // Test invalid indices
    EXPECT_FALSE(testMesh.isValidIndex(5, 0));
    EXPECT_FALSE(testMesh.isValidIndex(0, 5));
    EXPECT_FALSE(testMesh.isValidIndex(5, 5));
    
    // Test with very large indices (well outside range)
    EXPECT_FALSE(testMesh.isValidIndex(1000, 0));
    EXPECT_FALSE(testMesh.isValidIndex(0, 1000));
}

// Test getCell method produces valid cells
TEST_F(MeshBasicTest, GetCellValidity) {
    Mesh testMesh(5, 5);
    
    // Test with valid indices
    Cell cell1 = testMesh.getCell(0, 0);
    EXPECT_TRUE(cell1.isValid());
    EXPECT_EQ(cell1.i(), 0);
    EXPECT_EQ(cell1.j(), 0);
    EXPECT_EQ(cell1.mesh(), &testMesh);
    
    Cell cell2 = testMesh.getCell(4, 4);
    EXPECT_TRUE(cell2.isValid());
    EXPECT_EQ(cell2.i(), 4);
    EXPECT_EQ(cell2.j(), 4);
    EXPECT_EQ(cell2.mesh(), &testMesh);
    
    // Note: Mesh::getCell itself doesn't verify bounds in current implementation
    // But the resulting cell should be invalid if we pass illegal coordinates
    Cell invalidCell = testMesh.getCell(10, 10);
    EXPECT_FALSE(invalidCell.isValid());
}

} // namespace testing
} // namespace mesh
