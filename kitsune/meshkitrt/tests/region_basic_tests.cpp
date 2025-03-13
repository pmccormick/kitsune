/**
 * @file region_basic_tests.cpp
 * @brief Unit tests for the Region system
 */

#include <gtest/gtest.h>
#include "RegionAll.h"
#include "./Region/MockMesh.h"
#include "./Region/MockCell.h"

namespace mesh {

/**
 * @class RegionTest
 * @brief Base fixture for Region system tests
 */
class RegionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard 10x10 mesh for testing
        mockMesh = std::make_unique<MockMesh>(10, 10);
        
        // Initialize groups for different test scenarios
        mockMesh->initializeGroups([](int i, int j) {
            // Group 1: Rectangle in top-left (0-4, 5-9)
            if (i < 5 && j >= 5) return 1;
            
            // Group 2: Rectangle in bottom-right (5-9, 0-4)
            if (i >= 5 && j < 5) return 2;
            
            // Group 3: Bottom-left square (0-4, 0-4)
            if (i < 5 && j < 5) return 3;
            
            // Group 4: Top-right square (5-9, 5-9)
            return 4;
        });
    }
    
    std::unique_ptr<MockMesh> mockMesh;
};

/**
 * @brief Test basic bit array operations
 */
TEST_F(RegionTest, BitArrayBasicOperations) {
    BitArray bits(100, false);
    
    // Test setting bits
    bits.set(5, true);
    bits.set(10, true);
    bits.set(15, true);
    
    // Test getting bits
    EXPECT_TRUE(bits.get(5));
    EXPECT_TRUE(bits.get(10));
    EXPECT_TRUE(bits.get(15));
    EXPECT_FALSE(bits.get(20));
    
    // Test bit counting
    EXPECT_EQ(bits.count(), 3);
    
    // Test clearing
    bits.set(5, false);
    EXPECT_FALSE(bits.get(5));
    EXPECT_EQ(bits.count(), 2);
}

/**
 * @brief Test bit array bitwise operations
 */
TEST_F(RegionTest, BitArrayBitwiseOperations) {
    BitArray bits1(100, false);
    BitArray bits2(100, false);
    
    // Set some bits in each array
    bits1.set(5, true);
    bits1.set(10, true);
    
    bits2.set(10, true);
    bits2.set(15, true);
    
    // Test bitwise OR
    bits1.bitwiseOr(bits2);
    EXPECT_TRUE(bits1.get(5));
    EXPECT_TRUE(bits1.get(10));
    EXPECT_TRUE(bits1.get(15));
    EXPECT_EQ(bits1.count(), 3);
    
    // Test bitwise AND
    BitArray bits3(100, false);
    bits3.set(5, true);
    bits3.set(10, true);
    
    bits3.bitwiseAnd(bits2);
    EXPECT_FALSE(bits3.get(5));
    EXPECT_TRUE(bits3.get(10));
    EXPECT_FALSE(bits3.get(15));
    EXPECT_EQ(bits3.count(), 1);
    
    // Test bitwise AND-NOT
    BitArray bits4(100, false);
    for (int i = 0; i < 20; i++) {
        bits4.set(i, true);
    }
    
    BitArray bits5(100, false);
    bits5.set(5, true);
    bits5.set(10, true);
    
    bits4.bitwiseAndNot(bits5);
    EXPECT_FALSE(bits4.get(5));
    EXPECT_FALSE(bits4.get(10));
    EXPECT_TRUE(bits4.get(15));
    EXPECT_EQ(bits4.count(), 18); // 20 - 2
}

/**
 * @brief Test region definition classes
 */
TEST_F(RegionTest, RegionDefinitions) {
    // Test rectangular region definition
    auto rectDef = std::make_shared<RectangularRegion>("TestRect", 2, 2, 7, 7);
    
    // Test cells in and out of the region
    MockCell* cellInside = mockMesh->getTypedCell(3, 3);
    MockCell* cellOutside = mockMesh->getTypedCell(0, 0);
    MockCell* cellBoundary = mockMesh->getTypedCell(2, 2);
    
    EXPECT_TRUE(rectDef->contains(cellInside));
    EXPECT_FALSE(rectDef->contains(cellOutside));
    EXPECT_TRUE(rectDef->contains(cellBoundary));
    
    // Test predicate region definition
    auto predDef = std::make_shared<PredicateRegion>(
        "TestPred",
        [](const Cell* cell) {
            // Check if cell is in the top-left quadrant
            return cell->i() < 5 && cell->j() >= 5;
        }
    );
    
    // Check top-left quadrant cells
    MockCell* topLeft = mockMesh->getTypedCell(2, 7);
    MockCell* topRight = mockMesh->getTypedCell(7, 7);
    
    EXPECT_TRUE(predDef->contains(topLeft));
    EXPECT_FALSE(predDef->contains(topRight));
    
    // Instead of directly creating a CompositeRegion, test the functionality through region operations
    auto regionA = createRectangularRegion(*mockMesh, 0, 0, 4, 4, "R1");
    auto regionB = createRectangularRegion(*mockMesh, 5, 5, 9, 9, "R2");
    
    // Test union operation
    auto unionRegion = createUnionRegion(regionA, regionB, "TestUnion");
    
    // Test cells in different parts of the union
    MockCell* bottomLeft = mockMesh->getTypedCell(2, 2);
    MockCell* topRightCell = mockMesh->getTypedCell(7, 7);
    MockCell* bottomRight = mockMesh->getTypedCell(7, 2);
    
    EXPECT_TRUE(unionRegion.contains(bottomLeft));   // In regionA
    EXPECT_TRUE(unionRegion.contains(topRightCell)); // In regionB
    EXPECT_FALSE(unionRegion.contains(bottomRight)); // In neither
}

/**
 * @brief Test region construction and basic properties
 */
TEST_F(RegionTest, RegionConstruction) {
    // Create a rectangular region
    auto region = createRectangularRegion(*mockMesh, 2, 2, 7, 7, "TestRegion");
    
    // Check basic properties
    EXPECT_EQ(region.getDefinition()->getName(), "TestRegion");
    EXPECT_EQ(region.getMeshSize(), 100);  // 10x10 mesh
    
    // Force optimization to ensure materialized storage
    region.forceOptimization();
    
    // Should no longer be in DYNAMIC mode
    EXPECT_NE(region.getStorageMode(), RegionStorageMode::DYNAMIC);
    
    // Expected size is (7-2+1) * (7-2+1) = 6 * 6 = 36
    EXPECT_EQ(region.size(), 36);
}

/**
 * @brief Test region cell containment
 */
TEST_F(RegionTest, RegionCellContainment) {
    // Create a rectangular region
    auto region = createRectangularRegion(*mockMesh, 2, 2, 7, 7, "TestRegion");
    
    // Check cells in and out of the region
    auto cellInside = mockMesh->getTypedCell(3, 3);
    auto cellOutside = mockMesh->getTypedCell(0, 0);
    
    EXPECT_TRUE(region.contains(cellInside));
    EXPECT_FALSE(region.contains(cellOutside));
    
    // Check by index
    int insideIndex = cellInside->linearIndex();
    int outsideIndex = cellOutside->linearIndex();
    
    EXPECT_TRUE(region.containsIndex(insideIndex));
    EXPECT_FALSE(region.containsIndex(outsideIndex));
}

/**
 * @brief Test region storage mode transitions
 */
TEST_F(RegionTest, RegionStorageModes) {
    // Create a rectangular region
    auto region = createRectangularRegion(*mockMesh, 2, 2, 7, 7, "TestRegion");
    region.forceOptimization();
    
    // Get initial mode
    auto initialMode = region.getStorageMode();
    
    // Switch to the other mode
    if (initialMode == RegionStorageMode::CELL_SET) {
        region.setStorageMode(RegionStorageMode::BIT_ARRAY);
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::BIT_ARRAY);
    } else {
        region.setStorageMode(RegionStorageMode::CELL_SET);
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::CELL_SET);
    }
    
    // Size should remain the same
    EXPECT_EQ(region.size(), 36);
    
    // Check that containment still works the same way
    auto cellInside = mockMesh->getTypedCell(3, 3);
    auto cellOutside = mockMesh->getTypedCell(0, 0);
    
    EXPECT_TRUE(region.contains(cellInside));
    EXPECT_FALSE(region.contains(cellOutside));
}

/**
 * @brief Test region union operation
 */
TEST_F(RegionTest, RegionUnionOperation) {
    // Create two overlapping rectangular regions
    auto regionA = createRectangularRegion(*mockMesh, 1, 1, 5, 5, "RegionA");
    auto regionB = createRectangularRegion(*mockMesh, 3, 3, 7, 7, "RegionB");
    
    // Create union
    auto unionRegion = createUnionRegion(regionA, regionB, "UnionRegion");
    
    // Check union properties
    EXPECT_EQ(unionRegion.getDefinition()->getName(), "UnionRegion");
    
    // Check cells in various locations
    auto cellA = mockMesh->getTypedCell(2, 2);     // In A only
    auto cellB = mockMesh->getTypedCell(6, 6);     // In B only
    auto cellBoth = mockMesh->getTypedCell(4, 4);  // In both
    auto cellNeither = mockMesh->getTypedCell(8, 8); // In neither
    
    EXPECT_TRUE(unionRegion.contains(cellA));
    EXPECT_TRUE(unionRegion.contains(cellB));
    EXPECT_TRUE(unionRegion.contains(cellBoth));
    EXPECT_FALSE(unionRegion.contains(cellNeither));
}

/**
 * @brief Test region intersection operation
 */
TEST_F(RegionTest, RegionIntersectionOperation) {
    // Create two overlapping rectangular regions
    auto regionA = createRectangularRegion(*mockMesh, 1, 1, 5, 5, "RegionA");
    auto regionB = createRectangularRegion(*mockMesh, 3, 3, 7, 7, "RegionB");
    
    // Create intersection
    auto intersectionRegion = createIntersectionRegion(regionA, regionB, "IntersectionRegion");
    
    // Check intersection properties
    EXPECT_EQ(intersectionRegion.getDefinition()->getName(), "IntersectionRegion");
    
    // Check cells in various locations
    auto cellA = mockMesh->getTypedCell(2, 2);     // In A only
    auto cellB = mockMesh->getTypedCell(6, 6);     // In B only
    auto cellBoth = mockMesh->getTypedCell(4, 4);  // In both
    auto cellNeither = mockMesh->getTypedCell(8, 8); // In neither
    
    EXPECT_FALSE(intersectionRegion.contains(cellA));
    EXPECT_FALSE(intersectionRegion.contains(cellB));
    EXPECT_TRUE(intersectionRegion.contains(cellBoth));
    EXPECT_FALSE(intersectionRegion.contains(cellNeither));
}

/**
 * @brief Test region difference operation
 */
TEST_F(RegionTest, RegionDifferenceOperation) {
    // Create two overlapping rectangular regions
    auto regionA = createRectangularRegion(*mockMesh, 1, 1, 5, 5, "RegionA");
    auto regionB = createRectangularRegion(*mockMesh, 3, 3, 7, 7, "RegionB");
    
    // Create difference A - B
    auto differenceRegion = createDifferenceRegion(regionA, regionB, "DifferenceRegion");
    
    // Check difference properties
    EXPECT_EQ(differenceRegion.getDefinition()->getName(), "DifferenceRegion");
    
    // Check cells in various locations
    auto cellA = mockMesh->getTypedCell(2, 2);     // In A only
    auto cellB = mockMesh->getTypedCell(6, 6);     // In B only
    auto cellBoth = mockMesh->getTypedCell(4, 4);  // In both
    auto cellNeither = mockMesh->getTypedCell(8, 8); // In neither
    
    EXPECT_TRUE(differenceRegion.contains(cellA));
    EXPECT_FALSE(differenceRegion.contains(cellB));
    EXPECT_FALSE(differenceRegion.contains(cellBoth));
    EXPECT_FALSE(differenceRegion.contains(cellNeither));
}

/**
 * @brief Test region predicate creation
 */
TEST_F(RegionTest, RegionPredicateCreation) {
    // Create a region with cells in group 1
    auto region = filterMesh(
        *mockMesh,
        [](const Cell* cell) { 
            // Need to dynamic_cast to access group field from MockCell
            const MockCell* mockCell = dynamic_cast<const MockCell*>(cell);
            return mockCell && mockCell->group == 1; 
        },
        "Group1Region"
    );
    
    // Check that all group 1 cells are in the region
    auto group1Cells = mockMesh->getCellsByGroup(1);
    for (auto cell : group1Cells) {
        EXPECT_TRUE(region.contains(cell));
    }
    
    // Check that all other cells are not in the region
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            auto cell = mockMesh->getTypedCell(i, j);
            if (cell && cell->group != 1) {
                EXPECT_FALSE(region.contains(cell));
            }
        }
    }
}

/**
 * @brief Test region iteration functions
 */
TEST_F(RegionTest, RegionIteration) {
    // Create a region with cells in group 1
    auto region = filterMesh(
        *mockMesh,
        [](const Cell* cell) { 
            const MockCell* mockCell = dynamic_cast<const MockCell*>(cell);
            return mockCell && mockCell->group == 1; 
        },
        "Group1Region"
    );
    
    // Reset access tracking
    mockMesh->resetAllCellAccess();
    
    // Iterate through all cells in the region
    forEachCellInRegion<MockMesh, MockCell>(
        *mockMesh,
        region,
        [](MockCell* cell) { cell->markAccessed(); }
    );
    
    // Check that all group 1 cells were accessed
    auto group1Cells = mockMesh->getCellsByGroup(1);
    for (auto cell : group1Cells) {
        EXPECT_TRUE(cell->wasAccessed());
    }
    
    // Check that the access count matches expected size
    EXPECT_EQ(mockMesh->countAccessedCells(), group1Cells.size());
}

/**
 * @brief Parameterized test for regions of different sizes
 */
class RegionSizeTest : public RegionTest, 
                       public ::testing::WithParamInterface<std::tuple<int, int, int, int>> {
protected:
    // Parameters: minI, minJ, maxI, maxJ
};

TEST_P(RegionSizeTest, RegionSizeCalculation) {
    auto [minI, minJ, maxI, maxJ] = GetParam();
    
    auto region = createRectangularRegion(*mockMesh, minI, minJ, maxI, maxJ, "SizedRegion");
    region.forceOptimization();
    
    // Clamp indices to valid range for size calculation
    int clampedMinI = std::max(0, minI);
    int clampedMinJ = std::max(0, minJ);
    int clampedMaxI = std::min(mockMesh->nx() - 1, maxI);
    int clampedMaxJ = std::min(mockMesh->ny() - 1, maxJ);
    
    // Calculate expected size
    int expectedWidth = clampedMaxI - clampedMinI + 1;
    int expectedHeight = clampedMaxJ - clampedMinJ + 1;
    int expectedSize = (expectedWidth > 0 && expectedHeight > 0) ? expectedWidth * expectedHeight : 0;
    
    EXPECT_EQ(region.size(), expectedSize);
}

// Test with different region sizes
INSTANTIATE_TEST_SUITE_P(
    VariousSizes,
    RegionSizeTest,
    ::testing::Values(
        std::make_tuple(0, 0, 9, 9),     // Full mesh
        std::make_tuple(2, 2, 7, 7),     // Medium region
        std::make_tuple(4, 4, 5, 5),     // Small region
        std::make_tuple(0, 0, 0, 9),     // Left edge
        std::make_tuple(0, 0, 9, 0),     // Top edge
        std::make_tuple(-1, -1, 5, 5),   // Partial off-mesh (negative)
        std::make_tuple(5, 5, 15, 15)    // Partial off-mesh (beyond)
    )
);

/**
 * @brief Test region with boundary cells
 */
TEST_F(RegionTest, RegionBoundaryCreation) {
    // Create a boundary region
    auto boundaryRegion = createBoundaryRegion<MockMesh, MockCell>(*mockMesh, "BoundaryRegion");
    
    // Check boundary cells
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            auto cell = mockMesh->getTypedCell(i, j);
            bool isBoundary = (i == 0 || j == 0 || i == mockMesh->nx() - 1 || j == mockMesh->ny() - 1);
            
            if (isBoundary) {
                EXPECT_TRUE(boundaryRegion.contains(cell)) << "Cell at " << i << "," << j << " should be in boundary";
            } else {
                EXPECT_FALSE(boundaryRegion.contains(cell)) << "Cell at " << i << "," << j << " should not be in boundary";
            }
        }
    }
    
    // Expected size is 2*nx + 2*(ny-2) = 2*(nx+ny-2)
    int expectedSize = 2 * (mockMesh->nx() + mockMesh->ny() - 2);
    EXPECT_EQ(boundaryRegion.size(), expectedSize);
}

/**
 * @brief Test region with interior cells
 */
TEST_F(RegionTest, RegionInteriorCreation) {
    // Create an interior region
    auto interiorRegion = createInteriorRegion<MockMesh, MockCell>(*mockMesh, "InteriorRegion");
    
    // Check interior cells
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            auto cell = mockMesh->getTypedCell(i, j);
            bool isInterior = (i > 0 && j > 0 && i < mockMesh->nx() - 1 && j < mockMesh->ny() - 1);
            
            if (isInterior) {
                EXPECT_TRUE(interiorRegion.contains(cell)) << "Cell at " << i << "," << j << " should be interior";
            } else {
                EXPECT_FALSE(interiorRegion.contains(cell)) << "Cell at " << i << "," << j << " should not be interior";
            }
        }
    }
    
    // Expected size is (nx-2) * (ny-2)
    int expectedSize = (mockMesh->nx() - 2) * (mockMesh->ny() - 2);
    EXPECT_EQ(interiorRegion.size(), expectedSize);
}

/**
 * @brief Test region cell modification operations
 */
TEST_F(RegionTest, RegionCellModification) {
    // Create an empty region
    auto region = filterMesh(
        *mockMesh,
        [](const Cell* /* cell */) { return false; }, // No cells initially
        "ModifiableRegion"
    );
    
    // Verify it's empty
    EXPECT_EQ(region.size(), 0);
    
    // Add a cell
    auto cell1 = mockMesh->getTypedCell(1, 1);
    region.addCell(cell1);
    
    // Verify the cell was added
    EXPECT_TRUE(region.contains(cell1));
    EXPECT_EQ(region.size(), 1);
    
    // Add another cell
    auto cell2 = mockMesh->getTypedCell(2, 2);
    region.addCell(cell2);
    
    // Verify both cells are present
    EXPECT_TRUE(region.contains(cell1));
    EXPECT_TRUE(region.contains(cell2));
    EXPECT_EQ(region.size(), 2);
    
    // Remove the first cell
    region.removeCell(cell1);
    
    // Verify the state
    EXPECT_FALSE(region.contains(cell1));
    EXPECT_TRUE(region.contains(cell2));
    EXPECT_EQ(region.size(), 1);
    
    // Clear the region
    region.clear();
    
    // Verify it's empty again
    EXPECT_FALSE(region.contains(cell2));
    EXPECT_EQ(region.size(), 0);
}

/**
 * @brief Test region optimization based on density
 */
TEST_F(RegionTest, RegionOptimization) {
    // Create a small region (should be sparse)
    auto sparseRegion = createRectangularRegion(*mockMesh, 1, 1, 2, 2, "SparseRegion");
    sparseRegion.forceOptimization();
    
    // Small regions should typically use CELL_SET for efficiency
    EXPECT_EQ(sparseRegion.getStorageMode(), RegionStorageMode::CELL_SET);
    
    // Create a large region (should be dense)
    auto denseRegion = createRectangularRegion(*mockMesh, 0, 0, 9, 9, "DenseRegion");
    denseRegion.forceOptimization();
    
    // Large regions should typically use BIT_ARRAY for efficiency
    EXPECT_EQ(denseRegion.getStorageMode(), RegionStorageMode::BIT_ARRAY);
}

/**
 * @brief Test region rebinding to a different mesh
 */
TEST_F(RegionTest, RegionMeshRebinding) {
    // Create a region
    auto region = createRectangularRegion(*mockMesh, 1, 1, 5, 5, "OriginalRegion");
    region.forceOptimization();
    
    // Create a second mesh
    auto secondMesh = std::make_unique<MockMesh>(8, 8);
    
    // Verify the region works with the original mesh
    auto cell1 = mockMesh->getTypedCell(2, 2);
    EXPECT_TRUE(region.contains(cell1));
    
    // Rebind to the second mesh
    region.rebindToMesh(secondMesh.get());
    
    // Verify the region now works with the second mesh
    auto cell2 = secondMesh->getTypedCell(2, 2);
    EXPECT_TRUE(region.contains(cell2));
    
    // Verify size is correct with new mesh (smaller in this case)
    EXPECT_EQ(region.size(), 25); // 5x5 region
}

} // namespace mesh
