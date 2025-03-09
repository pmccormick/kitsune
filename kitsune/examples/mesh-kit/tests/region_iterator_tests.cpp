/**
 * @file RegionAccessorIteratorTests.cpp
 * @brief Unit tests for the RegionAccessor iterator implementations
 * 
 * This file contains unit tests for the region accessor iterator functionality,
 * focusing on region-based cell traversal patterns.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "RegionAccessor.h"
#include "AccessorIterators.h"
#include "Field.h"
#include <set>

// Forward declaration
class TestMesh;

// Simplified test cell that supports the necessary interfaces
class TestCell : public CellBase {
public:
    // Default constructor required by std::vector
    TestCell() : CellBase(0, 0, nullptr) {}
    
    TestCell(int i, int j, MeshBase* mesh) 
        : CellBase(i, j, mesh) {}
    
    TestMesh* getMesh() const;
    
    bool isVisited() const;
    
    void markVisited();
    
    // Helper for region testing
    RegionMask getRegionBit() const;
};

// We'll need a simple mesh implementation for testing
class TestMesh : public Mesh<TestCell> {
public:
    TestMesh(int nx, int ny) : 
        Mesh<TestCell>(nx, ny),
        m_temperatureField(nx, ny)
    {
        // Initialize fields for testing
        m_temperatureField = createField<double, CellCenterTag>(0, "temperature", 0.0);
        
        // Setup some test regions
        setupTestRegions();
    }
    
    // Override to return TestCell* instead of CellBase*
    TestCell* getCell(int i, int j) override {
        return static_cast<TestCell*>(Mesh<TestCell>::getCell(i, j));
    }
    
    CellCenterField<double>& temperatureField() { return m_temperatureField; }
    
    void markCellVisited(int i, int j) {
        if (i >= 0 && i < nx() && j >= 0 && j < ny()) {
            m_temperatureField(i, j) = 1.0;
        }
    }
    
    bool isCellVisited(int i, int j) const {
        if (i >= 0 && i < nx() && j >= 0 && j < ny()) {
            return m_temperatureField(i, j) > 0.0;
        }
        return false;
    }
    
    // Helper method to count visited cells
    int countVisitedCells() const {
        int count = 0;
        for (int j = 0; j < ny(); ++j) {
            for (int i = 0; i < nx(); ++i) {
                if (isCellVisited(i, j)) {
                    count++;
                }
            }
        }
        return count;
    }
    
    // Reset visitation status
    void resetVisitation() {
        for (int j = 0; j < ny(); ++j) {
            for (int i = 0; i < nx(); ++i) {
                m_temperatureField(i, j) = 0.0;
            }
        }
    }
    
    // Setup predefined regions for testing
    void setupTestRegions() {
        // Define region IDs
        m_leftRegionID = 1;   // Left half of the mesh
        m_rightRegionID = 2;  // Right half of the mesh
        m_topRegionID = 3;    // Top half of the mesh
        m_bottomRegionID = 4; // Bottom half of the mesh
        
        // Create region masks
        m_leftRegionMask = 0x1;    // 0001
        m_rightRegionMask = 0x2;   // 0010
        m_topRegionMask = 0x4;     // 0100
        m_bottomRegionMask = 0x8;  // 1000
        
        // Register regions
        m_regionMasks[m_leftRegionID] = m_leftRegionMask;
        m_regionMasks[m_rightRegionID] = m_rightRegionMask;
        m_regionMasks[m_topRegionID] = m_topRegionMask;
        m_regionMasks[m_bottomRegionID] = m_bottomRegionMask;
    }
    
    // Get region mask for a cell
    RegionMask getCellRegionMask(int i, int j) const {
        RegionMask mask = 0;
        
        // Left region: cells in left half
        if (i < nx() / 2) {
            mask |= m_leftRegionMask;
        }
        
        // Right region: cells in right half
        if (i >= nx() / 2) {
            mask |= m_rightRegionMask;
        }
        
        // Top region: cells in top half
        if (j >= ny() / 2) {
            mask |= m_topRegionMask;
        }
        
        // Bottom region: cells in bottom half
        if (j < ny() / 2) {
            mask |= m_bottomRegionMask;
        }
        
        return mask;
    }
    
    // Implementation of virtual method from Mesh
    RegionMask getRegionMask(RegionID regionID) const  {
        auto it = m_regionMasks.find(regionID);
        if (it != m_regionMasks.end()) {
            return it->second;
        }
        return 0; // No region
    }
    
    // Get predefined region IDs
    RegionID getLeftRegionID() const { return m_leftRegionID; }
    RegionID getRightRegionID() const { return m_rightRegionID; }
    RegionID getTopRegionID() const { return m_topRegionID; }
    RegionID getBottomRegionID() const { return m_bottomRegionID; }
    
    // Get predefined region masks
    RegionMask getLeftRegionMask() const { return m_leftRegionMask; }
    RegionMask getRightRegionMask() const { return m_rightRegionMask; }
    RegionMask getTopRegionMask() const { return m_topRegionMask; }
    RegionMask getBottomRegionMask() const { return m_bottomRegionMask; }

private:
    CellCenterField<double> m_temperatureField;
    
    // Region definitions
    RegionID m_leftRegionID;
    RegionID m_rightRegionID;
    RegionID m_topRegionID;
    RegionID m_bottomRegionID;
    
    RegionMask m_leftRegionMask;
    RegionMask m_rightRegionMask;
    RegionMask m_topRegionMask;
    RegionMask m_bottomRegionMask;
    
    std::unordered_map<RegionID, RegionMask> m_regionMasks;
};

//------------------------------------------------------------------------------
// Test fixture for RegionAccessor iterator tests
//------------------------------------------------------------------------------
class RegionAccessorIteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test mesh with known dimensions
        testMesh = new TestMesh(10, 10);
        regionAccessor = new RegionAccessor<TestMesh, TestCell>(*testMesh);
    }
    
    void TearDown() override {
        delete regionAccessor;
        delete testMesh;
    }
    
    TestMesh* testMesh;
    RegionAccessor<TestMesh, TestCell>* regionAccessor;
    
    // Helper to check if a cell is in a specific region
    bool isCellInRegion(int i, int j, RegionMask regionMask) const {
        return (testMesh->getCellRegionMask(i, j) & regionMask) != 0;
    }
    
    // Helper to get the expected count of cells in a region
    int getExpectedRegionCellCount(RegionMask regionMask) const {
        int count = 0;
        for (int j = 0; j < testMesh->ny(); ++j) {
            for (int i = 0; i < testMesh->nx(); ++i) {
                if (isCellInRegion(i, j, regionMask)) {
                    count++;
                }
            }
        }
        return count;
    }
};

//------------------------------------------------------------------------------
// RegionCellIterator Tests
//------------------------------------------------------------------------------

TEST_F(RegionAccessorIteratorTest, RegionCellIteratorTraversal) {
    // Test iterating through cells in a specific region (left region)
    testMesh->resetVisitation();
    
    RegionMask regionMask = testMesh->getLeftRegionMask();
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellIterator begin(*testMesh, regionMask);
    typename Iterators::RegionCellIterator end(*testMesh, regionMask, 0, testMesh->ny());
    
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        TestCell* cell = *it;
        ASSERT_NE(cell, nullptr) << "Cell should not be null at position (" 
                                << it.i() << "," << it.j() << ")";
        
        // Check if the cell is actually in the region
        EXPECT_TRUE(isCellInRegion(it.i(), it.j(), regionMask))
            << "Cell at (" << it.i() << "," << it.j() << ") should be in the left region";
        
        // Mark the cell as visited
        cell->markVisited();
        count++;
    }
    
    // Left region should have nx/2 * ny cells
    int expectedCount = testMesh->nx() / 2 * testMesh->ny();
    EXPECT_EQ(count, expectedCount) 
        << "Iterator should visit every cell in the left region";
    
    // Verify that only cells in the region were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isCellInRegion(i, j, regionMask)) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Cell in region at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Cell not in region at (" << i << "," << j << ") should not be visited";
            }
        }
    }
}

TEST_F(RegionAccessorIteratorTest, RegionCellRange) {
    // Test range-based for loop with RegionCellRange
    testMesh->resetVisitation();
    
    RegionMask regionMask = testMesh->getTopRegionMask();
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, regionMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        
        // Check if the cell is actually in the region
        EXPECT_TRUE(isCellInRegion(cell->i(), cell->j(), regionMask))
            << "Cell at (" << cell->i() << "," << cell->j() << ") should be in the top region";
        
        cell->markVisited();
        count++;
    }
    
    // Top region should have nx * ny/2 cells
    int expectedCount = testMesh->nx() * (testMesh->ny() / 2);
    EXPECT_EQ(count, expectedCount) 
        << "Range should cover all cells in the top region";
    
    // Verify that only cells in the region were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isCellInRegion(i, j, regionMask)) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Cell in region at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Cell not in region at (" << i << "," << j << ") should not be visited";
            }
        }
    }
}

//------------------------------------------------------------------------------
// Region Combination Tests
//------------------------------------------------------------------------------

TEST_F(RegionAccessorIteratorTest, RegionUnionIteration) {
    // Test iterating through a union of regions (left OR bottom)
    testMesh->resetVisitation();
    
    RegionMask leftMask = testMesh->getLeftRegionMask();
    RegionMask bottomMask = testMesh->getBottomRegionMask();
    RegionMask combinedMask = leftMask | bottomMask;  // Union of regions
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, combinedMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        
        // Check if the cell is actually in either region
        EXPECT_TRUE(isCellInRegion(cell->i(), cell->j(), combinedMask))
            << "Cell at (" << cell->i() << "," << cell->j() 
            << ") should be in either left or bottom region";
        
        cell->markVisited();
        count++;
    }
    
    // Count cells that should be in the combined region
    int expectedCount = getExpectedRegionCellCount(combinedMask);
    EXPECT_EQ(count, expectedCount) 
        << "Range should cover all cells in the combined region";
    
    // Verify that only cells in the combined region were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isCellInRegion(i, j, combinedMask)) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Cell in combined region at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Cell not in combined region at (" << i << "," << j 
                    << ") should not be visited";
            }
        }
    }
}

TEST_F(RegionAccessorIteratorTest, RegionIntersectionIteration) {
    // Test iterating through an intersection of regions (top AND right)
    testMesh->resetVisitation();
    
    RegionMask topMask = testMesh->getTopRegionMask();
    RegionMask rightMask = testMesh->getRightRegionMask();
    RegionMask combinedMask = topMask & rightMask;  // Intersection of regions
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, combinedMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        
        // Check if the cell is actually in both regions
        EXPECT_TRUE(isCellInRegion(cell->i(), cell->j(), topMask) && 
                   isCellInRegion(cell->i(), cell->j(), rightMask))
            << "Cell at (" << cell->i() << "," << cell->j() 
            << ") should be in both top and right regions";
        
        cell->markVisited();
        count++;
    }
    
    // Top-right quadrant should have nx/2 * ny/2 cells
    int expectedCount = (testMesh->nx() / 2) * (testMesh->ny() / 2);
    EXPECT_EQ(count, expectedCount) 
        << "Range should cover all cells in the intersection region";
    
    // Verify that only cells in the intersection were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isCellInRegion(i, j, combinedMask)) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Cell in intersection at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Cell not in intersection at (" << i << "," << j 
                    << ") should not be visited";
            }
        }
    }
}

TEST_F(RegionAccessorIteratorTest, RegionDifferenceIteration) {
    // Test iterating through a difference of regions (left - bottom)
    testMesh->resetVisitation();
    
    RegionMask leftMask = testMesh->getLeftRegionMask();
    RegionMask bottomMask = testMesh->getBottomRegionMask();
    RegionMask differenceMask = leftMask & (~bottomMask);  // Left but not bottom
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, differenceMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        
        // Check if the cell is in left but not bottom region
        EXPECT_TRUE(isCellInRegion(cell->i(), cell->j(), leftMask) && 
                   !isCellInRegion(cell->i(), cell->j(), bottomMask))
            << "Cell at (" << cell->i() << "," << cell->j() 
            << ") should be in left but not bottom region";
        
        cell->markVisited();
        count++;
    }
    
    // Calculate expected count: left-top quadrant
    int expectedCount = (testMesh->nx() / 2) * (testMesh->ny() / 2);
    EXPECT_EQ(count, expectedCount) 
        << "Range should cover all cells in the difference region";
    
    // Verify that only cells in the difference were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            bool shouldBeVisited = isCellInRegion(i, j, leftMask) && 
                                  !isCellInRegion(i, j, bottomMask);
            
            if (shouldBeVisited) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Cell in difference at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Cell not in difference at (" << i << "," << j 
                    << ") should not be visited";
            }
        }
    }
}

//------------------------------------------------------------------------------
// Dynamic Region Creation Tests
//------------------------------------------------------------------------------

TEST_F(RegionAccessorIteratorTest, CustomRegionCreation) {
    // Test creating a custom region with a predicate function
    testMesh->resetVisitation();
    
    // Define a diagonal region predicate
    auto diagonalPredicate = [](const TestCell* cell) -> bool {
        return cell->i() == cell->j();  // Only cells on the main diagonal
    };
    
    // Create the region mask using the RegionAccessor
    RegionMask diagonalMask = regionAccessor->createMask(diagonalPredicate);
    
    // Iterate through the custom region
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, diagonalMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        
        // Check if the cell is actually on the diagonal
        EXPECT_EQ(cell->i(), cell->j())
            << "Cell at (" << cell->i() << "," << cell->j() 
            << ") should be on the diagonal";
        
        cell->markVisited();
        count++;
    }
    
    // Diagonal should have min(nx, ny) cells
    int expectedCount = std::min(testMesh->nx(), testMesh->ny());
    EXPECT_EQ(count, expectedCount) 
        << "Range should cover all cells on the diagonal";
    
    // Verify that only diagonal cells were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (i == j) {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Diagonal cell at (" << i << "," << j << ") should be visited";
            } else {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Non-diagonal cell at (" << i << "," << j << ") should not be visited";
            }
        }
    }
}

//------------------------------------------------------------------------------
// Edge Case Tests
//------------------------------------------------------------------------------

TEST_F(RegionAccessorIteratorTest, EmptyRegionIteration) {
    // Test iterating through an empty region
    testMesh->resetVisitation();
    
    // Create an empty region mask
    RegionMask emptyMask = 0;
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, emptyMask);
    
    size_t count = 0;
    for (auto&& _ : range) {
        count++;
    }
    
    EXPECT_EQ(count, 0) << "Empty region should not have any cells to visit";
    
    // Verify no cells were visited
    EXPECT_EQ(testMesh->countVisitedCells(), 0)
        << "No cells should be visited for an empty region";
}

TEST_F(RegionAccessorIteratorTest, FullMeshRegionIteration) {
    // Test iterating through a region containing the entire mesh
    testMesh->resetVisitation();
    
    // Create a region mask that includes all cells
    RegionMask fullMask = testMesh->getLeftRegionMask() | testMesh->getRightRegionMask();
    
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, fullMask);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        cell->markVisited();
        count++;
    }
    
    // All cells should be visited
    EXPECT_EQ(count, testMesh->nx() * testMesh->ny()) 
        << "Range should cover all cells in the mesh";
    
    // Verify all cells were visited
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny())
        << "All cells should be visited for a full region";
}

//------------------------------------------------------------------------------
// Performance Comparison Test
//------------------------------------------------------------------------------

TEST_F(RegionAccessorIteratorTest, IteratorVsManualRegionTraversal) {
    // Test performance comparison between iterator and manual traversal
    testMesh->resetVisitation();
    
    RegionMask regionMask = testMesh->getTopRegionMask();
    
    // Manual traversal
    int manualCount = 0;
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isCellInRegion(i, j, regionMask)) {
                TestCell* cell = testMesh->getCell(i, j);
                if (cell) {
                    cell->markVisited();
                    manualCount++;
                }
            }
        }
    }
    
    // Reset for iterator traversal
    testMesh->resetVisitation();
    
    // Iterator traversal
    using Iterators = RegionAccessorIterators<TestMesh, TestCell>;
    typename Iterators::RegionCellRange range(*testMesh, regionMask);
    
    int iteratorCount = 0;
    for (auto cell : range) {
        cell->markVisited();
        iteratorCount++;
    }
    
    // Both approaches should visit the same number of cells
    EXPECT_EQ(manualCount, iteratorCount)
        << "Manual and iterator traversal should visit the same number of cells";
}

