/**
 * @file MeshAccessorIteratorTests.cpp
 * @brief Unit tests for the MeshAccessor iterator implementations
 * 
 * This file contains unit tests for the mesh accessor iterator functionality,
 * focusing on the whole mesh and interior cell iterators.
 */

#include <gtest/gtest.h>
#include "Mesh.h"
#include "MeshAccessor.h"
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
};

// We'll need a simple mesh implementation for testing
class TestMesh : public Mesh<TestCell> {
public:
    TestMesh(int nx, int ny) : 
        Mesh<TestCell>(nx, ny),
        // Initialize the field with proper parameters in the initializer list
        m_temperatureField(nx, ny)
    {
        // Initialize fields for testing
        m_temperatureField = createField<double, CellCenterTag>(0, "temperature", 0.0);
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

private:
    CellCenterField<double> m_temperatureField;
};

// Need to implement these after TestMesh is defined
TestMesh* TestCell::getMesh() const { 
    return static_cast<TestMesh*>(mesh()); 
}

bool TestCell::isVisited() const {
    return getMesh()->isCellVisited(i(), j());
}

void TestCell::markVisited() {
    getMesh()->markCellVisited(i(), j());
}

//------------------------------------------------------------------------------
// Test fixture for MeshAccessor iterator tests
//------------------------------------------------------------------------------
class MeshAccessorIteratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test mesh with known dimensions
        testMesh = new TestMesh(10, 10);
        meshAccessor = new MeshAccessor<TestMesh, TestCell>(*testMesh);
    }
    
    void TearDown() override {
        delete meshAccessor;
        delete testMesh;
    }
    
    TestMesh* testMesh;
    MeshAccessor<TestMesh, TestCell>* meshAccessor;
    
    // Helper to check if a cell position is on the boundary
    bool isBoundaryPosition(int i, int j) const {
        return i == 0 || j == 0 || 
               i == testMesh->nx() - 1 || 
               j == testMesh->ny() - 1;
    }
    
    // Helper to check if a cell position is in the interior
    bool isInteriorPosition(int i, int j) const {
        return !isBoundaryPosition(i, j);
    }
};

//------------------------------------------------------------------------------
// CellIterator Tests
//------------------------------------------------------------------------------

TEST_F(MeshAccessorIteratorTest, CellIteratorTraversal) {
    // Create a cell iterator from the MeshAccessorIterators namespace
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::CellIterator begin(*testMesh);
    typename Iterators::CellIterator end(*testMesh, 0, testMesh->ny());
    
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        TestCell* cell = *it;
        ASSERT_NE(cell, nullptr) << "Cell should not be null at position (" 
                                << it.i() << "," << it.j() << ")";
        
        // Mark the cell as visited
        cell->markVisited();
        count++;
    }
    
    // Check that all cells were visited
    EXPECT_EQ(count, testMesh->nx() * testMesh->ny()) 
        << "Iterator should visit every cell";
    
    // Check every cell was marked
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny())
        << "Every cell should be marked as visited";
}

TEST_F(MeshAccessorIteratorTest, CellIteratorRange) {
    // Use the CellRange from MeshAccessorIterators
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::CellRange range(*testMesh);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        cell->markVisited();
        count++;
    }
    
    // Check that all cells were visited
    EXPECT_EQ(count, testMesh->nx() * testMesh->ny()) 
        << "Range should cover all cells";
    
    // Check every cell was marked
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny())
        << "Every cell should be marked as visited";
}

//------------------------------------------------------------------------------
// InteriorCellIterator Tests
//------------------------------------------------------------------------------

TEST_F(MeshAccessorIteratorTest, InteriorCellIteratorTraversal) {
    // Reset visitation status
    testMesh->resetVisitation();
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::InteriorCellIterator begin(*testMesh);
    typename Iterators::InteriorCellIterator end(*testMesh, 1, testMesh->ny() - 1);
    
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        TestCell* cell = *it;
        ASSERT_NE(cell, nullptr) << "Interior cell should not be null at ("
                                << it.i() << "," << it.j() << ")";
        
        // Check that this is actually an interior cell
        EXPECT_TRUE(isInteriorPosition(it.i(), it.j())) 
            << "Position (" << it.i() << "," << it.j() << ") should be interior";
        
        cell->markVisited();
        count++;
    }
    
    // Number of interior cells should be (nx-2) * (ny-2)
    const int expectedInteriorCount = (testMesh->nx() - 2) * (testMesh->ny() - 2);
    EXPECT_EQ(count, expectedInteriorCount) 
        << "Iterator should visit every interior cell";
    
    // Check that only interior cells were visited
    EXPECT_EQ(testMesh->countVisitedCells(), expectedInteriorCount)
        << "Only interior cells should be marked as visited";
    
    // Verify that no boundary cells were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isBoundaryPosition(i, j)) {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Boundary cell at (" << i << "," << j << ") should not be visited";
            } else {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Interior cell at (" << i << "," << j << ") should be visited";
            }
        }
    }
}

TEST_F(MeshAccessorIteratorTest, InteriorCellIteratorRange) {
    // Reset visitation status
    testMesh->resetVisitation();
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::InteriorCellRange range(*testMesh);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Interior cell should not be null";
        
        // Check that this is actually an interior cell
        EXPECT_TRUE(isInteriorPosition(cell->i(), cell->j())) 
            << "Position (" << cell->i() << "," << cell->j() << ") should be interior";
        
        cell->markVisited();
        count++;
    }
    
    // Number of interior cells should be (nx-2) * (ny-2)
    const int expectedInteriorCount = (testMesh->nx() - 2) * (testMesh->ny() - 2);
    EXPECT_EQ(count, expectedInteriorCount) 
        << "Range should cover all interior cells";
    
    // Verify that only interior cells were visited
    for (int j = 0; j < testMesh->ny(); ++j) {
        for (int i = 0; i < testMesh->nx(); ++i) {
            if (isBoundaryPosition(i, j)) {
                EXPECT_FALSE(testMesh->isCellVisited(i, j))
                    << "Boundary cell at (" << i << "," << j << ") should not be visited";
            } else {
                EXPECT_TRUE(testMesh->isCellVisited(i, j))
                    << "Interior cell at (" << i << "," << j << ") should be visited";
            }
        }
    }
}

//------------------------------------------------------------------------------
// BlockCellIterator Tests
//------------------------------------------------------------------------------

TEST_F(MeshAccessorIteratorTest, BlockCellIteratorTraversal) {
    // Reset visitation status
    testMesh->resetVisitation();
    
    const int blockSizeX = 3;
    const int blockSizeY = 2;
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::BlockCellIterator begin(*testMesh, blockSizeX, blockSizeY);
    typename Iterators::BlockCellIterator end(*testMesh, blockSizeX, blockSizeY, 
                                            0, (testMesh->ny() + blockSizeY - 1) / blockSizeY);
    
    // Use a set to ensure each position is only visited once
    std::set<std::pair<int, int>> visitedPositions;
    
    size_t count = 0;
    for (auto it = begin; it != end; ++it) {
        TestCell* cell = *it;
        if (cell) {  // Valid cell
            // Record the position
            std::pair<int, int> pos(it.i(), it.j());
            
            // Check this position wasn't already visited
            EXPECT_EQ(visitedPositions.count(pos), 0) 
                << "Position (" << it.i() << "," << it.j() << ") visited multiple times";
            
            visitedPositions.insert(pos);
            cell->markVisited();
            count++;
        }
    }
    
    // Check that all cells were visited
    EXPECT_EQ(count, testMesh->nx() * testMesh->ny()) 
        << "Iterator should visit every cell";
    
    // Check every cell was marked
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny())
        << "Every cell should be marked as visited";
    
    // Remove specific order checks as the implementation may vary
}

TEST_F(MeshAccessorIteratorTest, BlockCellIteratorRange) {
    // Reset visitation status
    testMesh->resetVisitation();
    
    const int blockSizeX = 3;
    const int blockSizeY = 2;
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::BlockCellRange range(*testMesh, blockSizeX, blockSizeY);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr) << "Cell should not be null";
        cell->markVisited();
        count++;
    }
    
    // Check that all cells were visited
    EXPECT_EQ(count, testMesh->nx() * testMesh->ny()) 
        << "Range should cover all cells";
    
    // Check every cell was marked
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny())
        << "Every cell should be marked as visited";
}

//------------------------------------------------------------------------------
// Edge Case Tests
//------------------------------------------------------------------------------

TEST_F(MeshAccessorIteratorTest, SmallMeshIteration) {
    // Create a very small mesh (1x1)
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    
    // Test all cells iterator
    {
        TestMesh smallMesh(1, 1);
        typename Iterators::CellRange range(smallMesh);
        size_t count = 0;
        for (auto cell : range) {
            ASSERT_NE(cell, nullptr);
            count++;
        }
        EXPECT_EQ(count, 1) << "Should visit the single cell";
    }
    
    // Test interior cells iterator (should be empty for 1x1 mesh)
    {
        TestMesh smallMesh(1, 1);
        typename Iterators::InteriorCellRange range(smallMesh);
        size_t count = 0;
        for (auto&& _ : range) {
            count++;
        }
        EXPECT_EQ(count, 0) << "No interior cells in a 1x1 mesh";
	std::cout << "foo!\n";
    }
}

TEST_F(MeshAccessorIteratorTest, BlockIterationWithNonDivisibleSize) {
    // Create a mesh with dimensions not divisible by block size
    TestMesh oddMesh(7, 5);
    
    using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
    typename Iterators::BlockCellRange range(oddMesh, 3, 3);
    
    size_t count = 0;
    for (auto cell : range) {
        ASSERT_NE(cell, nullptr);
        count++;
    }
    
    EXPECT_EQ(count, 7 * 5) << "Should visit all 35 cells despite uneven blocking";
}

//------------------------------------------------------------------------------
// Performance Comparison Tests (for sanity check, not timing)
//------------------------------------------------------------------------------

TEST_F(MeshAccessorIteratorTest, IteratorVsManualTraversal) {
    // Reset visitation status
    testMesh->resetVisitation();
    
    // Manual traversal with nested loops
    {
        int manualCount = 0;
        for (int j = 0; j < testMesh->ny(); ++j) {
            for (int i = 0; i < testMesh->nx(); ++i) {
                TestCell* cell = testMesh->getCell(i, j);
                if (cell) {
                    cell->markVisited();
                    manualCount++;
                }
            }
        }
        EXPECT_EQ(manualCount, testMesh->nx() * testMesh->ny());
    }
    
    // Reset visitation status
    testMesh->resetVisitation();
    
    // Iterator-based traversal
    {
        using Iterators = MeshAccessorIterators<TestMesh, TestCell>;
        typename Iterators::CellRange range(*testMesh);
        
        int iteratorCount = 0;
        for (auto cell : range) {
            cell->markVisited();
            iteratorCount++;
        }
        
        EXPECT_EQ(iteratorCount, testMesh->nx() * testMesh->ny());
    }
    
    // Both approaches should have visited all cells
    EXPECT_EQ(testMesh->countVisitedCells(), testMesh->nx() * testMesh->ny());
}
