/**
 * @file region_advanced_tests.cpp
 * @brief Advanced unit tests for the Region system
 */

#include <gtest/gtest.h>
#include <thread>
#include <future>
#include <random>
#include "RegionAll.h"
#include "./Region/MockMesh.h"
#include "./Region/MockCell.h"
#include "./Region/MockField.h"

namespace mesh {

/**
 * @class AdvancedRegionTest
 * @brief Base fixture for advanced Region system tests
 */
class AdvancedRegionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a standard 50x50 mesh for testing
        // (Larger than basic tests to better test performance characteristics)
        mockMesh = std::make_unique<MockMesh>(50, 50);
        
        // Initialize various patterns for testing
        initializePatterns();
    }
    
    void initializePatterns() {
        // Initialize pattern groups
        mockMesh->initializeGroups([](int i, int j) {
            // Checkerboard pattern
            if ((i + j) % 2 == 0) return 1;
            
            // Diagonal pattern
            if (i == j) return 2;
            
            // Cross pattern
            if (i == 25 || j == 25) return 3;
            
            // Default
            return 0;
        });
    }
    
    std::unique_ptr<MockMesh> mockMesh;
};

/**
 * @brief Test BitArray thread safety
 */
TEST_F(AdvancedRegionTest, BitArrayThreadSafety) {
    // Create a large bit array for testing
    BitArray bits(10000, false);
    
    // Set up a concurrent test with multiple threads
    constexpr int threadCount = 8;
    constexpr int opsPerThread = 1000;
    
    // Function to randomly set and clear bits
    auto threadFunction = [&bits](int seed) {
        std::mt19937 rng(seed); // Deterministic but different per thread
        std::uniform_int_distribution<size_t> indexDist(0, bits.size() - 1);
        std::uniform_int_distribution<int> valueDist(0, 1);
        
        for (int i = 0; i < opsPerThread; ++i) {
            size_t index = indexDist(rng);
            bool value = valueDist(rng) == 1;
            bits.set(index, value);
        }
    };
    
    // Start threads
    std::vector<std::future<void>> futures;
    for (int i = 0; i < threadCount; ++i) {
        futures.push_back(std::async(std::launch::async, threadFunction, i));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Verify the bit array is in a consistent state by checking that count() works
    size_t count = bits.count();
    EXPECT_LE(count, bits.size()); // Count should be valid
    
    // Check if countTrailingZeros and findFirst/findNext are consistent
    if (count > 0) {
        size_t firstBit = bits.findFirst();
        EXPECT_LT(firstBit, bits.size());
        
        // Iterate through all set bits
        size_t iteratedCount = 0;
        for (size_t idx = firstBit; idx < bits.size(); idx = bits.findNext(idx)) {
            EXPECT_TRUE(bits.get(idx));
            ++iteratedCount;
        }
        
        // The number of bits we found through iteration should match the reported count
        EXPECT_EQ(iteratedCount, count);
    }
}

/**
 * @brief Test BitArray SIMD operations (if available)
 */
TEST_F(AdvancedRegionTest, BitArraySIMDOperations) {
    // Create bit arrays with patterns designed to test SIMD operations
    constexpr size_t size = 1024; // Multiple of 256 for AVX2
    BitArray bitsA(size, false);
    BitArray bitsB(size, false);
    
    // Set alternating patterns (will exercise all SIMD paths)
    for (size_t i = 0; i < size; i += 2) bitsA.set(i, true);
    for (size_t i = 1; i < size; i += 2) bitsB.set(i, true);
    
    // All bits should be set after OR
    BitArray bitsOr = bitsA; // Make a copy
    bitsOr.bitwiseOr(bitsB);
    EXPECT_EQ(bitsOr.count(), size);
    
    // No bits should be set after AND
    BitArray bitsAnd = bitsA; // Make a copy
    bitsAnd.bitwiseAnd(bitsB);
    EXPECT_EQ(bitsAnd.count(), 0);
    
    // Original patterns should be preserved after XOR
    BitArray bitsXor = bitsA; // Make a copy
    bitsXor.bitwiseXor(bitsB);
    EXPECT_EQ(bitsXor.count(), size);
    
    // AND-NOT should preserve original patterns
    BitArray bitsAndNot = bitsA; // Make a copy
    bitsAndNot.bitwiseAndNot(bitsB);
    EXPECT_EQ(bitsAndNot.count(), bitsA.count());
}

/**
 * @brief Test Region traversal strategies
 */
TEST_F(AdvancedRegionTest, RegionTraversalStrategies) {
    // Create a rectangular region
    auto region = createRectangularRegion(*mockMesh, 10, 10, 39, 39, "TraversalTestRegion");
    
    // Number of cells should be 30x30 = 900
    EXPECT_EQ(region.size(), 900);
    
    // Test each traversal order
    for (RegionTraversalOrder order : {
        RegionTraversalOrder::NATURAL,
        RegionTraversalOrder::ROW_MAJOR,
        RegionTraversalOrder::COLUMN_MAJOR,
        RegionTraversalOrder::BLOCKED,
        RegionTraversalOrder::Z_ORDER
    }) {
        mockMesh->resetAllCellAccess();
        
        // Traverse the region with the specified order
        forEachCellInRegionOrdered<MockMesh, MockCell>(
            *mockMesh,
            region,
            [](MockCell* cell) { cell->markAccessed(); },
            order
        );
        
        // Verify all cells in the region were accessed exactly once
        EXPECT_EQ(mockMesh->countAccessedCells(), 900);
        
        // Verify that only cells in the region were accessed
        for (int j = 0; j < mockMesh->ny(); ++j) {
            for (int i = 0; i < mockMesh->nx(); ++i) {
                auto cell = mockMesh->getTypedCell(i, j);
                bool inRegion = (i >= 10 && i <= 39 && j >= 10 && j <= 39);
                
                if (inRegion) {
                    EXPECT_TRUE(cell->wasAccessed()) << "Cell at " << i << "," << j 
                        << " should have been accessed with order " << static_cast<int>(order);
                } else {
                    EXPECT_FALSE(cell->wasAccessed()) << "Cell at " << i << "," << j 
                        << " should not have been accessed with order " << static_cast<int>(order);
                }
            }
        }
    }
}

/**
 * @brief Test RegionAccessor for field operations
 */
TEST_F(AdvancedRegionTest, RegionAccessorOperations) {
    // Create a test field on the mesh
    MockField<double> temperatureField(*mockMesh, 0.0);
    
    // Set initial temperature distribution (linear gradient)
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            temperatureField(i, j) = static_cast<double>(i + j) / 10.0;
        }
    }
    
    // Create a region for the center part of the mesh
    auto centerRegion = createRectangularRegion(*mockMesh, 15, 15, 34, 34, "CenterRegion");
    
    // Create accessor for the region
    auto accessor = createRegionAccessor(*mockMesh, centerRegion);
    
    // Test getting average field value in the region
    double averageTemp = accessor.getAverageFieldValue(temperatureField);
    
    // Calculate expected average manually
    double expectedSum = 0.0;
    int count = 0;
    for (int j = 15; j <= 34; ++j) {
        for (int i = 15; i <= 34; ++i) {
            expectedSum += static_cast<double>(i + j) / 10.0;
            ++count;
        }
    }
    double expectedAverage = expectedSum / count;
    
    EXPECT_NEAR(averageTemp, expectedAverage, 1e-10);
    
    // Test setting field values using accessor
    accessor.setFieldValue(temperatureField, 100.0);
    
    // Verify values were updated only in the region
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            bool inRegion = (i >= 15 && i <= 34 && j >= 15 && j <= 34);
            
            if (inRegion) {
                EXPECT_DOUBLE_EQ(temperatureField(i, j), 100.0) 
                    << "Cell at " << i << "," << j << " should have updated temperature";
            } else {
                EXPECT_DOUBLE_EQ(temperatureField(i, j), static_cast<double>(i + j) / 10.0)
                    << "Cell at " << i << "," << j << " should have original temperature";
            }
        }
    }
    
    // Test setting field values with a function
    accessor.setFieldValues(temperatureField, [](MockCell* cell) {
        return static_cast<double>(cell->i() * cell->j());
    });
    
    // Verify values were updated according to the function
    for (int j = 15; j <= 34; ++j) {
        for (int i = 15; i <= 34; ++i) {
            EXPECT_DOUBLE_EQ(temperatureField(i, j), static_cast<double>(i * j))
                << "Cell at " << i << "," << j << " should have updated temperature based on function";
        }
    }
}

/**
 * @brief Test Region optimization strategies
 */
TEST_F(AdvancedRegionTest, RegionOptimizationStrategies) {
    // Test default optimization strategy
    {
        // Create region and explicitly set default strategy
        auto region = createRectangularRegion(*mockMesh, 10, 10, 39, 39, "DefaultStrategyRegion");
        region.setDefaultOptimizationStrategy();
        
        // Remember current mode
        auto initialMode = region.getStorageMode();
        
        // Force to CELL_SET mode
        region.setStorageMode(RegionStorageMode::CELL_SET);
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::CELL_SET);
        
        // Force optimization
        region.forceOptimization();
        
        // For a region this size (900 cells in 2500-cell mesh = 36% density)
        // Default strategy should choose BIT_ARRAY for dense regions
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::BIT_ARRAY);
    }
    
    // Test simulation optimization strategy
    {
        // Create region and explicitly set simulation strategy
        auto region = createRectangularRegion(*mockMesh, 10, 10, 39, 39, "SimStrategyRegion");
        region.setSimulationOptimizationStrategy();
        
        // Force to CELL_SET mode
        region.setStorageMode(RegionStorageMode::CELL_SET);
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::CELL_SET);
        
        // Force optimization
        region.forceOptimization();
        
        // Simulation strategy prefers BIT_ARRAY at even lower densities
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::BIT_ARRAY);
    }
    
    // Test custom optimization strategy
    {
        // Create region
        auto region = createRectangularRegion(*mockMesh, 10, 10, 39, 39, "CustomStrategyRegion");
        
        // Set custom strategy that always chooses CELL_SET
        region.setOptimizationStrategy(
            // Always optimize
            [](const Region& region, size_t operationCount) { return true; },
            // Always choose CELL_SET
            [](const Region& region, RegionStorageMode currentMode, double threshold) {
                return RegionStorageMode::CELL_SET;
            }
        );
        
        // Force to BIT_ARRAY mode
        region.setStorageMode(RegionStorageMode::BIT_ARRAY);
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::BIT_ARRAY);
        
        // Force optimization
        region.forceOptimization();
        
        // Custom strategy should always choose CELL_SET
        EXPECT_EQ(region.getStorageMode(), RegionStorageMode::CELL_SET);
    }
}

/**
 * @brief Test region dilate operation
 */
TEST_F(AdvancedRegionTest, RegionDilateOperation) {
    // Only run this test if the dilate function exists
    if (!std::is_member_function_pointer<decltype(&Region::template dilateByDistance<units::meter>)>::value) {
        GTEST_SKIP() << "Region::dilateByDistance is not implemented";
    }
    
    // Create a small central region
    auto centralRegion = createRectangularRegion(*mockMesh, 23, 23, 26, 26, "CentralRegion");
    
    // Dilate by 1 cell width
    units::meter cellSize(1.0); // Assuming 1m cell size for the mock mesh
    auto dilatedRegion = centralRegion.dilateByDistance(cellSize);
    
    // Check that the dilated region includes the original region plus a 1-cell border
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            auto cell = mockMesh->getTypedCell(i, j);
            
            bool inOriginal = (i >= 23 && i <= 26 && j >= 23 && j <= 26);
            bool inDilated = (i >= 22 && i <= 27 && j >= 22 && j <= 27);
            
            if (inOriginal) {
                EXPECT_TRUE(centralRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should be in original region";
                EXPECT_TRUE(dilatedRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should be in dilated region";
            } else if (inDilated) {
                EXPECT_FALSE(centralRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should not be in original region";
                EXPECT_TRUE(dilatedRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should be in dilated region";
            } else {
                EXPECT_FALSE(centralRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should not be in original region";
                EXPECT_FALSE(dilatedRegion.contains(cell)) 
                    << "Cell at " << i << "," << j << " should not be in dilated region";
            }
        }
    }
}

/**
 * @brief Test region concurrent modification
 */
TEST_F(AdvancedRegionTest, RegionConcurrentModification) {
    // Create a region to test
    auto region = createRectangularRegion(*mockMesh, 0, 0, 49, 49, "ConcurrentModRegion");
    
    // Set up concurrent modification with multiple threads
    constexpr int threadCount = 4;
    constexpr int cellsPerThread = 100;
    
    // Function to randomly add and remove cells
    auto threadFunction = [this, &region](int seed) {
        std::mt19937 rng(seed); // Deterministic but different per thread
        std::uniform_int_distribution<int> indexDist(0, mockMesh->nx() - 1);
        
        for (int i = 0; i < cellsPerThread; ++i) {
            int x = indexDist(rng);
            int y = indexDist(rng);
            auto cell = mockMesh->getTypedCell(x, y);
            
            // Toggle cell membership
            if (region.contains(cell)) {
                region.removeCell(cell);
            } else {
                region.addCell(cell);
            }
        }
    };
    
    // Start threads
    std::vector<std::future<void>> futures;
    for (int i = 0; i < threadCount; ++i) {
        futures.push_back(std::async(std::launch::async, threadFunction, i));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    // Verify the region is in a consistent state
    size_t size = region.size();
    EXPECT_LE(size, mockMesh->nx() * mockMesh->ny());
    
    // Count cells manually to verify consistency
    size_t manualCount = 0;
    for (int j = 0; j < mockMesh->ny(); ++j) {
        for (int i = 0; i < mockMesh->nx(); ++i) {
            auto cell = mockMesh->getTypedCell(i, j);
            if (region.contains(cell)) {
                ++manualCount;
            }
        }
    }
    
    EXPECT_EQ(manualCount, size);
}

} // namespace mesh
